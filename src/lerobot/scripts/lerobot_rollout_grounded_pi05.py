# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live grounded inference + data harvest for a grasp-then-raw pi05 policy.

Runs ``Atabaku/pi05_strawberry_fulltask`` on the real SO-101, reproducing the
training-time visual prompt LIVE: during the grasp phase the scene camera frame
is tinted with the deep azure-blue strawberry mask (SAM3, single-image API);
once the berry is grasped (the mask is lost for a few chunks) the overlay drops
and placement proceeds on raw frames — matching the
``so101-strawberry-grasp-rawplace-grounded`` dataset.

pi05 runs open-loop action chunks: a fresh observation is only consumed when the
policy's action queue empties (every ``n_action_steps`` ticks). So SAM runs ONCE
PER CHUNK (gated on an empty queue), not every control tick.

Two modes:
  --smoke   : no robot. Pull frames from a raw dataset, lock-on + track + overlay,
              run predict_action_chunk, dump QC images + latency. Validates the
              whole SAM->overlay->policy path and the overlay/distribution match.
  (default) : connect the SO-101 and run episodes, scoring success per episode and
              logging success rate + SAM/policy latency + achieved Hz.

Run in the ``so101-sam`` conda env (has sam3 + lerobot + pi05).
"""

import json
import logging
import select
import sys
import time
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

try:  # POSIX-only: single-keypress early-stop during live episodes
    import termios
    import tty
    _HAS_TTY = True
except ImportError:  # pragma: no cover - non-POSIX
    _HAS_TTY = False

import draccus
import numpy as np
import torch

from lerobot.grounding.live_segmenter import LiveStrawberrySegmenter
from lerobot.grounding.pipeline import grounding_overlay_frame, terminal_binary_label
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grounded_rollout")
# lerobot imports set the root logger to WARNING; keep our progress/latency at INFO,
# and surface the SAM segmenter's messages ("Loading SAM3…", "Locked target…") too.
logger.setLevel(logging.INFO)
logging.getLogger("lerobot.grounding").setLevel(logging.INFO)


def _early_stop_enter(active: bool):
    """Put stdin in cbreak mode so one keypress can end the current episode early.
    Returns saved terminal state, or None when disabled / stdin isn't a TTY (e.g.
    piped input or a backgrounded run) — early-stop is simply off in that case."""
    if not active or not _HAS_TTY:
        return None
    try:
        if not sys.stdin.isatty():
            return None
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return (fd, old)
    except Exception:  # noqa: BLE001
        return None


def _early_stop_pressed(state):
    """Return the pressed key (consuming it) if the operator hit one, else None."""
    if state is None:
        return None
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1)
    except Exception:  # noqa: BLE001
        return None
    return None


def _early_stop_restore(state) -> None:
    """Restore the terminal mode saved by _early_stop_enter (safe to call twice)."""
    if state is None:
        return
    fd, old = state
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:  # noqa: BLE001
        pass

DEFAULT_OUTPUT = "/home/summer2026/projects/lerobotprocessing/debug_grounded_rollout"


@dataclass
class GroundedRolloutConfig:
    # ── Policy + dataset (defaults match the trained fulltask model) ──────────
    policy_path: str = "Atabaku/pi05_strawberry_fulltask"
    dataset_repo_id: str = "Atabaku/so101-strawberry-grasp-rawplace-grounded"
    task: str | None = None  # None -> read the single task string from the dataset meta
    device: str = "cuda"

    # ── SAM grounding (defaults baked from the dataset's grounding metadata) ──
    sam_prompt: str = "red fruit"
    sam_confidence: float = 0.5
    scene_camera_key: str = "scene"  # the camera that was grounded (wrist stays raw)
    overlay_color: tuple[int, int, int] = (0, 60, 190)
    overlay_alpha: float = 0.35
    draw_contour: bool = True
    redetect_max_dist_frac: float = 0.08
    # Placement-phase render: "raw" (fulltask/rawplace model — no marker) or
    # "box" (fullseg/boxplace model — box+crosshair on the green tray).
    placement_style: str = "raw"

    # ── Target selection ──────────────────────────────────────────────────────
    target_mode: str = "click"  # "click" (operator picks) | "auto"
    auto_rule: str = "score"  # used when target_mode=="auto": score|central|closest_to_gripper
    lock_on_timeout_s: float = 20.0

    # ── Phase switch (grasp -> raw placement) ─────────────────────────────────
    # Default trigger is sustained mask-loss (berry occluded/grasped). Optional
    # gripper trigger is OFF unless grasp_gripper_close_thresh is set.
    mask_hold_chunks: int = 3  # reuse last good mask for up to this many transient misses
    mask_lost_chunks: int = 3  # consecutive misses that flip the phase to PLACE
    # Grasp detection (disambiguates a real grasp from the arm merely occluding the berry):
    # data shows gripper.pos ~2-3 open, ~40-50 closed-on-berry, so closed = ABOVE ~25.
    # Set thresh=None to disable and fall back to mask-loss only.
    grasp_gripper_close_thresh: float | None = 25.0
    gripper_closed_is_below: bool = False
    grasp_hold_chunks: int = 2  # gripper held closed this many chunks => grasped => switch to placement
    # Reversible switch: a FAILED grasp that also occludes the berry trips the switch
    # falsely. If the target reappears on the table (near its pre-grasp spot) for
    # `revert_chunks` chunks, revert to grasp and keep trying. A real grasp carries the
    # berry away, so it won't reappear there.
    placement_revert: bool = True
    revert_chunks: int = 2
    revert_max_dist_frac: float = 0.07  # only the target's exact spot counts (not neighbour berries)
    revert_grace_chunks: int = 2  # after switching to placement, wait this many chunks before a revert can fire (let a real grasp carry the berry away)
    # Phase-switch MODE — accuracy tuning for the grasp->placement transition:
    #   "gripper" (legacy) — switch when the gripper holds closed OR mask-loss-while-closed. Fires easily;
    #             a WEAK grasp (gripper closes but berry not actually held) falsely switches to placement,
    #             then the arm wanders to a NEW berry. Relies on revert to recover.
    #   "lift"    — switch ONLY when the berry has left the frame (mask lost = lifted/occluded) AND the
    #             gripper is held closed. A weak grasp that leaves the berry visible does NOT switch, so the
    #             mask stays on and the policy retries the SAME berry. More robust; use for BOX.
    #   "never"   — never switch; keep the target mask on for the whole episode. Best for BERRY (placement
    #             is raw anyway): a dropped berry keeps its mask so the policy re-attempts the same berry
    #             instead of drifting. (Box can't use "never" — it needs the switch to draw the tray box.)
    #             NOTE: do NOT name this "off" — draccus parses "off" as a YAML boolean -> the string "False".
    phase_switch: str = "gripper"

    # ── Control loop ──────────────────────────────────────────────────────────
    fps: float | None = None  # None -> read from dataset meta (20)
    n_action_steps: int = 50  # actions executed per chunk before re-observing (<= chunk_size)
    episodes: int = 5
    episode_duration_s: float = 60.0
    reset_gate: bool = True  # pause before each episode for operator to reset scene + arm (press ENTER)
    release_torque_on_reset: bool = True  # go limp during the reset gate so the arm can be repositioned by hand

    # ── Robot (live mode only; mirror your recording setup) ───────────────────
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so101_follower"  # calibration is stored/reused per id
    # Cameras are Intel RealSense (D435 scene / D405 wrist), addressed by serial via
    # pyrealsense2 — NOT raw /dev/video (each RealSense exposes several V4L2 nodes,
    # and the RGB stream stalls through OpenCV/V4L2).
    camera_backend: str = "realsense"  # "realsense" | "opencv"
    scene_serial: str = "346522072484"  # Intel RealSense D435 (global view)
    wrist_serial: str = "335122272701"  # Intel RealSense D405 (wrist)
    reset_realsense: bool = True  # hardware_reset the RealSense cams on startup (clears stuck state from prior crashes)
    realsense_reset_wait_s: int = 8
    # OpenCV fallback only (camera_backend=="opencv"):
    scene_camera_index: int | str = 0
    wrist_camera_index: int | str = 1
    cam_fourcc: str | None = None
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30  # camera CAPTURE rate (decoupled from the control loop fps)
    cam_warmup_s: int = 5
    use_degrees: bool = True

    # ── Output / mode ─────────────────────────────────────────────────────────
    # grounding=False: raw (non-grounded) model — NO SAM/overlay/lock-on, run raw frames
    # straight to the policy (for the raw control / full-finetune A/B models).
    grounding: bool = True
    brighten_gamma: float | None = None  # set 0.6 for the brightness model — brightens every camera frame to match its training distribution
    run_tag: str = ""  # tags the harvest log + QC frames so A/B runs don't collide
    output_dir: str = DEFAULT_OUTPUT
    dry_run: bool = False  # connect + observe + run policy but NEVER send actions (arm holds still) — for input inspection
    smoke: bool = False
    smoke_source_repo_id: str = "Atabaku/so101-strawberry-raw"  # un-overlaid frames to segment
    smoke_frames: int = 8
    save_qc_frames: int = 6  # overlay frames to save per episode for QC


# ───────────────────────────── shared helpers ──────────────────────────────


def _to_hwc_uint8(img) -> np.ndarray:
    """Coerce a dataset/robot image to HWC uint8 RGB."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 + 1e-6 else img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _load_policy_and_processors(cfg: GroundedRolloutConfig, dataset_stats):
    logger.info("Loading pi05 policy from %s ...", cfg.policy_path)
    policy = PI05Policy.from_pretrained(cfg.policy_path)
    policy.to(cfg.device)
    policy.eval()
    # Build processors fresh from the policy config + dataset stats rather than
    # loading the checkpoint's saved pipeline: the checkpoint was trained on a
    # newer lerobot whose processor step names differ from this env's registry
    # (e.g. relative_actions_processor -> delta/absolute_actions_processor). The
    # normalization stats come from the same dataset the model trained on, so
    # the rebuilt pipeline is equivalent.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=None,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    return policy, preprocessor, postprocessor


def _brighten(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma brightening, identical to brighten_dataset.py (gamma<1 brightens)."""
    return np.clip(255.0 * (img.astype(np.float32) / 255.0) ** gamma, 0, 255).astype(np.uint8)


def _detect_tray_bbox(image: np.ndarray, min_green_px: int = 5000) -> list[int] | None:
    """Green-tray bbox via colour threshold (matches the segment script's detector).
    Returns [x1,y1,x2,y2] or None when too little green is visible (caller reuses last)."""
    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)
    green = (g > 85) & (g - b > 30) & (g - r > 0)
    ys, xs = np.where(green)
    if xs.size < min_green_px:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _overlay_scene(cfg: GroundedRolloutConfig, scene_rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return grounding_overlay_frame(
        scene_rgb,
        mask,
        color=cfg.overlay_color,
        alpha=cfg.overlay_alpha,
        draw_contour=cfg.draw_contour,
        bbox_xyxy=None,
    )


def _predict_chunk(cfg, policy, preprocessor, postprocessor, frame, task, robot_type, device):
    """Run prepare -> preprocess -> predict_action_chunk on an observation frame.

    Returns (chunk_tensor, policy_ms). Used by smoke mode.
    """
    autocast_ctx = (
        torch.autocast(device_type=torch.device(device).type)
        if torch.device(device).type == "cuda" and getattr(policy.config, "use_amp", False)
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        obs = prepare_observation_for_inference(copy(frame), torch.device(device), task, robot_type)
        batch = preprocessor(obs)
        t0 = time.perf_counter()
        chunk = policy.predict_action_chunk(batch)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        policy_ms = (time.perf_counter() - t0) * 1e3
    return chunk, policy_ms


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


_DEFAULT_TASK = "pick up the highlighted strawberry and place into the green bin"


def _resolve_task(meta, cfg: GroundedRolloutConfig) -> str:
    """Resolve the policy task string. ``meta.tasks`` is a column index, not the
    task text — the human-readable instruction lives in episode metadata."""
    if cfg.task:
        return cfg.task
    try:
        tasks = meta.episodes[0]["tasks"]
        if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks):
            return str(tasks[0])
        if isinstance(tasks, str) and tasks:
            return tasks
    except Exception:
        pass
    logger.warning("Could not read task from dataset meta; using default %r", _DEFAULT_TASK)
    return _DEFAULT_TASK


# ───────────────────────────── smoke mode ──────────────────────────────────


def run_smoke(cfg: GroundedRolloutConfig) -> None:
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    out = Path(cfg.output_dir) / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)
    task = _resolve_task(meta, cfg)
    fps = cfg.fps or meta.fps
    robot_type = getattr(meta, "robot_type", "so_follower")
    logger.info("Smoke: task=%r fps=%s robot_type=%s", task, fps, robot_type)

    policy, preprocessor, postprocessor = _load_policy_and_processors(cfg, meta.stats)
    seg = LiveStrawberrySegmenter(
        cfg.sam_prompt,
        confidence_threshold=cfg.sam_confidence,
        device=cfg.device,
        redetect_max_dist_frac=cfg.redetect_max_dist_frac,
    )

    # Pull un-overlaid frames to segment from the raw control dataset.
    src = LeRobotDataset(cfg.smoke_source_repo_id)
    n = min(cfg.smoke_frames, len(src))
    logger.info("Smoke: segmenting %d frames from %s", n, cfg.smoke_source_repo_id)

    seg.reset()
    sam_ms_list, policy_ms_list = [], []
    locked = False
    for i in range(n):
        item = src[i]
        scene = _to_hwc_uint8(item[f"{OBS_STR}.images.{cfg.scene_camera_key}"])

        t0 = time.perf_counter()
        if not locked:
            locked = seg.lock_on(scene, click_xy=None, auto_rule="score")
            mask = seg.last_mask if locked else None
        else:
            mask = seg.track(scene)
        sam_ms = (time.perf_counter() - t0) * 1e3
        sam_ms_list.append(sam_ms)

        overlaid = _overlay_scene(cfg, scene, mask)

        # Build an observation frame (state + both camera images) for the policy.
        state_vec = np.asarray(item[f"{OBS_STR}.state"], dtype=np.float32)
        frame = {f"{OBS_STR}.state": state_vec}
        for ck in meta.camera_keys:
            if ck.endswith(cfg.scene_camera_key):
                frame[ck] = overlaid
            else:
                frame[ck] = _to_hwc_uint8(item[ck])
        chunk, policy_ms = _predict_chunk(cfg, policy, preprocessor, postprocessor, frame, task, robot_type, cfg.device)
        policy_ms_list.append(policy_ms)
        finite = bool(torch.isfinite(chunk).all().item())
        logger.info(
            "frame %d: mask=%s sam=%.0fms policy=%.0fms chunk=%s finite=%s range=[%.2f,%.2f]",
            i, "yes" if mask is not None else "MISS", sam_ms, policy_ms,
            tuple(chunk.shape), finite, float(chunk.min()), float(chunk.max()),
        )
        if i < cfg.save_qc_frames:
            Image.fromarray(np.concatenate([scene, overlaid], axis=1)).save(out / f"qc_{i:03d}_raw_vs_overlay.png")

        if i == 0:
            # Validate the LIVE inference path (select_action queue + postprocessor)
            # offline: it must yield plausible joint-degree targets, not the raw
            # normalized [-1,1] chunk. De-risks the on-robot run.
            policy.reset()
            with torch.inference_mode():
                obs_p = prepare_observation_for_inference(copy(frame), torch.device(cfg.device), task, robot_type)
                a = postprocessor(policy.select_action(preprocessor(obs_p)))
            adict = make_robot_action(a.squeeze(0).cpu(), meta.features)
            names = meta.features["action"]["names"]
            logger.info("live-path postprocessed action[0]: %s", {k: round(float(adict[k]), 2) for k in names})

    logger.info(
        "SMOKE DONE. SAM ms median=%.0f p95=%.0f | policy ms median=%.0f p95=%.0f | QC -> %s",
        _pct(sam_ms_list, 50), _pct(sam_ms_list, 95), _pct(policy_ms_list, 50), _pct(policy_ms_list, 95), out,
    )


# ───────────────────────────── live mode ───────────────────────────────────


def _reset_realsense(serials: list[str], wait_s: int) -> None:
    """hardware_reset the given RealSense devices and wait for re-enumeration.

    Raw V4L2/OpenCV access (or a prior crash) can leave a RealSense streaming-but-
    stalled so frames never arrive; a hardware reset clears it.
    """
    import pyrealsense2 as rs

    targets = set(serials)
    did = []
    for d in rs.context().query_devices():
        s = d.get_info(rs.camera_info.serial_number)
        if s in targets:
            try:
                d.hardware_reset()
                did.append(s)
            except Exception as e:  # noqa: BLE001
                logger.warning("RealSense %s hardware_reset failed: %s", s, e)
    if did:
        logger.info("RealSense hardware_reset %s; waiting %ds for re-enumeration...", did, wait_s)
        time.sleep(wait_s)


def _build_robot(cfg: GroundedRolloutConfig):
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    if cfg.camera_backend == "realsense":
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        common = {"fps": cfg.cam_fps, "width": cfg.cam_width, "height": cfg.cam_height,
                  "warmup_s": cfg.cam_warmup_s, "use_depth": False}
        cameras = {
            "scene": RealSenseCameraConfig(serial_number_or_name=cfg.scene_serial, **common),
            "wrist": RealSenseCameraConfig(serial_number_or_name=cfg.wrist_serial, **common),
        }
    else:  # opencv fallback
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cam_kw = {"fps": cfg.cam_fps, "width": cfg.cam_width, "height": cfg.cam_height, "warmup_s": cfg.cam_warmup_s}
        if cfg.cam_fourcc:
            cam_kw["fourcc"] = cfg.cam_fourcc
        cameras = {
            "scene": OpenCVCameraConfig(index_or_path=cfg.scene_camera_index, **cam_kw),
            "wrist": OpenCVCameraConfig(index_or_path=cfg.wrist_camera_index, **cam_kw),
        }

    robot_cfg = SOFollowerRobotConfig(id=cfg.robot_id, port=cfg.robot_port, cameras=cameras, use_degrees=cfg.use_degrees)
    robot = make_robot_from_config(robot_cfg)
    return robot


def run_live(cfg: GroundedRolloutConfig) -> None:
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)
    task = _resolve_task(meta, cfg)
    fps = cfg.fps or meta.fps
    robot_type = getattr(meta, "robot_type", "so_follower")
    ordered_action_keys = list(meta.features["action"]["names"])
    features = meta.features
    control_dt = 1.0 / float(fps)
    device = cfg.device

    policy, preprocessor, postprocessor = _load_policy_and_processors(cfg, meta.stats)
    policy.config.n_action_steps = cfg.n_action_steps
    seg = None
    if cfg.grounding:
        seg = LiveStrawberrySegmenter(
            cfg.sam_prompt, confidence_threshold=cfg.sam_confidence, device=device,
            redetect_max_dist_frac=cfg.redetect_max_dist_frac,
        )
        if cfg.phase_switch in ("never", "off", "False"):
            logger.info("PHASE SWITCH = NEVER — mask stays on the target ALL episode; NEVER enters placement "
                        "mode (a failed/pushing grasp keeps trying the same berry).")
        else:
            logger.info("PHASE SWITCH = %r (placement=%s). Grasp->placement can trigger.",
                        cfg.phase_switch, cfg.placement_style)
    else:
        logger.info("grounding=False: RAW mode (no SAM/overlay/lock-on).")

    if cfg.camera_backend == "realsense" and cfg.reset_realsense:
        _reset_realsense([cfg.scene_serial, cfg.wrist_serial], cfg.realsense_reset_wait_s)

    robot = _build_robot(cfg)
    logger.info("Connecting robot on %s ...", cfg.robot_port)
    robot.connect()

    autocast_ctx = lambda: (  # noqa: E731
        torch.autocast(device_type=torch.device(device).type)
        if torch.device(device).type == "cuda" and getattr(policy.config, "use_amp", False)
        else nullcontext()
    )

    tag = f"_{cfg.run_tag}" if cfg.run_tag else ""
    log_path = out / f"harvest_log{tag}.jsonl"
    summary_rows = []
    estop_state = None
    try:
        for ep in range(cfg.episodes):
            # Reset gate: let the operator reposition the arm + re-place the berries/scene
            # before each episode, then start when ready. Optionally release motor torque
            # so the arm can be moved by hand, and re-engage it before the run.
            if cfg.reset_gate and not cfg.dry_run:
                if cfg.release_torque_on_reset:
                    try:
                        robot.bus.disable_torque()
                        logger.info("Torque released — reposition the arm by hand.")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Could not disable torque: %s", e)
                input(f"\n=== Reset scene + arm, then press ENTER to start episode {ep + 1}/{cfg.episodes} ===\n")
                if cfg.release_torque_on_reset:
                    try:
                        robot.bus.enable_torque()
                        logger.info("Torque re-engaged.")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Could not re-enable torque: %s", e)
            logger.info("=== Episode %d/%d ===", ep + 1, cfg.episodes)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            if seg is not None:
                seg.reset()

            # --- target acquisition (lock-on gate) — grounded mode only ---
            if cfg.grounding:
                click_xy = None
                if cfg.target_mode == "click":
                    from lerobot.grounding.pipeline import _select_click
                    frame0 = _to_hwc_uint8(robot.get_observation()[cfg.scene_camera_key])
                    click_xy = tuple(_select_click(frame0, prompt="target strawberry", radius_px=24, raw_image_path=out / f"ep{ep:02d}_lockon.png"))
                t_lock = time.perf_counter()
                while True:
                    scene = _to_hwc_uint8(robot.get_observation()[cfg.scene_camera_key])
                    if seg.lock_on(scene, click_xy=click_xy, auto_rule=cfg.auto_rule):
                        break
                    if time.perf_counter() - t_lock > cfg.lock_on_timeout_s:
                        logger.warning("Lock-on timed out; skipping episode."); break
                if seg.last_centroid is None:
                    continue

            # --- control loop ---
            phase = "grasp"
            cached_batch = None
            consecutive_miss = 0
            phase_switch_chunk = None
            last_tray_bbox = None
            grasp_centroid = None  # berry's table position at the moment of the grasp->placement switch
            revert_hits = 0
            gripper_closed_streak = 0
            sam_ms_list, policy_ms_list, chunk_ms_list = [], [], []
            n_chunks = 0
            tick = 0
            ep_start = time.perf_counter()
            loop_times = []

            estop_state = _early_stop_enter(active=not cfg.dry_run)
            if estop_state is not None:
                logger.info("Press any key to END this episode early (auto-ends at %.0fs).", cfg.episode_duration_s)

            while time.perf_counter() - ep_start < cfg.episode_duration_s:
                estop_key = _early_stop_pressed(estop_state)
                if estop_key is not None:
                    logger.info("Episode %d ended early by operator (key=%r).", ep + 1, estop_key)
                    break
                loop_start = time.perf_counter()
                need_infer = len(policy._action_queue) == 0
                obs = robot.get_observation()

                if need_infer:
                    n_chunks += 1
                    if cfg.grounding and phase == "grasp":
                        scene = _to_hwc_uint8(obs[cfg.scene_camera_key])
                        t0 = time.perf_counter()
                        mask = seg.track(scene)
                        sam_ms_list.append((time.perf_counter() - t0) * 1e3)
                        if mask is None:
                            consecutive_miss += 1
                            if consecutive_miss <= cfg.mask_hold_chunks and seg.last_mask is not None:
                                mask = seg.last_mask  # hold last good mask through a transient miss
                        else:
                            consecutive_miss = 0
                        obs[cfg.scene_camera_key] = _overlay_scene(cfg, scene, mask)
                        # Phase switch needs BOTH: the gripper actually CLOSED (a real grasp) AND
                        # the berry mask is gone. Mask-loss alone is ambiguous — the arm can occlude
                        # an ungrasped berry from the global camera. Gripper closed = pos past the
                        # threshold (data: open ~2-3, closed-on-berry ~40-50, so closed = ABOVE ~25).
                        # If the gripper signal is disabled (thresh=None), fall back to mask-loss only.
                        gpos = float(obs.get("gripper.pos", float("nan")))
                        gripper_closed = None
                        if cfg.grasp_gripper_close_thresh is not None:
                            gripper_closed = (gpos < cfg.grasp_gripper_close_thresh) if cfg.gripper_closed_is_below else (gpos > cfg.grasp_gripper_close_thresh)
                        gripper_closed_streak = gripper_closed_streak + 1 if gripper_closed else 0
                        mask_lost = consecutive_miss >= cfg.mask_lost_chunks
                        # Switch on the GRIPPER closing (held for grasp_hold_chunks) — the reliable grasp
                        # signal — OR mask-loss-while-closed. SAM usually keeps seeing the carried berry,
                        # so mask-loss alone rarely fires. The revert undoes false switches (berry still on table).
                        gripper_held = (cfg.grasp_gripper_close_thresh is None) or (gripper_closed_streak >= cfg.grasp_hold_chunks)
                        if cfg.phase_switch in ("never", "off", "False"):
                            grasped = False  # never switch: keep the mask on the target the whole episode
                        elif cfg.phase_switch == "lift":
                            # require the berry to have LEFT the frame (lifted/occluded) AND a held grasp;
                            # a weak grasp that leaves the berry visible on the table won't falsely switch.
                            grasped = mask_lost and gripper_held
                        else:  # "gripper" (legacy)
                            if cfg.grasp_gripper_close_thresh is None:
                                grasped = mask_lost
                            else:
                                grasped = gripper_closed_streak >= cfg.grasp_hold_chunks or (mask_lost and bool(gripper_closed))
                        if grasped:
                            phase = "placement"
                            phase_switch_chunk = n_chunks
                            grasp_centroid = seg.last_centroid  # remember the berry's table spot for the revert check
                            revert_hits = 0
                            logger.info("Phase -> PLACEMENT at chunk %d (miss=%d, gripper_closed=%s, gpos=%.1f)",
                                        n_chunks, consecutive_miss, gripper_closed, gpos)
                    elif cfg.grounding and phase == "placement":
                        scene = _to_hwc_uint8(obs[cfg.scene_camera_key])
                        reverted = False
                        # Reversible switch: did the target berry reappear on the table near where
                        # it was before the grasp? Then the grasp failed (mask-loss was occlusion) ->
                        # revert to grasp. A real grasp carries the berry away, so it won't reappear.
                        # Grace period: don't revert for the first few chunks after switching, so a
                        # genuine grasp has time to lift/carry the berry clear of its old spot.
                        grace_ok = phase_switch_chunk is not None and (n_chunks - phase_switch_chunk) >= cfg.revert_grace_chunks
                        if cfg.placement_revert and grasp_centroid is not None and grace_ok:
                            maxd = cfg.revert_max_dist_frac * float(np.hypot(scene.shape[0], scene.shape[1]))
                            near = [d for d in seg.detect(scene)
                                    if np.hypot(d.centroid[0] - grasp_centroid[0], d.centroid[1] - grasp_centroid[1]) <= maxd]
                            if near:
                                revert_hits += 1
                                if revert_hits >= cfg.revert_chunks:
                                    phase = "grasp"
                                    consecutive_miss = 0
                                    revert_hits = 0
                                    seg.last_centroid = near[0].centroid
                                    seg.last_mask = near[0].mask
                                    obs[cfg.scene_camera_key] = _overlay_scene(cfg, scene, near[0].mask)
                                    reverted = True
                                    logger.info("Reverted PLACEMENT -> GRASP at chunk %d (target reappeared on table)", n_chunks)
                            else:
                                revert_hits = 0
                        if not reverted and cfg.placement_style == "box":
                            # placement target = the green tray: box + crosshair (no fill).
                            tray = _detect_tray_bbox(scene)
                            if tray is not None:
                                last_tray_bbox = tray
                            tray = tray if tray is not None else last_tray_bbox
                            obs[cfg.scene_camera_key] = grounding_overlay_frame(
                                scene, None, color=cfg.overlay_color, alpha=cfg.overlay_alpha,
                                bbox_xyxy=tray, draw_crosshair=tray is not None,
                            )
                    # brightness model: brighten every camera frame to match training
                    if cfg.brighten_gamma is not None:
                        for ck in meta.camera_keys:
                            wk = ck.removeprefix(f"{OBS_STR}.images.")
                            if wk in obs:
                                obs[wk] = _brighten(_to_hwc_uint8(obs[wk]), cfg.brighten_gamma)
                    # --- input inspection dumps (first N chunks) ---
                    if n_chunks <= cfg.save_qc_frames:
                        Image.fromarray(_to_hwc_uint8(obs[cfg.scene_camera_key])).save(out / f"ep{ep:02d}{tag}_chunk{n_chunks:03d}_scene.png")
                        wrist_key = next((k for k in meta.camera_keys if k.endswith("wrist")), None)
                        if wrist_key:
                            wk = wrist_key.removeprefix(f"{OBS_STR}.images.")
                            if wk in obs:
                                Image.fromarray(_to_hwc_uint8(obs[wk])).save(out / f"ep{ep:02d}{tag}_chunk{n_chunks:03d}_wrist.png")
                        state_vals = {k: round(float(obs[k]), 2) for k in meta.features["observation.state"]["names"] if k in obs}
                        logger.info("ep%d chunk%d phase=%s state=%s", ep, n_chunks, phase, state_vals)
                    # build + preprocess the (possibly overlaid) observation
                    frame = build_dataset_frame(features, obs, prefix=OBS_STR)
                    with torch.inference_mode(), autocast_ctx():
                        observation = prepare_observation_for_inference(frame, torch.device(device), task, robot_type)
                        cached_batch = preprocessor(observation)

                # advance the policy (infers from cached_batch only when the queue is empty)
                with torch.inference_mode(), autocast_ctx():
                    t0 = time.perf_counter()
                    action = policy.select_action(cached_batch)
                    if need_infer and torch.device(device).type == "cuda":
                        torch.cuda.synchronize()
                    if need_infer:
                        policy_ms_list.append((time.perf_counter() - t0) * 1e3)
                    action = postprocessor(action)

                action_tensor = action.squeeze(0).cpu()
                action_dict = make_robot_action(action_tensor, features)
                if need_infer and n_chunks <= cfg.save_qc_frames:
                    logger.info("ep%d chunk%d predicted joint targets=%s",
                                ep, n_chunks, {k: round(float(action_dict[k]), 2) for k in ordered_action_keys})
                if not cfg.dry_run:
                    robot.send_action({k: action_dict[k] for k in ordered_action_keys})

                if need_infer:
                    chunk_ms_list.append((time.perf_counter() - loop_start) * 1e3)
                loop_times.append(time.perf_counter() - loop_start)
                tick += 1
                sleep_t = control_dt - (time.perf_counter() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            _early_stop_restore(estop_state)
            estop_state = None
            achieved_hz = (tick / (time.perf_counter() - ep_start)) if tick else 0.0
            logger.info("Episode %d done: %d ticks, %d chunks, ~%.1f Hz.", ep + 1, tick, n_chunks, achieved_hz)

            # --- operator success scoring (skipped in dry_run; arm didn't move) ---
            if cfg.dry_run:
                logger.info("dry_run: skipping success scoring. Inspect dumps in %s", out)
                continue
            grasp_ok = terminal_binary_label("Grasp success? [y/n]: ")
            place_ok = terminal_binary_label("Placement success? [y/n]: ")
            row = {
                "episode": ep,
                "grasp_success": grasp_ok,
                "place_success": place_ok,
                "overall_success": bool(grasp_ok and place_ok),
                "n_chunks": n_chunks,
                "n_ticks": tick,
                "achieved_hz": round(achieved_hz, 2),
                "phase_switch_chunk": phase_switch_chunk,
                "sam_ms_median": round(_pct(sam_ms_list, 50), 1),
                "sam_ms_p95": round(_pct(sam_ms_list, 95), 1),
                "policy_ms_median": round(_pct(policy_ms_list, 50), 1),
                "policy_ms_p95": round(_pct(policy_ms_list, 95), 1),
                "chunk_ms_median": round(_pct(chunk_ms_list, 50), 1),
            }
            summary_rows.append(row)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info("Logged: %s", row)
    finally:
        _early_stop_restore(estop_state)
        try:
            robot.disconnect()
        except Exception:
            pass

    if summary_rows:
        n = len(summary_rows)
        gr = sum(r["grasp_success"] for r in summary_rows)
        ov = sum(r["overall_success"] for r in summary_rows)
        logger.info(
            "SUMMARY (%d eps): grasp %d/%d (%.0f%%), overall %d/%d (%.0f%%) | median SAM %.0fms, policy %.0fms | log -> %s",
            n, gr, n, 100 * gr / n, ov, n, 100 * ov / n,
            _pct([r["sam_ms_median"] for r in summary_rows], 50),
            _pct([r["policy_ms_median"] for r in summary_rows], 50),
            log_path,
        )


@draccus.wrap()
def main(cfg: GroundedRolloutConfig) -> None:
    logger.info("Config: %s", cfg)
    if cfg.smoke:
        run_smoke(cfg)
    else:
        run_live(cfg)


if __name__ == "__main__":
    main()
