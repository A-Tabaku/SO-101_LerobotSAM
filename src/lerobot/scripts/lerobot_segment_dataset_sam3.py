#!/usr/bin/env python
"""Derive a SAM3.1-grounded dataset from a recorded grounded dataset.

For each source episode this reads the stored frame-0 click point
(``grounding_click_xy``) from ``meta/episodes``, decodes the episode's scene
frames out of the MP4 videos, seeds SAM3.1 on frame 0 with that point, then
``propagate_in_video`` tracks the target mask across every frame. Each frame's
``observation.images.scene`` is rewritten to the grounded view (target bright,
rest dimmed by ``dim_factor``); all other features (wrist image, state, action)
are copied unchanged. The result is a new LeRobot dataset ready for policy
training (e.g. pi0.5).

Runs on a CUDA GPU (SAM video propagation is not practical on CPU). When SAM
returns no mask for a frame (e.g. brief occlusion), it falls back to the static
click-circle mask so the frame is still grounded.

NOTE: this script intentionally does NOT use ``from __future__ import
annotations`` — ``@parser.wrap()`` reads the function's raw annotation, and a
stringized annotation breaks draccus.
"""

import inspect
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
from PIL import Image

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.grounding import (
    GROUNDING_BBOX_KEY,
    GROUNDING_CLICK_XY_KEY,
    GROUNDING_DIM_FACTOR_KEY,
    GROUNDING_POLICY_PROMPT_KEY,
    GROUNDING_SOURCE_CAMERA_KEY,
    GROUNDING_SUCCESS_KEY,
    grounding_frame,
    grounding_overlay_frame,
)
from lerobot.grounding.pipeline import ellipse_mask_from_box
from lerobot.utils.constants import DEFAULT_FEATURES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class SegmentSam3Config:
    source_repo_id: str
    output_repo_id: str
    source_root: str | Path | None = None
    output_root: str | Path | None = None
    # Dataset-feature name of the scene stream (NOT the bare obs key "scene").
    scene_camera_key: str = "observation.images.scene"
    # Text concept SAM detects each frame; the stored click point disambiguates
    # which instance is the target. "red fruit" detects the strawberries on all
    # 50 episodes; "strawberry" misses them on ~60% of episodes (the concept
    # word matters a lot for these particular berries).
    sam_prompt: str = "red fruit"
    # SAM backend:
    #   "sam3_tracker"     -> SAM2-style point-seeded tracker on the sam3 ckpt;
    #                         ~8-12GB peak (no OOM), point-seeded, masks ~0.96 IoU
    #                         vs multiplex. Default — cheaper for data generation.
    #   "sam3.1_multiplex" -> text-concept detect-all + select-by-click; ~25-28GB
    #                         peak; mirrors the intended inference-time selector.
    # (sam_prompt / redetect_max_dist_frac / seed_scan_frames apply to multiplex only.)
    sam_backend: str = "sam3_tracker"
    # Path to a SAM checkpoint. None -> download the backend's default from HF.
    sam_checkpoint: str | None = None
    sam_version: str = "sam3.1"
    # ── Visual prompt (training render) ──────────────────────────────────────
    # "overlay" = normal-brightness RGB + semi-transparent mask + contour (the
    # training input). "dim" = legacy debug render that dims the whole scene.
    render_style: str = "overlay"
    # Marker colour (R, G, B), interpreted in image channel order. Deep azure-blue
    # stays distinct from red fruit and green foliage at inference.
    overlay_color: tuple[int, int, int] = (0, 60, 190)
    # Mask opacity for the overlay render (0..1). Lighter = more berry shows through.
    overlay_alpha: float = 0.35
    # Also draw a bbox around the target mask. Off by default: a box occludes
    # neighbours in clustered scenes; the tight mask is the primary marker.
    draw_target_bbox: bool = False
    # Dim factor for the legacy "dim" debug render only.
    dim_factor: float | None = None
    # ── Phase split ──────────────────────────────────────────────────────────
    # Grasp-only dataset: truncate each episode at the detected lift/grasp so SAM
    # only ever marks the target *before* it leaves the table.
    grasp_only: bool = True
    # Keep this many frames past the detected lift. 0: the mask-motion detector
    # already fires at lift onset, so no carry frames are needed.
    grasp_margin: int = 0
    # Full-episode mode (keeps EVERY frame; overrides grasp_only truncation):
    #   None  -> grasp-only truncation (drop placement frames).
    #   "raw" -> strawberry overlay on grasp frames [0:lift], NORMAL RGB on the
    #            placement frames [lift:] (grasp segmented + normal placement).
    #   "box" -> (future) mark the tray on placement frames instead of raw RGB.
    placement: str | None = None
    # When the tracker drops the target, re-acquire the nearest detection within
    # this fraction of the image diagonal (keeps it from jumping to a different
    # strawberry). ~0.08 of a 640x480 frame ≈ 64px — tight enough to reject a
    # neighbouring berry (~115px away) while still following the target into the
    # gripper/bin. Larger values risk locking onto the wrong berry on re-acquire.
    redetect_max_dist_frac: float = 0.08
    # Opening frames can be motion-blurred (detector finds nothing); scan up to
    # this many frames for the first usable seed detection near the click.
    seed_scan_frames: int = 30
    # Process only the first N episodes (e.g. 1 for a quick spot-check). None = all.
    max_episodes: int | None = None
    # Periodic predictor rebuild. DISABLED (0) by default: diagnosis showed there
    # is NO cross-episode leak (steady-state is flat ~5GB); each episode just
    # transiently PEAKS at ~25-28GB on a 31GB card. Rebuilding mid-run fragments
    # the CUDA allocator so the next peak no longer fits -> OOM. Leaving the
    # allocator untouched (+ PYTORCH_ALLOC_CONF=expandable_segments:True) lets the
    # peaks recur safely. Only raise this if a genuine leak is reintroduced.
    predictor_rebuild_every: int = 0
    push_to_hub: bool = False
    private: bool = False
    video: bool = True
    batch_encoding_size: int = 1
    device: str = "cuda"


def _to_numpy_image(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    else:
        value = np.asarray(value)

    if value.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {value.shape}.")
    if value.shape[0] in {1, 3} and value.shape[-1] not in {1, 3}:
        value = np.moveaxis(value, 0, -1)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value, 0.0, 1.0 if value.max() <= 1.0 else 255.0)
        if value.max() <= 1.0:
            value = (value * 255.0).round()
    return value.astype(np.uint8)


def _to_numpy_feature(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return value


def _episode_column(meta, name: str):
    episodes = meta.episodes
    column_names = getattr(episodes, "column_names", None)
    if column_names is not None and name not in column_names:
        return None
    try:
        return episodes[name]
    except Exception:
        return None


def _episode_value(meta, name: str, episode_index: int, default=None):
    column = _episode_column(meta, name)
    if column is None:
        return default
    value = column[episode_index]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _gripper_index(src) -> int:
    """Index of `gripper.pos` in observation.state (falls back to last dim)."""
    names = (src.features.get("observation.state") or {}).get("names") or []
    if "gripper.pos" in names:
        return names.index("gripper.pos")
    return len(names) - 1 if names else -1


def _gripper_series(src, start_index: int, end_index: int, gripper_idx: int) -> np.ndarray:
    """Per-frame gripper.pos over [start,end) — read WITHOUT decoding video."""
    hf = getattr(src, "hf_dataset", None)
    if hf is not None:
        try:
            rows = hf[start_index:end_index]["observation.state"]
            return np.asarray([float(r[gripper_idx]) for r in rows], dtype=np.float32)
        except Exception:
            pass
    return np.asarray(
        [float(_to_numpy_feature(src[i]["observation.state"])[gripper_idx]) for i in range(start_index, end_index)],
        dtype=np.float32,
    )


def _detect_grasp_frame(gripper: np.ndarray, close_frac: float = 0.6, hold: int = 15) -> int | None:
    """Episode-relative frame where the grasp completes.

    The gripper opens to a peak to receive the berry, then closes onto it. We
    return the first frame after the open-peak where gripper.pos stays below
    ``close_frac`` of the peak for ``hold`` consecutive frames (sustained close).
    Returns None if no clear grasp is found (caller then keeps the full episode).
    """
    n = len(gripper)
    if n < hold + 1:
        return None
    peak = int(np.argmax(gripper))
    threshold = close_frac * float(gripper.max())
    for i in range(peak, n - hold):
        if np.all(gripper[i : i + hold] < threshold):
            return i
    return None


def _detect_lift_frame(
    masks: list, image_hw, disp_frac: float = 0.06, hold: int = 8, base_frac: float = 0.15
) -> int | None:
    """Episode-relative frame where the target berry is lifted off the table.

    Tied to the actual event, not the (noisy, multi-cycle) gripper signal: the
    target is stationary pre-grasp, so the lift is the first frame where its
    tracked mask centroid moves more than ``disp_frac`` of the image diagonal
    from its resting position and stays moved. If the mask is instead *lost* for
    ``hold`` consecutive frames after the resting period (gripper occludes the
    grasped berry), that loss onset is taken as the lift. Returns None if the
    berry never settles or never moves (caller keeps the full episode).
    """
    n = len(masks)
    if n < hold + 1:
        return None
    height, width = int(image_hw[0]), int(image_hw[1])
    threshold = disp_frac * (height**2 + width**2) ** 0.5
    centroids = [_mask_centroid(m) if (m is not None and m.any()) else None for m in masks]

    base_n = max(int(n * base_frac), 10)
    base = [c for c in centroids[:base_n] if c is not None]
    if not base:
        return None
    bx = float(np.median([c[0] for c in base]))
    by = float(np.median([c[1] for c in base]))

    lost = 0
    for i in range(base_n, n - 1):
        c = centroids[i]
        if c is None:  # mask lost — likely grasp occlusion
            lost += 1
            if lost >= hold:
                return i - hold + 1
            continue
        lost = 0
        if ((c[0] - bx) ** 2 + (c[1] - by) ** 2) ** 0.5 > threshold:
            # confirm sustained motion (not a one-frame jitter)
            fwd = [centroids[j] for j in range(i, min(i + hold, n)) if centroids[j] is not None]
            if len(fwd) >= hold // 2 and all(
                ((c2[0] - bx) ** 2 + (c2[1] - by) ** 2) ** 0.5 > 0.7 * threshold for c2 in fwd
            ):
                return i
    return None


def _detect_tray_bbox(image: np.ndarray, min_green_px: int = 5000) -> list[int] | None:
    """Bounding box of the green tray via a colour threshold (it's a distinct
    green and only drifts a few cm). Returns [x1,y1,x2,y2] or None when too
    little green is visible (e.g. the arm occludes the tray); the caller then
    reuses the last good box.

    The tray green is fairly dark/desaturated (a tray pixel is ~RGB(93,107,30)),
    so the gate is "green clearly dominates blue, and G >= R" rather than a high
    absolute G — this reliably covers ~30-40k px of the tray.
    """
    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)
    green = (g > 85) & (g - b > 30) & (g - r > 0)
    ys, xs = np.where(green)
    if xs.size < min_green_px:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ── SAM3.1 video predictor ────────────────────────────────────────────────────
# These helpers wrap the SAM 3.1 *multiplex* video predictor, which exposes a
# session/request API (start_session -> add_prompt -> propagate_in_video) rather
# than the SAM2-style init_state/add_new_points pattern. The call shapes below
# mirror `examples/sam3.1_video_predictor_example.ipynb` in the sam3 repo:
# points are passed as RELATIVE (0-1) coordinates, and per-frame masks are read
# from the streamed outputs' ``out_obj_ids`` / ``out_binary_masks``. If your
# installed `sam3` build differs, this is the only section that should need
# touching.


def _shim_init_state(obj) -> None:
    """Wrap ``obj.init_state`` to silently drop kwargs it doesn't accept.

    Both backends hit a version skew where the base predictor forwards extra
    kwargs (e.g. ``offload_state_to_cpu``) that the underlying model's
    ``init_state`` doesn't take. Filtering to the accepted params fixes it.
    """
    orig = obj.init_state
    allowed = set(inspect.signature(orig).parameters)

    def wrapped(*args, **kwargs):
        return orig(*args, **{k: v for k, v in kwargs.items() if k in allowed})

    obj.init_state = wrapped


def _build_sam3_video_predictor(cfg: SegmentSam3Config):
    """Build the SAM predictor for the selected backend.

    - ``sam3.1_multiplex``: text-concept multi-instance video tracker (detect-all
      + select-by-click). Heavier (~25-28GB peak) but mirrors the intended
      inference-time selector.
    - ``sam3_tracker``: SAM2-style point-seeded single-object tracker on the
      lighter sam3 checkpoint (~8-12GB peak). Masks are ~0.96 IoU vs multiplex,
      so interchangeable for the policy. No text prompt / instance selection.
    """
    import torch

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if cfg.sam_backend == "sam3_tracker":
        from sam3.model_builder import build_sam3_video_model

        model = build_sam3_video_model(checkpoint_path=cfg.sam_checkpoint)  # defaults to sam3 ckpt
        predictor = model.tracker
        predictor.backbone = model.detector.backbone
        _shim_init_state(predictor)
    else:  # sam3.1_multiplex
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        # use_fa3=False uses the SDPA attention path (no FlashAttention-3 build needed).
        predictor = build_sam3_multiplex_video_predictor(checkpoint_path=cfg.sam_checkpoint, use_fa3=False)
        _shim_init_state(predictor.model)

    return predictor, device


def _propagate_episode_tracker(predictor, frames_dir: str, num_frames: int, point_xy, image_hw, device) -> list:
    """SAM2-style point-seeded tracking (``sam3_tracker`` backend).

    Seed the stored click on frame 0 and propagate that single object forward.
    Returns a list of length ``num_frames``; entries are bool HxW masks or None.
    """
    import torch

    height, width = int(image_hw[0]), int(image_hw[1])
    rel_points = torch.tensor([[point_xy[0] / width, point_xy[1] / height]], dtype=torch.float32)
    point_labels = torch.tensor([1], dtype=torch.int32)

    masks: list = [None] * num_frames
    with torch.inference_mode():
        state = predictor.init_state(video_path=frames_dir)
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
        predictor.add_new_points(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=rel_points,
            labels=point_labels,
            clear_old_points=False,
        )
        for out in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=num_frames, reverse=False, propagate_preflight=True
        ):
            frame_idx, video_res_masks = out[0], out[3]
            if not (0 <= frame_idx < num_frames):
                continue
            arr = (video_res_masks[0] > 0.0).squeeze().detach().cpu().numpy().astype(bool)
            if arr.ndim == 2 and arr.any():
                masks[frame_idx] = arr
    return masks


def _outputs_to_masks(outputs):
    """Return (obj_ids: list[int], masks: np.ndarray[N,H,W] bool) from an output dict.

    Outputs carry parallel ``out_obj_ids`` and ``out_binary_masks`` arrays
    (mirrors sam3's ``prepare_masks_for_visualization``).
    """
    obj_ids = outputs.get("out_obj_ids")
    binary_masks = outputs.get("out_binary_masks")
    if obj_ids is None or binary_masks is None:
        return [], np.empty((0, 0, 0), dtype=bool)
    obj_ids = [int(o) for o in (obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids))]
    if hasattr(binary_masks, "detach"):
        binary_masks = binary_masks.detach().cpu().numpy()
    return obj_ids, np.asarray(binary_masks).astype(bool)


def _select_obj_id(outputs, point_xy, max_dist=None):
    """Pick which detected instance the user clicked on.

    The text concept (e.g. "strawberry") detects every matching instance on the
    seed frame; the per-episode click disambiguates which one to track. Prefer
    the instance whose mask covers the click; otherwise the nearest mask
    centroid. When ``max_dist`` is given, reject a nearest match farther than
    that (pixels) from the click — used while scanning for a seed frame so a
    spurious blurry detection elsewhere isn't mistaken for the target. Returns
    the chosen obj_id, or None if nothing suitable was detected.
    """
    obj_ids, masks = _outputs_to_masks(outputs)
    if not obj_ids:
        return None
    h, w = masks.shape[1:]
    cx = min(max(int(round(point_xy[0])), 0), w - 1)
    cy = min(max(int(round(point_xy[1])), 0), h - 1)
    for i, oid in enumerate(obj_ids):
        if masks[i][cy, cx]:
            return oid
    best_id, best_dist = None, None
    for i, oid in enumerate(obj_ids):
        ys, xs = np.where(masks[i])
        if xs.size == 0:
            continue
        dist = (xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2
        if best_dist is None or dist < best_dist:
            best_id, best_dist = oid, dist
    if best_id is not None and max_dist is not None and best_dist > max_dist**2:
        return None
    return best_id


def _mask_centroid(mask):
    """(cx, cy) of a bool mask, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mask_by_id(obj_ids, masks, obj_id):
    """Bool HxW mask for ``obj_id`` (None if absent/empty)."""
    if obj_id not in obj_ids:
        return None
    arr = masks[obj_ids.index(obj_id)].squeeze()
    if arr.ndim != 2 or not arr.any():
        return None
    return arr


def _nearest_instance(obj_ids, masks, anchor_xy, max_dist):
    """Closest detected instance to ``anchor_xy`` within ``max_dist`` pixels.

    Returns (obj_id, mask, centroid) or None. Used to re-acquire the target
    after the tracker drops it (e.g. it's grasped and re-detected under a new
    id): we adopt the nearest detection to the last-known position, but the
    distance gate keeps us from jumping to a different, stationary instance.
    """
    best = None
    for i, oid in enumerate(obj_ids):
        centroid = _mask_centroid(masks[i])
        if centroid is None:
            continue
        dist = ((centroid[0] - anchor_xy[0]) ** 2 + (centroid[1] - anchor_xy[1]) ** 2) ** 0.5
        if dist <= max_dist and (best is None or dist < best[0]):
            best = (dist, oid, masks[i].squeeze(), centroid)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _propagate_episode(
    predictor, frames_dir: str, num_frames: int, point_xy, sam_prompt: str,
    redetect_max_dist_frac: float, seed_scan_frames: int, device,
) -> list:
    """Seed the text concept at the first usable frame, pick the clicked
    instance, and propagate.

    SAM 3.1's multiplex tracker is concept-driven: a text prompt detects and
    densely tracks every matching instance across the video, while the stored
    click selects which instance is "the" target.

    Robustness details:
    - The opening frames can be motion-blurred so the detector finds nothing on
      frame 0; we scan forward up to ``seed_scan_frames`` for the first frame
      with a detection near the click and seed there (earlier frames fall back).
    - The target is then tracked by *position continuity*: we follow its object
      id while present, and when the tracker drops it (e.g. it's grasped and
      later re-detected under a NEW id) we re-acquire the nearest detection to
      the last-known location, gated by ``redetect_max_dist_frac`` of the image
      diagonal so we never jump to a different strawberry.

    Returns a list of length ``num_frames``; entries are bool HxW masks for the
    target, or None where it couldn't be located (caller applies the
    click-circle fallback).
    """
    import torch

    masks: list = [None] * num_frames
    with torch.inference_mode():
        # offload_video_to_cpu keeps the decoded frames off the GPU; robotics
        # episodes run many hundreds of frames and otherwise OOM a 32GB card.
        session_id = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": frames_dir,
                "offload_video_to_cpu": True,
            }
        )["session_id"]
        try:
            # Scan for the first frame whose detection lands near the click.
            cur_id, seed_outputs, seed_frame = None, None, 0
            scan_limit = min(max(seed_scan_frames, 1), num_frames)
            for frame_index in range(scan_limit):
                if frame_index > 0:  # clear the previous frame's prompt before retrying
                    predictor.handle_request(
                        request={"type": "reset_session", "session_id": session_id}
                    )
                seed = predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "text": sam_prompt,
                    }
                )
                _, scan_masks = _outputs_to_masks(seed["outputs"])
                max_dist = (
                    redetect_max_dist_frac * (scan_masks.shape[1] ** 2 + scan_masks.shape[2] ** 2) ** 0.5
                    if scan_masks.size
                    else None
                )
                candidate = _select_obj_id(seed["outputs"], point_xy, max_dist=max_dist)
                if candidate is not None:
                    cur_id, seed_outputs, seed_frame = candidate, seed["outputs"], frame_index
                    break
            if cur_id is None:  # nothing detected near the click -> caller falls back
                return masks
            seed_ids, seed_masks = _outputs_to_masks(seed_outputs)
            height, width = seed_masks.shape[1:]
            max_dist = redetect_max_dist_frac * (height**2 + width**2) ** 0.5
            anchor_xy = _mask_centroid(seed_masks[seed_ids.index(cur_id)])

            for response in predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": seed_frame,
                    "propagation_direction": "forward",
                }
            ):
                frame_idx = response["frame_index"]
                if not (0 <= frame_idx < num_frames):
                    continue
                obj_ids, frame_masks = _outputs_to_masks(response["outputs"])
                # 1) keep following the current id if it's still tracked
                mask = _mask_by_id(obj_ids, frame_masks, cur_id)
                # 2) otherwise re-acquire the nearest detection to last position
                if mask is None and anchor_xy is not None:
                    found = _nearest_instance(obj_ids, frame_masks, anchor_xy, max_dist)
                    if found is not None:
                        cur_id, mask, _ = found
                if mask is not None:
                    centroid = _mask_centroid(mask)
                    if centroid is not None:
                        anchor_xy = centroid
                    masks[frame_idx] = mask
        finally:
            try:
                predictor.handle_request(request={"type": "close_session", "session_id": session_id})
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return masks


@parser.wrap()
def segment_dataset_sam3(cfg: SegmentSam3Config) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    src = LeRobotDataset(cfg.source_repo_id, root=cfg.source_root, return_uint8=True)

    predictor, device = _build_sam3_video_predictor(cfg)
    logging.info(f"SAM3.1 video predictor ready on {device}.")

    robot_type = getattr(getattr(src.meta, "info", None), "robot_type", "robot")
    dst = LeRobotDataset.create(
        cfg.output_repo_id,
        src.fps,
        root=cfg.output_root,
        robot_type=robot_type,
        features=src.features,
        use_videos=cfg.video and len(src.meta.video_keys) > 0,
        batch_encoding_size=cfg.batch_encoding_size,
    )

    num_episodes = src.num_episodes
    if cfg.max_episodes is not None:
        num_episodes = min(num_episodes, cfg.max_episodes)

    gripper_idx = _gripper_index(src)
    if cfg.grasp_only and gripper_idx < 0:
        logging.warning("grasp_only requested but no gripper channel found; keeping full episodes.")

    with VideoEncodingManager(dst):
        for episode_index in range(num_episodes):
            # Periodically rebuild the predictor to release accumulated GPU memory.
            if cfg.predictor_rebuild_every and episode_index and episode_index % cfg.predictor_rebuild_every == 0:
                import gc

                import torch

                del predictor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                predictor, device = _build_sam3_video_predictor(cfg)
                logging.info(f"Rebuilt SAM predictor before episode {episode_index} (memory reset).")

            start_index = int(_episode_value(src.meta, "dataset_from_index", episode_index))
            end_index = int(_episode_value(src.meta, "dataset_to_index", episode_index))
            full_num_frames = end_index - start_index

            bbox_xyxy = _episode_value(src.meta, GROUNDING_BBOX_KEY, episode_index)
            if bbox_xyxy is None:
                raise ValueError(f"Missing {GROUNDING_BBOX_KEY} for episode {episode_index}.")
            bbox_xyxy = [int(v) for v in bbox_xyxy]

            click_xy = _episode_value(src.meta, GROUNDING_CLICK_XY_KEY, episode_index)
            if click_xy:
                point_xy = [float(click_xy[0]), float(click_xy[1])]
            else:  # fall back to bbox center if this episode was recorded in box mode
                x1, y1, x2, y2 = bbox_xyxy
                point_xy = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

            dim_factor = cfg.dim_factor
            if dim_factor is None:
                dim_factor = float(_episode_value(src.meta, GROUNDING_DIM_FACTOR_KEY, episode_index, default=0.35))

            # Decode + propagate the FULL episode so the lift detector sees the
            # whole trajectory (we truncate afterward).
            images = []
            for frame_index in range(start_index, end_index):
                images.append(_to_numpy_image(src[frame_index][cfg.scene_camera_key]))

            tmp_dir = tempfile.mkdtemp(prefix=f"sam_ep{episode_index:06d}_")
            try:
                for i, img in enumerate(images):
                    Image.fromarray(img).save(Path(tmp_dir) / f"{i:05d}.jpg")

                if cfg.sam_backend == "sam3_tracker":
                    masks = _propagate_episode_tracker(
                        predictor, tmp_dir, full_num_frames, point_xy, images[0].shape, device
                    )
                else:
                    masks = _propagate_episode(
                        predictor, tmp_dir, full_num_frames, point_xy, cfg.sam_prompt,
                        cfg.redetect_max_dist_frac, cfg.seed_scan_frames, device,
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Find the LIFT frame (target berry leaves the table) from the mask
            # trajectory — the gripper signal is too noisy on multi-cycle grasps.
            # This is the shared phase switch: grasp ends / placement begins.
            need_lift = cfg.grasp_only or cfg.placement is not None
            lift_frame = _detect_lift_frame(masks, images[0].shape) if need_lift else None
            if lift_frame is None and need_lift and gripper_idx >= 0:  # fallback to gripper
                gripper = _gripper_series(src, start_index, end_index, gripper_idx)
                lift_frame = _detect_grasp_frame(gripper)

            if cfg.placement is not None:
                # Full-episode mode: keep everything, switch the render at lift.
                num_frames = full_num_frames
            elif cfg.grasp_only and lift_frame is not None:
                num_frames = min(full_num_frames, lift_frame + 1 + cfg.grasp_margin)
            else:
                num_frames = full_num_frames
                if cfg.grasp_only and lift_frame is None:
                    logging.warning(
                        f"Episode {episode_index}: no grasp/lift detected; keeping all {full_num_frames} frames."
                    )
            proc_end_index = start_index + num_frames
            # Frames [0:switch] are the grasp phase (strawberry overlay); frames
            # after `switch` are placement (raw RGB in "raw" mode).
            switch = lift_frame if lift_frame is not None else num_frames

            # Static click-circle fallback for any frame SAM left empty.
            fallback_mask = ellipse_mask_from_box(images[0].shape, bbox_xyxy)
            n_fallback = 0
            overlay_color = tuple(int(c) for c in cfg.overlay_color)
            # For placement="box": seed the tray bbox from frame 0 (fully visible),
            # then re-detect per frame (the tray drifts ~±3cm) with this fallback.
            last_tray_bbox = _detect_tray_bbox(images[0]) if cfg.placement == "box" else None

            n_placement = 0
            for i, frame_index in enumerate(range(start_index, proc_end_index)):
                item = src[frame_index]
                frame = {"task": item["task"]}

                grasp_phase = i <= switch  # strawberry overlay up to the lift
                mask = masks[i]
                if grasp_phase and (mask is None or not mask.any()):
                    mask = fallback_mask
                    n_fallback += 1

                for key, feature in src.features.items():
                    if key in DEFAULT_FEATURES:
                        continue
                    value = item[key]
                    if key == cfg.scene_camera_key:
                        if not grasp_phase and cfg.placement == "raw":
                            frame[key] = images[i]  # normal RGB placement frame
                            n_placement += 1
                        elif not grasp_phase and cfg.placement == "box":
                            # placement target = the tray: bbox + crosshair (no fill,
                            # so the bin interior / berry stays visible).
                            tray = _detect_tray_bbox(images[i])
                            if tray is not None:
                                last_tray_bbox = tray
                            tray = tray if tray is not None else last_tray_bbox
                            frame[key] = grounding_overlay_frame(
                                images[i],
                                None,
                                color=overlay_color,
                                alpha=cfg.overlay_alpha,
                                bbox_xyxy=tray,
                                draw_crosshair=tray is not None,
                            )
                            n_placement += 1
                        elif cfg.render_style == "dim":  # legacy debug render
                            frame[key] = grounding_frame(images[i], mask, dim_factor=dim_factor)
                        else:  # light visual prompt on normal-brightness RGB (training input)
                            frame[key] = grounding_overlay_frame(
                                images[i],
                                mask,
                                color=overlay_color,
                                alpha=cfg.overlay_alpha,
                                bbox_xyxy=(bbox_xyxy if cfg.draw_target_bbox else None),
                            )
                    elif feature["dtype"] in {"image", "video"}:
                        frame[key] = _to_numpy_image(value)
                    else:
                        frame[key] = _to_numpy_feature(value)

                dst.add_frame(frame)

            n_grasp = num_frames - n_placement
            logging.info(
                f"Episode {episode_index}: {num_frames}/{full_num_frames} frames "
                f"(lift@{lift_frame}; grasp={n_grasp}, placement={n_placement}), "
                f"{n_fallback} grasp frames fell back to click-circle "
                f"({100 * n_fallback / max(n_grasp, 1):.0f}%)."
            )

            episode_metadata = {
                GROUNDING_BBOX_KEY: bbox_xyxy,
                GROUNDING_CLICK_XY_KEY: [int(round(point_xy[0])), int(round(point_xy[1]))],
                GROUNDING_DIM_FACTOR_KEY: float(dim_factor),
                GROUNDING_POLICY_PROMPT_KEY: _episode_value(src.meta, GROUNDING_POLICY_PROMPT_KEY, episode_index),
                GROUNDING_SOURCE_CAMERA_KEY: _episode_value(src.meta, GROUNDING_SOURCE_CAMERA_KEY, episode_index),
                GROUNDING_SUCCESS_KEY: bool(_episode_value(src.meta, GROUNDING_SUCCESS_KEY, episode_index, default=False)),
                "grounding_source_dataset_repo_id": cfg.source_repo_id,
                "grounding_source_episode_index": int(episode_index),
                "grounding_segmentation_backend": ("sam3" if cfg.sam_backend == "sam3_tracker" else cfg.sam_version),
                "grounding_sam_backend": cfg.sam_backend,
                "grounding_sam_fallback_frames": int(n_fallback),
                # phase / render bookkeeping. grounding_lift_frame is the shared
                # phase-switch point (grasp overlay ends / placement begins).
                "grounding_phase": (
                    f"grasp_then_{cfg.placement}"
                    if cfg.placement is not None
                    else ("pre_grasp" if (cfg.grasp_only and lift_frame is not None) else "full_episode")
                ),
                "grounding_lift_frame": (int(lift_frame) if lift_frame is not None else -1),
                "grounding_grasp_frame": (int(lift_frame) if lift_frame is not None else -1),
                "grounding_source_num_frames": int(full_num_frames),
                "grounding_render_style": cfg.render_style,
                "grounding_overlay_color": list(overlay_color),
                "grounding_overlay_alpha": float(cfg.overlay_alpha),
            }
            dst.save_episode(episode_metadata=episode_metadata)

    dst.finalize()

    if cfg.push_to_hub and dst.num_episodes > 0:
        dst.push_to_hub(private=cfg.private)

    return dst


def main():
    register_third_party_plugins()
    segment_dataset_sam3()


if __name__ == "__main__":
    main()
