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

"""Live, single-image SAM3 segmentation + single-target tracking for inference.

The offline dataset pipeline (``lerobot_segment_dataset_sam3.py``) used the SAM3
*video* predictor (``init_state`` / ``propagate_in_video``) over a directory of
frames. Live robot inference instead needs a *single image* segmented per
observation, so this module wraps SAM3's image API (``Sam3Processor``):

    set_image(rgb) -> set_text_prompt("red fruit") -> state["masks"/"scores"/"boxes"]

A policy trained on grasp-phase frames that already carry the blue overlay must
see that overlay live, so SAM runs once per action chunk and the resulting mask
feeds ``grounding_overlay_frame``. To follow the *same* berry the operator picked
(not whichever scores highest each frame), we lock onto a target at episode start
and then track it by centroid continuity — the same idea as the dataset
pipeline's position-continuity re-detection (``redetect_max_dist_frac``).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext as _nullcontext

import numpy as np

logger = logging.getLogger(__name__)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(x, y) pixel centroid of a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class Detection:
    __slots__ = ("mask", "score", "centroid")

    def __init__(self, mask: np.ndarray, score: float, centroid: tuple[float, float]):
        self.mask = mask
        self.score = score
        self.centroid = centroid


class LiveStrawberrySegmenter:
    """SAM3 single-image text-grounded segmenter with single-target tracking.

    Build once (loads the SAM3 image model), then per episode call ``lock_on``
    to acquire the target and ``track`` each chunk to follow it. ``track``
    returns ``None`` when the target is not found this frame (the mask-loss
    signal the caller uses for the grasp->placement phase switch).
    """

    def __init__(
        self,
        prompt: str = "red fruit",
        *,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        redetect_max_dist_frac: float = 0.08,
    ) -> None:
        self.prompt = prompt
        self.device = device
        self.redetect_max_dist_frac = redetect_max_dist_frac
        self.last_centroid: tuple[float, float] | None = None
        self.last_mask: np.ndarray | None = None

        # Lazy import so this module imports fine in envs without sam3.
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self._torch = torch
        dev = device if torch.cuda.is_available() else "cpu"
        self._use_cuda = dev != "cpu"
        logger.info("Loading SAM3 image model (device=%s, prompt=%r)...", dev, prompt)
        model = build_sam3_image_model(device=dev, checkpoint_path=checkpoint_path)
        self._processor = Sam3Processor(model, device=dev, confidence_threshold=confidence_threshold)
        logger.info("SAM3 image model ready.")

    # ── core detection ────────────────────────────────────────────────────────
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run text-grounded segmentation on one RGB frame -> list of detections."""
        torch = self._torch
        # The SAM3 image model is designed to run under AMP bfloat16 (its backbone
        # features are expected to already be bf16); Sam3Processor does not wrap
        # itself, so we provide the autocast context (the video predictors do this
        # internally, which is why they worked without it).
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_cuda
            else _nullcontext()
        )
        # Pass a PIL image: Sam3Processor's numpy path assumes CHW
        # (height, width = image.shape[-2:]), which mis-reads our HWC frames; the
        # PIL path reads image.size correctly.
        from PIL import Image

        pil = Image.fromarray(np.ascontiguousarray(image_rgb))
        with torch.inference_mode(), amp:
            state = self._processor.set_image(pil)
            state = self._processor.set_text_prompt(self.prompt, state)

        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or len(masks) == 0:
            return []
        masks_np = masks.detach().float().cpu().numpy().astype(bool)  # .float(): bf16 -> numpy-safe
        scores_np = scores.detach().float().cpu().numpy().astype(float) if scores is not None else np.ones(len(masks_np))

        dets: list[Detection] = []
        for m, s in zip(masks_np, scores_np, strict=False):
            m2 = m[0] if m.ndim == 3 else m  # squeeze a possible channel dim
            c = _centroid(m2)
            if c is None:
                continue
            dets.append(Detection(m2, float(s), c))
        return dets

    # ── target acquisition + tracking ──────────────────────────────────────────
    def reset(self) -> None:
        self.last_centroid = None
        self.last_mask = None

    def lock_on(
        self,
        image_rgb: np.ndarray,
        click_xy: tuple[float, float] | None = None,
        *,
        auto_rule: str = "score",
        gripper_xy: tuple[float, float] | None = None,
    ) -> bool:
        """Acquire the target on the first frame.

        With ``click_xy`` (operator-selected): pick the detection whose mask
        contains the click, else the one whose centroid is nearest the click.
        Without it (auto mode): pick by ``auto_rule`` —
        ``score`` (highest confidence), ``central`` (nearest image centre), or
        ``closest_to_gripper`` (nearest ``gripper_xy``). Seeds tracking state.
        Returns False if nothing was detected (caller keeps re-querying).
        """
        dets = self.detect(image_rgb)
        if not dets:
            return False

        if click_xy is not None:
            cx, cy = int(round(click_xy[0])), int(round(click_xy[1]))
            containing = [d for d in dets if 0 <= cy < d.mask.shape[0] and 0 <= cx < d.mask.shape[1] and d.mask[cy, cx]]
            target = min(containing or dets, key=lambda d: (d.centroid[0] - click_xy[0]) ** 2 + (d.centroid[1] - click_xy[1]) ** 2)
        elif auto_rule == "central":
            h, w = image_rgb.shape[:2]
            ic = (w / 2.0, h / 2.0)
            target = min(dets, key=lambda d: (d.centroid[0] - ic[0]) ** 2 + (d.centroid[1] - ic[1]) ** 2)
        elif auto_rule == "closest_to_gripper" and gripper_xy is not None:
            target = min(dets, key=lambda d: (d.centroid[0] - gripper_xy[0]) ** 2 + (d.centroid[1] - gripper_xy[1]) ** 2)
        else:  # "score"
            target = max(dets, key=lambda d: d.score)

        self.last_centroid = target.centroid
        self.last_mask = target.mask
        logger.info("Locked target at %s (score=%.3f, %d candidates).", tuple(round(v, 1) for v in target.centroid), target.score, len(dets))
        return True

    def track(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Follow the locked target on a new frame.

        Returns the boolean mask of the detection whose centroid is within
        ``redetect_max_dist_frac * image_diagonal`` of the last known centroid,
        or ``None`` if the target is lost this frame. Updates tracking state.
        """
        if self.last_centroid is None:
            raise RuntimeError("track() called before a successful lock_on().")
        dets = self.detect(image_rgb)
        if not dets:
            return None
        h, w = image_rgb.shape[:2]
        max_dist = self.redetect_max_dist_frac * float(np.hypot(h, w))
        lx, ly = self.last_centroid
        best = min(dets, key=lambda d: np.hypot(d.centroid[0] - lx, d.centroid[1] - ly))
        if np.hypot(best.centroid[0] - lx, best.centroid[1] - ly) > max_dist:
            return None  # nearest detection is too far -> target lost (occluded/grasped)
        self.last_centroid = best.centroid
        self.last_mask = best.mask
        return best.mask
