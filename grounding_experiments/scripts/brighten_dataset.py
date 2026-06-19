#!/usr/bin/env python
"""Create a brightness-boosted copy of a grounded LeRobot dataset.

Dataset-to-dataset transform (no SAM, CPU only): decode each frame, apply a
gamma boost to EVERY camera image (scene + wrist), copy state/action/task
unchanged, and write a new dataset. Only brightness differs from the source, so
it's a clean single-variable ablation. Pushes to the Hub when done.
"""
import numpy as np

from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.scripts.lerobot_segment_dataset_sam3 import _episode_value, _to_numpy_feature, _to_numpy_image
from lerobot.utils.constants import DEFAULT_FEATURES

GAMMA = 0.6  # <1 brightens; mean ~113 -> ~154
# The "completely raw" control recording (no masks/boxes) — the same dataset the
# control_raw training run uses. Brightness is the only variable changed.
SRC = "Atabaku/so101-strawberry-raw"
DST = "Atabaku/so101-strawberry-raw-bright"


def brighten(img: np.ndarray) -> np.ndarray:
    return np.clip(255.0 * (img.astype(np.float32) / 255.0) ** GAMMA, 0, 255).astype(np.uint8)


def main():
    src = LeRobotDataset(SRC, return_uint8=True)
    robot_type = getattr(getattr(src.meta, "info", None), "robot_type", "robot")
    dst = LeRobotDataset.create(
        DST,
        src.fps,
        robot_type=robot_type,
        features=src.features,
        use_videos=len(src.meta.video_keys) > 0,
        batch_encoding_size=1,
    )
    with VideoEncodingManager(dst):
        for ep in range(src.num_episodes):
            start = int(_episode_value(src.meta, "dataset_from_index", ep))
            end = int(_episode_value(src.meta, "dataset_to_index", ep))
            for frame_index in range(start, end):
                item = src[frame_index]
                frame = {"task": item["task"]}
                for key, feature in src.features.items():
                    if key in DEFAULT_FEATURES:
                        continue
                    value = item[key]
                    if feature["dtype"] in {"image", "video"}:
                        frame[key] = brighten(_to_numpy_image(value))  # brighten ALL cameras
                    else:
                        frame[key] = _to_numpy_feature(value)
                dst.add_frame(frame)
            dst.save_episode(
                episode_metadata={
                    "brightness_gamma": float(GAMMA),
                    "brightness_source_repo_id": SRC,
                }
            )
            print(f"episode {ep}: {end - start} frames brightened (gamma={GAMMA}).", flush=True)
    dst.finalize()
    dst.push_to_hub(private=True)
    print(f"DONE: pushed {DST}")


if __name__ == "__main__":
    main()
