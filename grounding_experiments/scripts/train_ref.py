import numpy as np, torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

OUT = "/home/summer2026/projects/lerobotprocessing/debug_grounded_rollout/train_ref"
import os; os.makedirs(OUT, exist_ok=True)

def hwc(x):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3): x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8: x = np.clip(x*255 if x.max() <= 1.01 else x, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)

m = LeRobotDatasetMetadata("Atabaku/so101-strawberry-grasp-rawplace-grounded")
print("state stats:")
st = m.stats["observation.state"]
names = m.features["observation.state"]["names"]
for i, n in enumerate(names):
    print(f"  {n}: min={float(st['min'][i]):.1f} max={float(st['max'][i]):.1f} mean={float(st['mean'][i]):.1f}")
print("action stats:")
at = m.stats["action"]
for i, n in enumerate(m.features["action"]["names"]):
    print(f"  {n}: min={float(at['min'][i]):.1f} max={float(at['max'][i]):.1f} mean={float(at['mean'][i]):.1f}")

ds = LeRobotDataset("Atabaku/so101-strawberry-grasp-rawplace-grounded")
it = ds[5]  # an early (grasp-phase) frame -> has the blue overlay
Image.fromarray(hwc(it["observation.images.scene"])).save(f"{OUT}/TRAIN_scene_grasp.png")
Image.fromarray(hwc(it["observation.images.wrist"])).save(f"{OUT}/TRAIN_wrist_grasp.png")
print("saved TRAIN_scene_grasp.png + TRAIN_wrist_grasp.png to", OUT)
print("frame5 state:", {n: round(float(np.asarray(it["observation.state"])[i]),1) for i,n in enumerate(names)})
