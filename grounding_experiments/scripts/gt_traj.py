"""Ground-truth reference: in the REAL demos, at what horizon does shoulder_pan
diverge with the target berry position? Tells us if the 50-step chunk even covers
the target-dependent reach, and which joints carry the reach."""
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

REPO = "Atabaku/so101-strawberry-grasp-rawplace-grounded"
meta = LeRobotDatasetMetadata(REPO); ds = LeRobotDataset(REPO)
names = meta.features["action"]["names"]

def to_hwc(v):
    if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
    v = np.asarray(v)
    if v.ndim == 3 and v.shape[0] in (1, 3): v = np.transpose(v, (1, 2, 0))
    if v.dtype != np.uint8: v = np.clip(v*255 if v.max() <= 1.01 else v, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(v)

def target_x(scene):
    b = scene[..., 2].astype(int); r = scene[..., 0].astype(int)
    m = (b > 140) & (r < 90); xs = np.nonzero(m)[1]
    return float(xs.mean()) if xs.size > 50 else None

HOR = [10, 25, 49, 80, 120]
bxs = []
panH = {h: [] for h in HOR}   # GT shoulder_pan at each horizon
allH = {h: [] for h in HOR}   # GT full action vector at each horizon
for ep in range(len(ds.meta.episodes["dataset_from_index"])):
    a = int(ds.meta.episodes["dataset_from_index"][ep]); b = int(ds.meta.episodes["dataset_to_index"][ep])
    scene = to_hwc(ds[a]["observation.images.scene"]); bx = target_x(scene)
    if bx is None: continue
    bxs.append(bx)
    for h in HOR:
        idx = min(a + h, b - 1)
        act = np.asarray(ds[idx]["action"], dtype=np.float32)
        panH[h].append(act[0])           # shoulder_pan
        allH[h].append(act)

bxs = np.array(bxs)
print(f"{len(bxs)} episodes, berry_x spread {bxs.max()-bxs.min():.0f}px")
print("GT shoulder_pan vs berry_x by horizon (frames after start):")
for h in HOR:
    p = np.array(panH[h]); cc = np.corrcoef(bxs, p)[0,1]
    print(f"  frame +{h:>3}: corr={cc:+.3f}  pan_spread={p.max()-p.min():.1f} deg")
print("\nWhich joint diverges most with berry_x at frame +49 (corr):")
A = np.stack(allH[49])
for i, nm in enumerate(names):
    print(f"  {nm}: corr={np.corrcoef(bxs, A[:,i])[0,1]:+.3f}  spread={A[:,i].max()-A[:,i].min():.1f}")
