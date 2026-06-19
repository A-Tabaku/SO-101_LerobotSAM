"""Does the demo START POSE already encode the target? If frame-0 shoulder_pan
correlates with the target berry's image x, proprioception leaks the target ->
the model can solve training without vision -> learns the proprioceptive shortcut."""
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

REPO = "Atabaku/so101-strawberry-grasp-rawplace-grounded"
meta = LeRobotDatasetMetadata(REPO); ds = LeRobotDataset(REPO)
names = meta.features["observation.state"]["names"]

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

n = len(ds.meta.episodes["dataset_from_index"])
print(f"{n} episodes. Frame-0 state vs target berry image-x:")
print(f"{'ep':>3} {'berry_x':>8} " + " ".join(f"{nm.split('.')[0]:>9}" for nm in names))
bx_list, st_list = [], []
for ep in range(n):
    a = int(ds.meta.episodes["dataset_from_index"][ep])
    it = ds[a]
    scene = to_hwc(it["observation.images.scene"])
    bx = target_x(scene)
    st = np.asarray(it["observation.state"], dtype=np.float32)
    print(f"{ep:>3} {('%.0f'%bx) if bx else 'n/a':>8} " + " ".join(f"{v:>9.1f}" for v in st))
    if bx is not None:
        bx_list.append(bx); st_list.append(st)

bx = np.array(bx_list); st = np.stack(st_list)
print(f"\ncorr(frame0 berry_x, frame0 state) over {len(bx)} episodes:")
for i, nm in enumerate(names):
    print(f"  {nm}: corr={np.corrcoef(bx, st[:, i])[0,1]:+.3f}  (state spread {st[:,i].max()-st[:,i].min():.1f} deg)")
