"""Visual-conditioning ablation: fix arm state + wrist image, swap ONLY the global
scene image (target berry in different positions). If predicted reach (shoulder_pan)
tracks the berry's image x, the policy uses vision. If not, it's proprioception-driven."""
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.scripts.lerobot_rollout_grounded_pi05 import GroundedRolloutConfig, _load_policy_and_processors, _resolve_task
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR

REPO = "Atabaku/so101-strawberry-grasp-rawplace-grounded"
cfg = GroundedRolloutConfig(); meta = LeRobotDatasetMetadata(REPO)
task = _resolve_task(meta, cfg); robot_type = "so_follower"; names = meta.features["action"]["names"]
device = "cuda"
policy, preprocessor, postprocessor = _load_policy_and_processors(cfg, meta.stats)
ds = LeRobotDataset(REPO)

def to_hwc(v):
    if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
    v = np.asarray(v)
    if v.ndim == 3 and v.shape[0] in (1, 3): v = np.transpose(v, (1, 2, 0))
    if v.dtype != np.uint8: v = np.clip(v*255 if v.max() <= 1.01 else v, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(v)

def target_x(scene):  # blue overlay (0,60,190) RGB -> high B, low R
    b = scene[..., 2].astype(int); r = scene[..., 0].astype(int)
    m = (b > 140) & (r < 90)
    xs = np.nonzero(m)[1]
    return float(xs.mean()) if xs.size > 50 else None

HORIZONS = [0, 10, 25, 49]
def predict_chunk_pan(state, wrist, scene):
    """Return shoulder_pan (deg) at several horizons within the predicted chunk."""
    frame = {f"{OBS_STR}.state": np.asarray(state, dtype=np.float32),
             "observation.images.scene": scene, "observation.images.wrist": wrist}
    policy.reset()
    with torch.inference_mode():
        obs = prepare_observation_for_inference(frame, torch.device(device), task, robot_type)
        chunk = policy.predict_action_chunk(preprocessor(obs))  # (1, H, A) normalized
        pans = []
        for h in HORIZONS:
            a = postprocessor(chunk[:, min(h, chunk.shape[1]-1)])
            pans.append(make_robot_action(a.squeeze(0).cpu(), meta.features)["shoulder_pan.pos"])
    return np.array(pans, dtype=np.float32)

# Fixed base state + wrist from episode 0 frame 0
base = ds[int(ds.meta.episodes["dataset_from_index"][0])]
base_state = np.asarray(base[f"{OBS_STR}.state"], dtype=np.float32)
base_wrist = to_hwc(base["observation.images.wrist"])

print("Fixed home state + wrist; ONLY scene (target berry pos) changes. shoulder_pan at chunk horizons:")
print(f"{'ep':>3} {'berry_x':>8} " + " ".join(f"pan@{h:<3}" for h in HORIZONS))
bxs, pans = [], []
for ep in range(min(12, len(ds.meta.episodes["dataset_from_index"]))):
    a = int(ds.meta.episodes["dataset_from_index"][ep])
    scene = to_hwc(ds[a]["observation.images.scene"])
    bx = target_x(scene)
    p = predict_chunk_pan(base_state, base_wrist, scene)
    print(f"{ep:>3} {('%.0f'%bx) if bx else 'n/a':>8} " + " ".join(f"{v:>6.1f}" for v in p))
    if bx is not None:
        bxs.append(bx); pans.append(p)

bxs = np.array(bxs); pans = np.stack(pans)
print(f"\nberry_x spread = {bxs.max()-bxs.min():.0f}px over {len(bxs)} layouts")
print("corr(berry_x, shoulder_pan) and pan spread, per horizon:")
for j, h in enumerate(HORIZONS):
    cc = np.corrcoef(bxs, pans[:, j])[0, 1]
    print(f"  horizon {h:>2}: corr={cc:+.3f}  pan_spread={pans[:,j].max()-pans[:,j].min():.1f} deg")
