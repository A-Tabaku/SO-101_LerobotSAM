"""Teacher-forced open-loop eval: run the policy on its OWN training frames and
compare predicted actions to recorded ground-truth. Decisive model-vs-distribution test."""
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.scripts.lerobot_rollout_grounded_pi05 import (
    GroundedRolloutConfig, _load_policy_and_processors, _resolve_task,
)
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR

REPO = "Atabaku/so101-strawberry-grasp-rawplace-grounded"
cfg = GroundedRolloutConfig()
meta = LeRobotDatasetMetadata(REPO)
task = _resolve_task(meta, cfg)
robot_type = getattr(meta, "robot_type", "so_follower")
names = meta.features["action"]["names"]
device = "cuda"

policy, preprocessor, postprocessor = _load_policy_and_processors(cfg, meta.stats)
ds = LeRobotDataset(REPO)

def frame_obs(it):
    frame = {f"{OBS_STR}.state": np.asarray(it[f"{OBS_STR}.state"], dtype=np.float32)}
    for ck in meta.camera_keys:
        v = it[ck]
        if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
        v = np.asarray(v)
        if v.ndim == 3 and v.shape[0] in (1, 3): v = np.transpose(v, (1, 2, 0))
        if v.dtype != np.uint8: v = np.clip(v*255 if v.max() <= 1.01 else v, 0, 255).astype(np.uint8)
        frame[ck] = np.ascontiguousarray(v)
    return frame

def predict_first(it):
    frame = frame_obs(it)
    policy.reset()
    with torch.inference_mode():
        obs = prepare_observation_for_inference(frame, torch.device(device), task, robot_type)
        chunk = policy.predict_action_chunk(preprocessor(obs))
        a0 = postprocessor(chunk[:, 0])
    pred = make_robot_action(a0.squeeze(0).cpu(), meta.features)
    return np.array([pred[k] for k in names], dtype=np.float32)

n_eps = min(8, ds.meta.total_episodes if hasattr(ds.meta, "total_episodes") else 8)
print(f"Eval across {n_eps} episodes (different berry layouts), grasp-phase frames:", flush=True)
per_ep = []
# Also track: do the FIRST-frame predicted actions DIFFER across episodes (=> conditions on layout)?
first_preds = []
for ep in range(n_eps):
    a = int(ds.meta.episodes["dataset_from_index"][ep])
    b = int(ds.meta.episodes["dataset_to_index"][ep])
    idxs = list(range(a, min(b, a + 80), 8))  # grasp-phase frames
    errs = []
    for i in idxs:
        it = ds[i]
        gt = np.asarray(it["action"], dtype=np.float32)
        errs.append(np.abs(predict_first(it) - gt))
    errs = np.stack(errs)
    per_ep.append(errs.mean())
    first_preds.append(predict_first(ds[a]))
    print(f"  ep{ep}: mean MAE = {errs.mean():.2f} deg ({len(idxs)} frames)", flush=True)

per_ep = np.array(per_ep)
fp = np.stack(first_preds)
print(f"\nMean MAE across {n_eps} episodes = {per_ep.mean():.2f} deg (std {per_ep.std():.2f})", flush=True)
print("Spread of first-frame predicted actions across episodes (std per joint, deg):", flush=True)
for k, s in zip(names, fp.std(0)):
    print(f"  {k}: std={s:.1f}", flush=True)
