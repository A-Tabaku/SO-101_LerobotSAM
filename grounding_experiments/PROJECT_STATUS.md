# Project status — SO-101 strawberry pick-and-place with π0.5 + SAM grounding
_Updated 2026-06-18. Handoff/status for AI context._

## Goal
Fine-tune π0.5 (pi05 VLA) on an SO-101 strawberry pick→place task, and test whether **SAM-based
visual grounding** (overlaying the target strawberry with a blue mask, and the tray with a box) at
both train and inference time improves on-robot performance vs raw frames.

## Hardware / environment
- **Robot:** SO-101 follower on `/dev/ttyACM0`. **Cameras:** Intel RealSense **D435 = scene/global**
  (serial `346522072484`), **D405 = wrist** (serial `335122272701`). Addressed by serial, not /dev/video.
- **Local conda env: `so101-sam`** — has `sam3`, `lerobot`(+`pi05`), `torch 2.10/cu128`, `transformers`,
  `pyrealsense2`, `pyserial`, `feetech-servo-sdk`. (The Palmetto training env is separate, named `lerobot`.)
- **Training:** Clemson Palmetto HPC; checkpoints → `/scratch/atabaku/pi05_runs/`; models pushed to HF under `Atabaku/`.

## Code (additions, no edits to core lerobot)
- `SO-101_Lerobot/src/lerobot/scripts/lerobot_rollout_grounded_pi05.py` — live inference + harvest loop.
- `SO-101_Lerobot/src/lerobot/grounding/live_segmenter.py` — SAM3 single-image segment + target tracking.
- Key flags: `--grounding` (true/false), `--placement_style` (raw|box), `--brighten_gamma`,
  `--target_mode` (click|auto), `--placement_revert`, `--run_tag`, `--dry_run`, `--episodes`.
- Per-run results → `debug_grounded_rollout/harvest_log_<run_tag>.jsonl`; QC frames `_scene.png`/`_wrist.png`.

## Datasets (HF, Atabaku/, ~50 eps each)
`so101-strawberry-raw` (control) · `-raw-bright` (γ0.6) · `-grasp-rawplace-grounded` (mask on grasp,
raw place = "fulltask") · `-grasp-boxplace-grounded` (mask + tray box = "fullseg") · `-grasponly-grounded`.

## On-robot results (success rate; ✅/❌ behaviors)
| Model | grounding | finetune | steps | grasp | overall |
|---|---|---|---|---|---|
| control_raw_15k | none | expert-only | 15k | 30%* | 20% (*assisted) |
| full_finetune 5k/10k/15k | none | **full** | sweep | 10/**30**/20% | same |
| bright_15k | none (γ0.6) | expert-only | 15k | 20% | 20% (no effect) |
| fullseg_15k | mask+box | expert-only | 15k | 20% | 10% |
| **fulltask_fullft_8k** | mask | **full** | 8k | **40%** | 20% ← best so far |
| fulltask_fullft_12k | mask | full | 12k | 10% | 0% |
| fulltask_fullft 20k | mask | full | 20k | **0%** | 0% (over-trained) |
| fullseg_fullft 8k/12k | mask+box | full | — | **untested on-robot** | — |

## ✅ What's WORKING
- **Live SAM grounding pipeline:** SAM3 single-image segments the target ~60 ms/chunk; overlay matches the
  training distribution; the overlaid frame is verified to be the actual policy input.
- **Full fine-tune ≫ expert-only.** Expert-only (frozen VLM) **ignores the overlay** (proprioceptive
  shortcut / causal confusion) → grounding is wasted on it.
- **Grounding helps only with an unfrozen VLM (full-FT).** The grounded full-FT model **conditions on the
  masked target** (turns toward the clicked berry; accurate on the middle berry).
- **Best config so far:** grounded **mask** + **full fine-tune** at **~8k steps** (40% grasp, on-target).
- pi05 inference plumbing (load from HF, processors rebuilt from dataset stats, chunked select_action, ~19 Hz).

## ❌ What's NOT working / open issues
- **Over-training kills it.** Performance peaks ~8–10k then declines (12k worse, **20k reverts to
  memorized berry positions and stops using vision** — over-specialization → causal confusion returns).
- **Grasp precision** on left/right berries is weak at low steps (turns the right way but misses).
- **Self-occlusion:** when the arm reaches in, it blocks the berry from the **global** camera → policy
  loses the target → miss. Caps how much global-camera grounding can help; wrist cam should carry more load.
- **Brightness:** no measurable effect (pi05 normalizes images internally).
- **Placement** weaker than grasp (place-given-grasp ~50%); the grasp→place phase switch needs tuning.

## ⚠️ Critical gotchas (for whoever runs this)
- **Calibration `wrist_roll` trap:** the SO-101 routine does NOT sweep `wrist_roll`, so its zero = whatever
  orientation the wrist is in at the "middle" step — it can land **90° off** and break ALL inference
  (arm reaches beside the berry). **Always verify rest-pose `wrist_roll ≈ −90` via a dry_run before
  trusting a run; never re-run calibration unless the wrist is set identically.** Fix if off: patch
  `wrist_roll.homing_offset` in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`
  by ±1024 steps (= 90°; register limit ±2047, so use the mod-4096 in-range equivalent).
- **Processors:** load policy then build pre/post-processors with `pretrained_path=None` + dataset stats —
  the checkpoint's saved processors error (step-name version skew in this env).
- **SAM3 image model:** must run under `torch.autocast(bf16)` and be fed a **PIL** image (its numpy path
  assumes CHW); cast bf16→float before numpy.
- **RealSense:** script `hardware_reset`s both cams on startup (they get stuck after V4L2/OpenCV access).
- **Serial port:** `sudo chmod 666 /dev/ttyACM0` each session (user not in `dialout`).

## Methodology notes
Run ≥10 episodes per model (n=3–5 is noisy), **same berry layouts across models**, autonomous (no
hand-assist), strict scoring. Full results + reasoning in `debug_grounded_rollout/AB_results.md`.

## Next tests for a complete comparison
1. `fullseg_fullft_8k` (mask+box, full-FT) — **never tested on-robot**; completes the raw vs mask vs mask+box 3-way.
2. `fullseg_fullft_12k` — box-lineage step curve.
3. Re-run the best ones at n≥10 with matched layouts for presentable numbers.
