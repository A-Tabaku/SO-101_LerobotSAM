# SO-101 strawberry pi0.5 — on-robot inference results (2026-06-17)

**Setup:** SO-101 follower; RealSense **D435 = scene/global**, **D405 = wrist**; task = *"pick up the
highlighted strawberry and place into the green bin"*; **3 berries** in scene; **10 episodes/model**,
60 s each, ~19 Hz, pi0.5 open-loop chunks (n_action_steps=50). Raw models run with `--grounding=false`;
grounded models with live SAM strawberry mask (+ green-tray box+crosshair for the fullseg/boxplace
model). Scoring: **grasp** = berry lifted off table, **place** = landed in bin. Fresh (no-saved)
calibration; autonomous unless noted.

## Results summary

| Model | Dataset | Finetune | Steps | Grasp | Overall | Notes |
|---|---|---|---|---|---|---|
| `pi05_strawberry_control_raw_15k` | raw | expert-only | 15k | **30%\*** | 20% | \*operator hand-fed berry into gripper on misses → logged grasp 40% corrected to 30%; **assisted** |
| `pi05_full_finetune_15k` | raw | **full** | 15k | 20% | 20% | autonomous |
| `pi05_bright_15k` | raw-bright (γ0.6) | expert-only | 15k | 20% | 20% | brightening applied live; no detectable difference vs control |
| `pi05_strawberry_fullseg_15k` | grasp-boxplace (grounded) | expert-only | 15k | 20% | 10% | SAM mask→tray box; phase switch fired only **1/10** (box prompt under-exercised) |
| `pi05_full_finetune_10k` | raw | **full** | 10k | **30%** | **30%** | best run; place-given-grasp 3/3; still a left bias but noticeably milder than 5k (visual targeting partly emerged) |
| `pi05_full_finetune_5k` | raw | **full** | 5k | 10% | 10% | undertrained; left-collapse + occlusion (see below) |
| `pi05_fulltask_fullft_8k` | grasp-rawplace (grounded) | **full** | 8k | **40%** | 20% | **grounded full-finetune** — best autonomous grasp; place-given-grasp 2/4; clearly conditions on the masked target (see below). Re-run post-wrist-fix (revert ON): 2/3 grasp (n=3). |
| `pi05_fulltask_fullft_12k` | grasp-rawplace (grounded) | **full** | 12k | 10% | 0% | declined vs 8k (on-target but imprecise) — over-specialized past the peak |
| `pi05_fulltask_fullft` (20k, final) | grasp-rawplace (grounded) | **full** | 20k | **0% (0/2)** | 0% | OVER-TRAINED: reverts to **memorized/previous strawberry positions instead of using vision** — ignores where the berry actually is. (Earlier logged grasp=True was a mis-input; both episodes were misses.) |

Latency all healthy: ~107–118 ms/chunk, ~19 Hz; SAM ~60 ms/chunk (grounded runs only).

## Per-model qualitative notes

- **`pi05_fulltask_fullft_8k` (grounded SAM mask + FULL finetune, 8k) — KEY RESULT:** best autonomous
  grasp so far (**40%**). Operator: **accurate on the MIDDLE strawberry**, but for the **left / right**
  berries it's inaccurate and usually messes up the grasp — **however it does turn in the correct
  direction**, so it *has* learned to localize the masked target; it just lacks precision at 8k
  (undertrained). This is the **first clear evidence that grounding is actually used once the VLM is
  unfrozen** — contrast the frozen-VLM `fullseg_15k`, which ignored the overlay entirely.
- **`pi05_full_finetune_5k` (5k ≈ ~6 epochs):** **~half** the episodes the arm **fixates LEFT** — reaches
  to the left side and tries to grab even when **no strawberry is there** (positional collapse / fixed
  canonical reach, not visually localizing the target). The **other ~half** it goes to the *correct*
  berry area and attempts a pickup but usually **fails** — operator observation: **the arm occludes the
  berry from the global D435 during approach**, the model loses the target, and misses the grasp.
  → undertrained: only partial visual conditioning, compounded by self-occlusion.
- **`pi05_full_finetune_10k`:** best success (30%); **still has a left bias but noticeably milder than
  5k** — visual targeting is partly online by 10k (it reaches actual berry positions more often, though
  the leftward pull hasn't fully gone).
- **`pi05_full_finetune_15k`:** grasps autonomously; **no left-collapse** — so the left bias fades
  progressively across 5k→10k→15k as visual conditioning strengthens.
- **`pi05_strawberry_fullseg_15k`:** grounding on a **frozen** VLM bought nothing (≤ full-finetune
  without grounding). Phase switch barely fired, so the tray-box placement prompt was hardly tested.
- **`pi05_bright_15k`:** brightness made no difference — same 20% as control. pi0.5 normalizes images
  internally (washes out a global gamma shift) and exposure was never the bottleneck.
- **`pi05_strawberry_control_raw_15k`:** could not grasp reliably unaided — needed operator assistance.

## Calibration root-cause (2026-06-18) — important
A long run of "reaches left/right of the berry / grabs the table / 0%" failures was traced to a
**`wrist_roll` calibration offset of 90°**, NOT the model or the inference code. `wrist_roll` is the
one joint the SO-101 routine does **not** sweep, so its zero is whatever orientation the wrist is in
at the "middle" step — it landed 90° off (rest read `wrist_roll≈0` vs training `≈−90`). Fixed by
patching `wrist_roll.homing_offset` (sign-magnitude, ±2047 limit → used `−1419`, the in-range
equivalent of +1024 steps/+90°); rest then read **−90.1** ✓. After the fix, `fulltask_fullft_8k`
grasped 2/3 again. **Lesson:** verify rest-pose `wrist_roll≈−90` (dry_run) before trusting any run;
always reuse the calibration file (never re-run calibration unless the wrist is set identically).
Grounded full-FT (fulltask) sweep: **8k best (40%) → 12k 10% → 20k 0% (0/2)** — peaks at 8k, declines
after. **The 20k failure mode is diagnostic:** over-trained, it **reverts to memorized/previous berry
positions instead of using vision** (the proprioceptive shortcut returns at the over-trained extreme).
So the decline-after-peak = over-specialization that progressively *stops using the camera* — the same
causal-confusion failure seen in the 30k expert-only model, just reached via too many full-FT steps. My placement-revert code was ruled out (8k
grasped 2/3 with revert ON; 12k/20k fail in the grasp phase, which the revert doesn't touch).

## Key findings

1. **Full finetune > expert-only** for autonomy (expert-only needed hand-feeding to grasp). With a
   **frozen** VLM, neither **grounding** (fullseg) nor **brightness** moved the needle — all stuck ~10–20%.
2. **Grounding only matters if the VLM is unfrozen** — expert-only models ignore the overlay
   (causal-confusion / proprioceptive shortcut; the 30k expert-only `fulltask` ignored the global camera
   in offline ablations). **Now confirmed on-robot:** the grounded **full-finetune** `pi05_fulltask_fullft_8k`
   hit **40% grasp (best autonomous)** and visibly **conditions on the masked target** — accurate on the
   middle berry, turns the right way for left/right (just imprecise at 8k). The frozen-VLM `fullseg`
   ignored the same overlay. → grounding **does** help once the VLM is trainable; needs more steps for
   precision. (`fullseg_fullft` 8k/12k still to test.)
3. **Brightness: no effect.**
4. **Step count (full-finetune lineage, raw):** non-monotonic — **5k 10%** (undertrained, left-collapse)
   → **10k 30%** (best; left bias milder, visual targeting partly online) → **15k 20%** (dip; no left bias). Peak at ~10k; the 15k dip
   suggests mild over-specialization beyond ~10k (supports the fewer-epochs intuition). n=10 so the
   10k-vs-15k gap is within noise, but 10k is the clear leader and 5k is clearly undertrained.
   (30k checkpoint still available to confirm the downward trend.)
5. **Recurring failure mode — global-camera self-occlusion:** when the arm reaches in, it blocks the
   berry from the global D435, the policy loses the target, and the grasp fails. This is a
   viewpoint/hardware limitation affecting all global-camera-dependent models. Mitigations to consider:
   lean on the **wrist (D405)** camera (sees the berry up close during approach), and/or reposition the
   global camera. Also caps how much *global*-camera grounding can help — once occluded, the overlay is gone too.

## Models / data on HF (Atabaku)
pi05 lineages: `control_raw_{15k,20k}` (expert-only raw), `full_finetune_{5k,10k,15k,30k}` (full raw),
`bright_{5k,10k,15k,30k}` (expert-only raw-bright), `fulltask` (expert-only grasp-rawplace, 30k),
`fullseg_{10k,15k,20k,30k}` (expert-only grasp-boxplace). In training: `fullseg_fullft`, `fulltask_fullft`
(full finetune, grounded, 20k, → scratch).
