# SO-101 Strawberry Pick-and-Place — π0.5 + SAM Visual Grounding
### On-robot A/B results (2026-06-18)

## Headline
**Best model: full visual grounding (strawberry mask + tray box) + full fine-tune @ ~8k steps → 60% success.**
Grounding helps — but **only when the vision backbone is fine-tuned (not frozen)**, and only up to a
**step sweet spot (~8k)**; training longer makes it *worse*.

---

## Setup
SO-101 arm, RealSense global + wrist cameras. Task: *"pick up the highlighted strawberry and place it in
the green bin"*, 3 berries in scene. Policy = π0.5 (PaliGemma VLM + flow-matching action expert), fine-tuned
on ~50 demos. SAM3 segments the target live and overlays a blue mask (grasp) / box+crosshair on the tray
(placement), reproducing the training overlay. Metric = on-robot success over N episodes (autonomous).

---

## Results

### Full fine-tune (vision unfrozen) — the regime that works
| Grounding | Model | Steps | Grasp | **Overall** | N |
|---|---|---|---|---|---|
| None (raw) | full_finetune | 5k | 10% | 10% | 10 |
| None (raw) | full_finetune | 10k | 30% | **30%** | 10 |
| None (raw) | full_finetune | 15k | 20% | 20% | 10 |
| Mask | fulltask_fullft | 8k | **40%** | 20% | 10 |
| Mask | fulltask_fullft | 12k | 10% | 0% | 10 |
| Mask | fulltask_fullft | 20k | 0% | 0% | 3 |
| **Mask + box** | **fullseg_fullft** | **8k** | **60%** | **60%** | 5 |

### Expert-only (vision FROZEN) — grounding is wasted here
| Grounding | Model | Steps | Grasp | Overall | N |
|---|---|---|---|---|---|
| None (raw control) | control_raw | 15k | 36%* | 18% | 11 |
| None (raw, γ0.6 bright) | bright | 15k | 20% | 20% | 10 |
| Mask + box | fullseg | 15k | 20% | 10% | 10 |

\* operator-assisted on misses → inflated; true autonomous grasp was lower.

---

## Key findings

1. **Full fine-tune ≫ expert-only.** Freezing the VLM makes the policy lean on proprioception and **ignore
   the visual overlay** (causal confusion). Only with the VLM unfrozen does the model actually *use* the mask.

2. **Grounding helps — when the VLM can use it.** Best overall, all full-FT @ ~8k:
   **mask + box (60%) > raw (30%) > mask-only (20% overall).**
   - The **mask** grounds the *grasp* → high grasp rate (40%).
   - Adding the **tray box** grounds the *placement* → placement succeeds far more often → highest overall (60%).

3. **There's a training sweet spot (~8k steps).** Performance **peaks then declines**: mask full-FT
   8k(40%) → 12k(10%) → 20k(0%). At 20k the over-trained model **reverts to memorized berry positions and
   stops using vision** — over-specialization re-creates the frozen-VLM failure.

4. **Brightness preprocessing: no effect** (π0.5 normalizes images internally; exposure was never the bottleneck).

5. **Dominant failure mode = self-occlusion:** as the arm reaches in, it blocks the berry from the *global*
   camera and the policy loses the target. This caps global-camera grounding and argues for leaning on the
   wrist camera near the grasp.

---

## Takeaway for next steps
- Train/deploy the **mask+box grounded model, full fine-tune, ~8k steps** (the 60% config).
- Re-run the top configs at **N≥10** with matched berry layouts to tighten the numbers (fullseg_fullft was N=5).
- Mitigate global-camera self-occlusion (wrist-camera weighting / camera placement).

_Caveats: small N per cell (5–11); the control_raw grasp number was operator-assisted. Trends are
consistent across the sweep, but treat single-percentage gaps as indicative, not significant._
