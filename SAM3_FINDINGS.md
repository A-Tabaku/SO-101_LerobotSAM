# SAM3.1 strawberry-grounding — findings & current state (2026-06-10/11)

Working notes on grounding `Atabaku/so101-strawberry-raw` (50 episodes, SO-101
pick-a-strawberry-into-the-green-bin) into a SAM3.1-grounded dataset for π0.5.
Companion to the runbook [`SAM_SEGMENTATION_GPU.md`](./SAM_SEGMENTATION_GPU.md).
Pipeline code: [`src/lerobot/scripts/lerobot_segment_dataset_sam3.py`](./src/lerobot/scripts/lerobot_segment_dataset_sam3.py).

## TL;DR

The pipeline works end-to-end and is validated, but the full 50-episode run has
**not** been completed/pushed yet. Two big lessons today:

1. **The text concept word dominates quality.** `"strawberry"` fails to detect
   the berries on **31 of 50 episodes** (frame-0 target selection). `"red fruit"`
   detects all 3 berries and selects the clicked target on **50 of 50**. Default
   prompt is now `"red fruit"`. (Validated at the *detection* level across all 50;
   end-to-end propagation re-validation on the worst episodes is still pending —
   see Open issues.)
2. **The first full run OOM-crashed at episode 36/50** (never pushed). Two
   causes: torch 2.10 renamed the alloc env var, and the reused predictor leaks
   GPU memory across episodes. Both now fixed (correct env var + periodic
   predictor rebuild).

Nothing has been pushed to the Hub. `Atabaku/so101-strawberry-raw-grounded`
does not exist yet.

## Environment (don't rebuild from scratch)

- conda env **`so101-sam`** (cloned from `sam31-grounding`). torch 2.10+cu128 on
  RTX 5090 (Blackwell), CUDA OK. HF logged in as `Atabaku`.
- Pins that matter: `torchcodec==0.10.*` (0.11 needs torch 2.11), `av>=15,<16`
  (17 breaks `av.option`). numpy stays 1.26.4.
- `sam3` at `~/projects/rizon4sla/external/sam3`; sam3.1 multiplex checkpoint
  (3.3 GB) cached. Build with `build_sam3_multiplex_video_predictor(use_fa3=False)`
  (no flash-attn installed).
- **Run with `PYTORCH_ALLOC_CONF=expandable_segments:True`** (torch 2.10 renamed
  it from `PYTORCH_CUDA_ALLOC_CONF`, which is silently ignored).

## How the pipeline currently works

`lerobot_segment_dataset_sam3.py`, per episode:

1. **Decode** the episode's scene frames from the MP4 to a temp dir of JPEGs.
2. **Seed-frame scan** (`seed_scan_frames=30`): the opening frames can be
   motion-blurred so the detector finds nothing. Scan forward for the first
   frame where the text concept detects an instance whose mask covers / is near
   the stored `grounding_click_xy` (gated by `redetect_max_dist_frac` of the
   image diagonal). Seed there; earlier frames fall back.
3. **Propagate** forward from the seed frame (`propagate_in_video`). SAM3.1's
   multiplex tracker is concept-driven — a text prompt densely tracks every
   matching instance; the click picks which one is "the" target.
4. **Position-continuity tracking**: follow the target's object id while present;
   when the tracker drops it (e.g. grasped and re-detected under a NEW id),
   re-acquire the nearest detection to the last-known centroid, gated by
   `redetect_max_dist_frac` so it never jumps to a neighbouring berry.
5. **Fallback**: frames where the target can't be located use the static
   click-circle (`ellipse_mask_from_box` at the frame-0 bbox). Counted in
   `grounding_sam_fallback_frames` per episode (audit/filter later).
6. **Render** the grounded scene (target bright, rest dimmed by `dim_factor`),
   copy wrist/state/action unchanged, `save_episode` with grounding metadata.

Memory: `predictor_rebuild_every=8` rebuilds the predictor every 8 episodes to
release accumulated GPU memory; `start_session(offload_video_to_cpu=True)` keeps
decoded frames off the GPU.

### Current config defaults (`SegmentSam3Config`)

| field | default | note |
|---|---|---|
| `sam_prompt` | `"red fruit"` | **changed from "strawberry" today** — robust across all 50 |
| `redetect_max_dist_frac` | `0.08` | ~64px; tight enough to reject neighbour berries (~115px), loose enough to follow target into gripper/bin |
| `seed_scan_frames` | `30` | handle motion-blurred episode starts |
| `predictor_rebuild_every` | `8` | **added today** — bounds GPU memory (OOM ~ep36 otherwise) |
| `dim_factor` | `None` | uses per-episode value recorded at capture (default 0.35) |
| `max_episodes` | `None` | process all; set to N for spot-checks |
| `sam_version` | `"sam3.1"` | downloads gated `facebook/sam3.1` |

## What we learned (chronological)

- **SAM3.1 API differs from SAM2.** The installed multiplex build uses a
  session/request API (`handle_request` start_session/add_prompt, then
  `handle_stream_request` propagate_in_video), NOT `init_state`/`add_new_points`.
  Also needs a shim to drop `offload_state_to_cpu`, which the base predictor
  forwards but the multiplex model's `init_state` doesn't accept.
- **Point-only seeding does not propagate** in the multiplex model — object id
  vanishes after frame 0. Must seed with a **text concept**; the click only
  selects the instance.
- **Re-detection through the pick** matters. Without it, the berry is dropped at
  grasp and the static click-circle highlights the now-empty original location
  for the whole pick+place half (fallback 50% on ep0). Position-continuity
  re-acquisition cut ep0 to 4% and follows the berry into the gripper/bin.
- **Distance gate must be tight.** `0.15`×diag (~120px) let re-acquisition jump
  to a neighbouring berry (~115px away) when the target was occluded (ep2 ended
  highlighting the WRONG berry). `0.08` (~64px) fixed it — ep2 now ends on the
  correct berry in the bin, at the cost of more honest fallback during the
  fully-occluded carry.
- **Concept word is the biggest quality lever.** `"strawberry"`/`"berry"`/
  `"strawberries"` fail on many episodes; `"red fruit"` (and often
  `"red strawberry"`) detect all 3 berries reliably. Frame-0 target-selection
  failure rate: `"strawberry"` 31/50, `"red fruit"` 0/50.
- **Carry-occlusion is inherent.** When the gripper fully hides the berry during
  the carry (e.g. ep2), even re-detection can't find it — those frames are
  honest fallback (click-circle), and tracking recovers at the bin. Not a bug;
  filter by `grounding_sam_fallback_frames` if needed.

## First full-run result (DO NOT trust this output — superseded)

Run on 2026-06-10 with `sam_prompt="strawberry"`, gate 0.08, wrong env var.
- **Crashed at episode 36/50 — CUDA OOM. Never pushed.**
- Of 36 completed: 24 ≤5% fallback, but 8 >20% incl. ep22 & ep34 at 100%
  (those were `"strawberry"` detection failures, since fixed by `"red fruit"`).
- A stale local partial dataset may exist at
  `~/.cache/huggingface/lerobot/Atabaku/so101-strawberry-raw-grounded` — delete
  before any re-run.

## Open issues / caveats

- **End-to-end `"red fruit"` re-validation pending.** Detection scan is 50/50,
  but a propagation run on the worst episodes hasn't completed: the diagnostic
  harness OOM'd on the **748-frame ep2** before printing. That means a *single*
  long episode can peak near the 32 GB limit on its own — so `predictor_rebuild_every`
  (which bounds *cross-episode* accumulation) may not be the whole story; per-episode
  peak headroom on the longest episodes (ep2=748f, others 600–750f) needs a clean
  confirm under `PYTORCH_ALLOC_CONF=expandable_segments:True`. If single-episode
  peak is the limit, options: lower SAM image size, cap frames/chunk, or process
  long episodes in their own subprocess.
- Worth re-checking the gate (`0.08`) holds with `"red fruit"` (more/denser
  detections could change re-acquisition behaviour) — only spot-checked with
  `"strawberry"` so far.

## Next steps (when ready — NOT yet authorised to launch)

1. Confirm single-episode peak memory fits (clean run of ep2 with correct env
   var); re-validate `"red fruit"` end-to-end fallback on episodes 2,18,22,26,34.
2. Re-run all 50 with current defaults + `PYTORCH_ALLOC_CONF=expandable_segments:True`,
   push to `Atabaku/so101-strawberry-raw-grounded` (private). Delete the stale
   local partial first.
3. Audit per-episode `grounding_sam_fallback_frames`; high values = carry-occlusion.
4. Train π0.5 on the grounded repo (see `SAM_SEGMENTATION_GPU.md` §5).
