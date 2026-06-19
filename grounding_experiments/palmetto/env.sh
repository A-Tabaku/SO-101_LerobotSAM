#!/usr/bin/env bash
# ============================================================================
# Shared configuration for the pi0.5 fine-tune on Palmetto 2.
# Every other script sources this file. Edit the values in the CONFIG block,
# then never touch the rest.
# ============================================================================

# ---------------------------- CONFIG (edit me) ------------------------------

# Conda env name to create / use on Palmetto.
export PI05_ENV="pi05"

# Python version for the env. pi0.5 needs >= 3.12.
export PI05_PY="3.12"

# Where the LeRobot source tree lives ON PALMETTO (after rsync).
export PI05_REPO="/scratch/${USER}/SO-101_Lerobot"

# Scratch root for big stuff (HF cache, checkpoints, training outputs).
# Home dirs on Palmetto have small quotas -> keep everything heavy on scratch.
export PI05_SCRATCH="/scratch/${USER}/pi05"

# The base checkpoint to fine-tune from (a real HF repo, NOT gated).
export PI05_BASE="lerobot/pi05_base"

# Your dataset. Either a HuggingFace Hub repo id (org/name) that you pushed,
# or leave PI05_DATASET_ID and set PI05_DATASET_ROOT to a local LeRobot
# dataset directory on scratch. Set ONE of these.
export PI05_DATASET_ID="your_hf_username/so101_yourdataset"   # <-- change me
export PI05_DATASET_ROOT=""   # e.g. /scratch/$USER/datasets/so101_yourdataset

# GPU request. A100 80GB is strongly recommended for a full pi0.5 fine-tune.
# Format is <model>:<count>, e.g. a100:1, h100:1, l40s:1.
export PI05_GPU="a100:1"

# Weights & Biases. Set to "true" to log online (needs `wandb login` first &
# outbound internet). "false" runs without W&B. "offline" buffers locally.
export PI05_WANDB="false"

# ---------------------------- DERIVED (leave) -------------------------------

# Keep ALL HuggingFace downloads (datasets + model weights) on scratch so the
# small home quota never fills, and so compute nodes reuse what the login node
# fetched.
export HF_HOME="${PI05_SCRATCH}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1          # faster parallel downloads
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

export PI05_OUTPUT="${PI05_SCRATCH}/outputs"
export PI05_JOBLOG="${PI05_SCRATCH}/logs"

mkdir -p "${HF_HOME}" "${PI05_OUTPUT}" "${PI05_JOBLOG}" 2>/dev/null || true
