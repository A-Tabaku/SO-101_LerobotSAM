#!/bin/bash
# One-time environment setup on Palmetto 2 (run on the LOGIN node).
# Usage:  bash 01_setup_env.sh
set -e

# If 'anaconda3' isn't found, run `module avail anaconda` and use the exact name.
module load anaconda3

conda create -n lerobot python=3.12 -y
source activate lerobot                      # Palmetto uses `source activate`, not `conda activate`

# Stock LeRobot + pi0/pi0.5 deps + dataset (video decode) deps. NOT our fork —
# training only needs upstream lerobot; the grounded datasets are plain LeRobot datasets.
#
# Install from git MAIN, not PyPI: the current PyPI release renamed a processor step
# ('relative_actions_processor' -> 'delta_actions_processor') and can't load the
# lerobot/pi05_base checkpoint. main re-added it for backward compatibility.
pip install "lerobot[pi,dataset] @ git+https://github.com/huggingface/lerobot.git"

# torchcodec (decodes the dataset's MP4 videos) needs FFmpeg's shared libs, which
# aren't on the compute nodes by default. Provide them in the env. (Alternatively,
# `module load ffmpeg` at runtime instead of this conda install.)
conda install -c conda-forge ffmpeg -y

# Log in so compute nodes can pull your private dataset + the pi05_base weights.
# Paste a HF token (https://huggingface.co/settings/tokens) with READ access.
hf auth login

echo "Setup done. Verify with:  python -c 'import lerobot, torch; print(torch.__version__, torch.cuda.is_available())'"
