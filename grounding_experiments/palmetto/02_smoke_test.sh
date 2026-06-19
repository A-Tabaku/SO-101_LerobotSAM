#!/bin/bash
# Interactive smoke-test on a GPU node (catches env/data/config problems in ~2 min
# BEFORE you submit the long batch job). Run these from the LOGIN node.
#
#   salloc --cpus-per-task 8 --mem 64GB --time 01:00:00 --gpus a100:1   # 64GB: model load needs >32GB RAM
#   # ^ once you land on the compute node, run the rest:
#   module load anaconda3
#   source activate lerobot
#   bash 02_smoke_test.sh
set -e

# Make the conda-installed FFmpeg libs visible to torchcodec (decodes dataset videos).
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-grasp-rawplace-grounded \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --output_dir=/scratch/atabaku/pi05_runs/smoke \
  --job_name=smoke \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=50 \
  --wandb.enable=false

echo "Smoke test finished. If it ran 50 steps and loss moved, the pipeline is GREEN -> submit 03_train_fulltask.slurm"
