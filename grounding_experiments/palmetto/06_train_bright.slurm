#!/bin/bash
#SBATCH --job-name pi05_bright
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --output pi05_bright_%j.log
# Brightness ablation: brightened raw control (gamma 0.6) vs control_raw (normal
# brightness). Same config as control_raw (train_expert_only=true) so the only
# variable is image brightness.
# COMPUTE: expert-only is light (~18GB GPU), so any A100 (40 or 80GB) is plenty —
# no 80GB constraint, which lets it schedule faster.

module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-raw-bright \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_bright \
  --policy.push_to_hub=true \
  --output_dir=/scratch/atabaku/pi05_runs/pi05_bright \
  --job_name=pi05_bright \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=false
