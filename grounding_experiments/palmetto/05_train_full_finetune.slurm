#!/bin/bash
#SBATCH --job-name pi05_full_finetune
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node h100:1
#SBATCH --output pi05_full_finetune_%j.log
# Action-expert ablation: full finetune (train_expert_only=FALSE) vs the
# expert-only baseline (control_raw, train_expert_only=TRUE). Same RAW control
# dataset as control_raw so the only variable is the expert flag.
# COMPUTE: full finetune trains the whole ~3B VLM -> ~50-65GB GPU + ~2x per-step
# compute. Needs a big/fast card: H100 (80GB) keeps batch 32 (clean vs control_raw).
# Fallbacks if H100 is unavailable or it OOMs: --gpus-per-node h200:1 (141GB, most
# headroom), or --gpus-per-node a100:1 --constraint gpu_a100_80gb with --batch_size=16
# (note: batch 16 changes the apples-to-apples comparison vs the batch-32 control).

module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-raw \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_full_finetune \
  --policy.push_to_hub=true \
  --output_dir=/scratch/atabaku/pi05_runs/pi05_full_finetune \
  --job_name=pi05_full_finetune \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=false \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=false
# Full finetune is MUCH heavier than expert-only (~18GB). If it CUDA-OOMs on the
# 80GB A100: drop --batch_size to 16 (note: changes apples-to-apples vs the
# batch-32 expert-only run) OR request a bigger GPU: --constraint gpu_h100_80gb
# / --gpus-per-node h200:1.
