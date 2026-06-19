#!/bin/bash
#SBATCH --job-name pi05_strawberry_fulltask
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --constraint gpu_a100_80gb
#SBATCH --output pi05_fulltask_%j.log
# If your PI's allocation requires it, add:  #SBATCH --account <your_account>
# Check your max walltime; if 48h is too long, lower --time (and --steps).

module load anaconda3
source activate lerobot
# Make conda-installed video libs visible to the decoder stack.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-grasp-rawplace-grounded \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_fulltask \
  --policy.push_to_hub=true \
  --output_dir=/scratch/atabaku/pi05_runs/pi05_strawberry_fulltask \
  --job_name=pi05_strawberry_fulltask \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=true
# If you hit CUDA OOM on the 80GB A100, drop --batch_size to 16 (or request h100/h200),
# and/or add --policy.train_expert_only=true.
