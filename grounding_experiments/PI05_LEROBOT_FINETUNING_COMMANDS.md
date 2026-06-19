# PI0.5 LeRobot Fine-Tuning Commands on Palmetto

This runbook captures the commands used to fine-tune `lerobot/pi05_base` on the
SO-101 strawberry datasets. These jobs use expert-only PI0.5 fine-tuning,
`pyav` video decoding, Hugging Face model uploads, and no W&B logging for new
runs.

## 1. Connect to Palmetto

```bash
ssh <your_clemson_username>@slogin.palmetto.clemson.edu
```

## 2. Activate the LeRobot Environment

Run this before using `lerobot-train`, `hf`, or checking Hub auth:

```bash
module load anaconda3
source activate lerobot
```

Check Hugging Face auth:

```bash
hf auth whoami
```

If needed, log in:

```bash
hf auth login
```

## 3. Create a Training Directory

```bash
mkdir -p ~/strawberrytrain
cd ~/strawberrytrain
```

## 4. Smoke Test Command

Use this in an interactive GPU session before a long run:

```bash
salloc --cpus-per-task 8 --mem 64GB --time 01:00:00 --gpus a100:1
module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-grasp-rawplace-grounded \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --output_dir=./outputs/smoke \
  --job_name=smoke \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=50 \
  --wandb.enable=false
```

Leave the interactive node when done:

```bash
exit
```

## 5. Full Task: Raw Placement Run

Dataset: `Atabaku/so101-strawberry-grasp-rawplace-grounded`

Hub model repo: `Atabaku/pi05_strawberry_rawplace`

```bash
cd ~/strawberrytrain

cat > train_rawplace.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name pi05_rawplace
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --constraint gpu_a100_80gb
#SBATCH --output pi05_rawplace_%j.log

module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-grasp-rawplace-grounded \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_rawplace \
  --policy.push_to_hub=true \
  --output_dir=./outputs/pi05_strawberry_rawplace \
  --job_name=pi05_strawberry_rawplace \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=false
EOF

sbatch train_rawplace.slurm
```

## 6. Full Task: Fully Segmented / Box-Place Run

Dataset: `Atabaku/so101-strawberry-grasp-boxplace-grounded`

Hub model repo: `Atabaku/pi05_strawberry_fullseg`

```bash
cd ~/strawberrytrain

cat > train_fully_segmented.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name pi05_fullseg
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --constraint gpu_a100_80gb
#SBATCH --output pi05_fullseg_%j.log

module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-grasp-boxplace-grounded \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_fullseg \
  --policy.push_to_hub=true \
  --output_dir=./outputs/pi05_strawberry_fullseg \
  --job_name=pi05_strawberry_fullseg \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=false
EOF

sbatch train_fully_segmented.slurm
```

## 7. Control: No Segmentation / Raw Dataset Run

Dataset: `Atabaku/so101-strawberry-raw`

Hub model repo: `Atabaku/pi05_strawberry_control_raw`

```bash
cd ~/strawberrytrain

cat > train_control_raw.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name pi05_control_raw
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --constraint gpu_a100_80gb
#SBATCH --output pi05_control_raw_%j.log

module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-raw \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_control_raw \
  --policy.push_to_hub=true \
  --output_dir=./outputs/pi05_strawberry_control_raw \
  --job_name=pi05_strawberry_control_raw \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=false
EOF

sbatch train_control_raw.slurm
```

## 8. Grasp-Only Baseline Run

Dataset: `Atabaku/so101-strawberry-grasponly-grounded`

Hub model repo: `Atabaku/pi05_strawberry_grasponly`

```bash
cd ~/strawberrytrain

cat > train_grasponly.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name pi05_grasponly
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96gb
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --constraint gpu_a100_80gb
#SBATCH --output pi05_grasponly_%j.log

module load anaconda3
source activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-grasponly-grounded \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=Atabaku/pi05_strawberry_grasponly \
  --policy.push_to_hub=true \
  --output_dir=./outputs/pi05_strawberry_grasponly \
  --job_name=pi05_strawberry_grasponly \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=false
EOF

sbatch train_grasponly.slurm
```

## 9. Queue and Log Monitoring

Check all jobs:

```bash
squeue -u $USER
```

Check one job:

```bash
squeue -j <jobid>
```

Watch logs:

```bash
tail -f pi05_rawplace_<jobid>.log
tail -f pi05_fullseg_<jobid>.log
tail -f pi05_control_raw_<jobid>.log
tail -f pi05_grasponly_<jobid>.log
```

Show recent log lines:

```bash
tail -n 80 pi05_fullseg_<jobid>.log
```

Stop watching a log without stopping the job:

```bash
Ctrl+C
```

## 10. Local Checkpoints

Training saves local checkpoints before pushing to Hugging Face.

Example checkpoint paths:

```bash
~/strawberrytrain/outputs/pi05_strawberry_rawplace/checkpoints/005000/pretrained_model/model.safetensors
~/strawberrytrain/outputs/pi05_strawberry_fullseg/checkpoints/005000/pretrained_model/model.safetensors
~/strawberrytrain/outputs/pi05_strawberry_control_raw/checkpoints/005000/pretrained_model/model.safetensors
```

List checkpoints:

```bash
find ~/strawberrytrain/outputs -path "*/pretrained_model/model.safetensors" -ls
```

Check where `last` points:

```bash
readlink -f ~/strawberrytrain/outputs/<run_name>/checkpoints/last
```

## 11. Storage / Home Quota Commands

Check home usage:

```bash
du -sh ~
du -h --max-depth=1 ~ | sort -h
```

Remove W&B local artifacts if they exist:

```bash
rm -rf ~/wandb
rm -rf ~/strawberrytrain/wandb
```

Delete old checkpoints only after a newer checkpoint exists. Example for a run
where `020000` or later exists:

```bash
rm -rf ~/strawberrytrain/outputs/pi05_strawberry_rawplace/checkpoints/005000
rm -rf ~/strawberrytrain/outputs/pi05_strawberry_rawplace/checkpoints/010000
rm -rf ~/strawberrytrain/outputs/pi05_strawberry_rawplace/checkpoints/015000
```

Do not delete the checkpoint that `last` points to.

## 12. Notes

- `--dataset.video_backend=pyav` avoids the previous `torchcodec` / FFmpeg shared
  library issue.
- `--policy.train_expert_only=true` means this is expert-only PI0.5 fine-tuning,
  not full VLM fine-tuning.
- `--wandb.enable=false` prevents W&B logging/artifact storage for these runs.
- `--policy.push_to_hub=true` pushes final policy weights to Hugging Face.
- If `batch_size=32` OOMs, change it to `--batch_size=16` and resubmit.
- For future large runs, prefer scratch output paths to avoid filling `/home`.
