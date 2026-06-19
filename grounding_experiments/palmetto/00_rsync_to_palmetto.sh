#!/usr/bin/env bash
# ============================================================================
# Run this ON YOUR LOCAL WORKSTATION (not on Palmetto).
# It copies your LeRobot fork up to Palmetto scratch, skipping the heavy
# SAM/debug output dirs and the git history that you don't need on the cluster.
#
# Usage:
#   PALMETTO_USER=atabaku ./00_rsync_to_palmetto.sh
# (set PALMETTO_USER to your Clemson username; default falls back to $USER)
# ============================================================================
set -euo pipefail

PALMETTO_USER="${PALMETTO_USER:-$USER}"
PALMETTO_HOST="${PALMETTO_HOST:-slogin.palmetto.clemson.edu}"   # login node
LOCAL_REPO="${LOCAL_REPO:-/home/summer2026/projects/lerobotprocessing/SO-101_Lerobot}"
REMOTE_REPO="/scratch/${PALMETTO_USER}/SO-101_Lerobot"

echo ">> Syncing ${LOCAL_REPO}  ->  ${PALMETTO_USER}@${PALMETTO_HOST}:${REMOTE_REPO}"

# Make sure the remote scratch dir exists.
ssh "${PALMETTO_USER}@${PALMETTO_HOST}" "mkdir -p '/scratch/${PALMETTO_USER}'"

rsync -avhP \
  --exclude '.git/' \
  --exclude 'sam_*/' \
  --exclude '*.log' \
  --exclude 'outputs/' \
  --exclude 'wandb/' \
  --exclude '__pycache__/' \
  --exclude '*.egg-info/' \
  "${LOCAL_REPO}/" \
  "${PALMETTO_USER}@${PALMETTO_HOST}:${REMOTE_REPO}/"

# Also copy this whole palmetto_pi05/ helper dir up next to the repo.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rsync -avhP "${HELPER_DIR}/" \
  "${PALMETTO_USER}@${PALMETTO_HOST}:/scratch/${PALMETTO_USER}/palmetto_pi05/"

echo ">> Done. Now SSH in and run the setup:"
echo "     ssh ${PALMETTO_USER}@${PALMETTO_HOST}"
echo "     cd /scratch/${PALMETTO_USER}/palmetto_pi05 && bash 10_setup_env.sh"
