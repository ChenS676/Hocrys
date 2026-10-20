#!/usr/bin/env bash
# HAICORE / HoreKa example. Submit:  sbatch --array=0-89 scripts/slurm_array.sh
#SBATCH --job-name=hocrys
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate   # adjust to your env
TASK_ID=$SLURM_ARRAY_TASK_ID bash scripts/run_pilot.sh
