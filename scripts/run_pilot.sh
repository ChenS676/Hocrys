#!/usr/bin/env bash
# Pilot grid: 3 synthetic potentials + 2 Matbench tasks x 6 model variants x 3 seeds.
# Usage:  bash scripts/run_pilot.sh            (sequential)
#         TASK_ID=k bash scripts/run_pilot.sh  (run only job k; used by the SLURM array)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS="sw eam lj phonons perovskites"
MODELS=("pair" "ace --nu 1" "ace --nu 2" "linegraph" "mace --nu 1" "mace --nu 2")
SEEDS="0 1 2"
jobs=()
for d in $DATASETS; do for m in "${MODELS[@]}"; do for s in $SEEDS; do
  jobs+=("--data $d --model $m --seed $s")
done; done; done
echo "total jobs: ${#jobs[@]}"
if [ -n "${TASK_ID:-}" ]; then
  python -W ignore train.py ${jobs[$TASK_ID]} ${EXTRA:-}
else
  for j in "${jobs[@]}"; do python -W ignore train.py $j ${EXTRA:-}; done
fi
