#!/usr/bin/env bash
# Download the Matbench tasks used by the pilot (official files, ~1-15 MB each).
set -euo pipefail
cd "$(dirname "$0")/../data" && mkdir -p matbench && cd matbench
for t in matbench_phonons matbench_perovskites matbench_log_gvrh; do
  [ -f "$t.json.gz" ] || curl -fL -O "https://ml.materialsproject.org/projects/$t.json.gz"
done
ls -lh
