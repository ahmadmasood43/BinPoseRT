#!/usr/bin/env bash
# Milestone Alpha on the GPU machine, end to end and resumable (every stage is skipped once its
# _SUCCESS marker exists): CNOS + FoundPose + MegaPose adapters, A0 / A1 / A5 rows, official
# bop_toolkit evaluation of each BOP CSV, failure galleries and the results table.
#
#   tools/run_alpha.sh              # everything
#   ONLY="A0 A1" tools/run_alpha.sh # a subset of rows
#
# Prerequisites: data/bop/tless (tools/download_bop.py tless --parts models test_primesense_bop19),
# data/checkpoints/{sam_vit_h_4b8939.pth,dinov2_vitl14_pretrain.pth,megapose/megapose-models},
# envs/{cnos,foundpose,megapose} (tools/setup_estimator_env.sh <name>) or the docker images.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
rows="${ONLY:-A0 A1 A5}"
dataset="${DATASET:-tless}"
log_dir="outputs/runs"
mkdir -p "$log_dir"

for exp in $rows; do
  echo "=== $exp on $dataset"
  uv run python tools/run.py "experiment=$exp" "dataset=$dataset" run_external=true \
    2>&1 | tee "$log_dir/${exp}_${dataset}.log"
  manifest="$log_dir/${exp}_${dataset}/run_manifest.json"
  ev_dir="$(python3 -c "import json;print(json.load(open('$manifest'))['stages']['evaluate']['dir'])")"
  csv="$(ls "$ev_dir"/*.csv | head -1)"
  # Official evaluation and gallery are CPU work: run them in the background so the next row's
  # GPU stages start immediately; wait for all of them before the report.
  (
    if [ ! -f "$ev_dir/bop_eval/$(basename "$csv" .csv)/scores_bop19.json" ]; then
      tools/bop_eval.sh "$csv" > "$log_dir/${exp}_${dataset}.bop_eval.log" 2>&1
    fi
    if [ ! -f "$ev_dir/gallery/gallery.png" ]; then
      uv run python tools/gallery.py "$ev_dir" --n 20 >> "$log_dir/${exp}_${dataset}.bop_eval.log" 2>&1
    fi
  ) &
done
wait

uv run python tools/report.py --dataset "$dataset" $rows | tee "outputs/${dataset}_alpha_report.md"
