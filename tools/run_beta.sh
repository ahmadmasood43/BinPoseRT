#!/usr/bin/env bash
# Milestone Beta: the refinement rows A2 / A3 / A4 on top of Alpha's cached CNOS + FoundPose
# stages (no GPU needed), official bop_toolkit evaluation of each BOP CSV, refinement analysis
# (stratified before/after, gate precision, alpha/beta sweep on the val scenes) and galleries.
# Resumable: every stage is skipped once its _SUCCESS marker exists.
#
#   tools/run_beta.sh                          # A2 A3 A4 vs A1 -> outputs/<dataset>_beta_report.md
#   ONLY="A2" tools/run_beta.sh                # a subset of rows
#   ONLY=A5r BEFORE=A5 tools/run_beta.sh       # another before/after pair -> outputs/<dataset>_beta_report_A5r.md
#
# Prerequisites: data/bop/tless and the cached segment + coarse_pose stages of the BEFORE row
# (tools/run_alpha.sh), i.e. outputs/runs/<BEFORE>_<dataset>/run_manifest.json.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
rows="${ONLY:-A2 A3 A4}"
before="${BEFORE:-A1}"
dataset="${DATASET:-tless}"
# The default rows write the milestone report; any other selection gets its own file.
if [ -z "${ONLY:-}" ] && [ -z "${BEFORE:-}" ]; then
  report="${REPORT:-outputs/${dataset}_beta_report.md}"
  plot="outputs/${dataset}_beta_strata.png"
else
  report="${REPORT:-outputs/${dataset}_beta_report_$(echo "$rows" | tr ' ' '_').md}"
  plot="${report%.md}_strata.png"
fi
log_dir="outputs/runs"
mkdir -p "$log_dir"

# The BEFORE row is (re-)evaluated first so that before/after share one evaluate stage version;
# its GPU stages are cached, so this is CPU work only.
for exp in $before $rows; do
  echo "=== $exp on $dataset"
  uv run python tools/run.py "experiment=$exp" "dataset=$dataset" \
    2>&1 | tee "$log_dir/${exp}_${dataset}.log"
  manifest="$log_dir/${exp}_${dataset}/run_manifest.json"
  ev_dir="$(python3 -c "import json;print(json.load(open('$manifest'))['stages']['evaluate']['dir'])")"
  csv="$(ls "$ev_dir"/*.csv | head -1)"
  # Official evaluation and galleries are CPU work that does not need the next row's refine
  # stage; run them in the background and wait for all of them before the report.
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

uv run python tools/refine_report.py --dataset "$dataset" --before "$before" $rows \
  --plot "$plot" | tee "$report"
