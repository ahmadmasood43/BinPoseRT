#!/usr/bin/env bash
# Milestone Gamma on T-LESS (multi-view debug set, D5) from the cached A2 chain (CNOS + FoundPose +
# pt2plane refinement): the AR-vs-views rows, the extrinsic-perturbation sweep, official
# bop_toolkit scores and the report. Resumable: every stage is skipped once its _SUCCESS exists.
#
#   tools/run_gamma.sh                    # curve rows + sweep -> outputs/<dataset>_gamma_report.md
#   ROWS=curve tools/run_gamma.sh         # only the AR-vs-views rows
#   ROWS=sweep tools/run_gamma.sh         # only the extrinsic sweep
#   DATASET=xyzibd MULTIVIEW=bop25 tools/run_gamma.sh
#
# Row naming: A6_k<n>_<fusion> (fusion in none|best|mean) and A7_k<n> (mean + joint ICP), so every
# point of the curve is one tools/run.py invocation with its own run_manifest.json (D12).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
dataset="${DATASET:-tless}"
multiview="${MULTIVIEW:-strided}"
rows="${ROWS:-curve sweep}"
ks="${KS:-1 2 3 4}"
sweep_k="${SWEEP_K:-4}"
sweep_t="${SWEEP_T:-0 1 2 5 10}"
sweep_r="${SWEEP_R:-0 0.1 0.25 0.5 1}"
log_dir="outputs/runs"
mkdir -p "$log_dir"

run_row() {  # name, extra overrides...
  local name="$1"; shift
  echo "=== $name on $dataset ($*)"
  uv run python tools/run.py "experiment.name=$name" "dataset=$dataset" "multiview=$multiview" "$@" \
    2>&1 | tee "$log_dir/${name}_${dataset}.log" | grep -v "^\[" | tail -3
}

official() {  # name: bop_toolkit on the row's CSV, restricted to the view-group images
  local name="$1"
  local manifest="$log_dir/${name}_${dataset}/run_manifest.json"
  local ev_dir csv
  ev_dir="$(python3 -c "import json;print(json.load(open('$manifest'))['stages']['evaluate']['dir'])")"
  csv="$(ls "$ev_dir"/*.csv | head -1)"
  if [ ! -f "$ev_dir/bop_eval/$(basename "$csv" .csv)/scores_bop19.json" ]; then
    BOP_TARGETS="$ev_dir/targets_subset.json" tools/bop_eval.sh "$csv" \
      > "$log_dir/${name}_${dataset}.bop_eval.log" 2>&1 || echo "official eval failed: $name"
  fi
}

if [[ " $rows " == *" curve "* ]]; then
  for k in $ks; do
    fusions="none best mean"
    [ "$k" = 1 ] && fusions="none"
    for f in $fusions; do
      run_row "A6_k${k}_${f}" experiment=A6 "fusion=$f" "multiview.params.groups.n_views=$k"
      official "A6_k${k}_${f}" &
    done
    if [ "$k" != 1 ]; then
      run_row "A7_k${k}" experiment=A7 "multiview.params.groups.n_views=$k"
      official "A7_k${k}" &
    fi
  done
fi

if [[ " $rows " == *" sweep "* ]]; then
  # calibration-error sweep: k views, weighted mean, core evaluator without VSD (MSSD/MSPD only)
  for t in $sweep_t; do
    for r in $sweep_r; do
      run_row "A6_sweep_t${t}_r${r}" experiment=A6 fusion=mean \
        "multiview.params.groups.n_views=$sweep_k" \
        "multiview.params.extrinsic_noise.t_mm=$t" "multiview.params.extrinsic_noise.deg=$r" \
        evaluate.with_vsd=false
    done
  done
fi
wait

uv run python tools/multiview_report.py --dataset "$dataset" --ks $ks --sweep-k "$sweep_k" \
  --plot "outputs/${dataset}_gamma" | tee "outputs/${dataset}_gamma_report.md"
echo "gamma done: outputs/${dataset}_gamma_report.md"
