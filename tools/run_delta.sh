#!/usr/bin/env bash
# Milestone Delta (confidence and Verdict, D11) from the cached Gamma rows: the labelled tables,
# the fitted models and the A8 rows (A6 chain + confidence stage, ranked by Confidence) with the
# Model-H-weighted variant A8w. Resumable: pipeline stages are skipped once their _SUCCESS exists,
# tables once build.json exists, the fit once its report exists, rows once their manifest was
# written with this tag's model files (delete them to rebuild; FORCE=1 re-runs rows regardless).
#
#   tools/run_delta.sh                       # everything on tless + xyzibd
#   STEPS=tables tools/run_delta.sh          # only the labelled tables
#   STEPS=fit TAG=v2 tools/run_delta.sh      # only the fit
#   STEPS=rows DATASET=tless tools/run_delta.sh
#
# Row naming: A8_k<k> (product weights, ranked by Confidence) and A8w_k<k> (Model H as the fusion
# weight), one tools/run.py invocation per row with its own run_manifest.json (D12).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
datasets="${DATASET:-tless xyzibd}"
steps="${STEPS:-tables fit rows report}"
ks="${KS:-1 2 3 4}"
tag="${TAG:-v1}"
log_dir="outputs/runs"
mkdir -p "$log_dir"

run_row() {  # name, dataset, extra overrides...
  local name="$1" dataset="$2"; shift 2
  local manifest="$log_dir/${name}_${dataset}/run_manifest.json"
  # a row is skipped only if its manifest exists *and*, for a confidence row, was written with
  # this tag's model files: the row name does not carry the tag, and while the stage cache would
  # re-run confidence + evaluate for a refit, the manifest-based skip must not hide that
  if [ -f "$manifest" ] && [ "${FORCE:-0}" = 0 ]; then
    if [[ "$*" != *confidence.params* ]] || grep -q "models/confidence/$tag/model_f.json" "$manifest"; then
      echo "=== $name on $dataset (manifest exists, skipped)"; return
    fi
  fi
  echo "=== $name on $dataset ($*)"
  uv run python tools/run.py "experiment.name=$name" "dataset=$dataset" multiview=strided "$@" \
    2>&1 | tee "$log_dir/${name}_${dataset}.log" | grep -v "^\[" | tail -3
}

official() {  # name dataset: bop_toolkit on the row's CSV, restricted to the view-group images
  local name="$1" dataset="$2"
  local manifest="$log_dir/${name}_${dataset}/run_manifest.json"
  local ev_dir csv
  ev_dir="$(python3 -c "import json;print(json.load(open('$manifest'))['stages']['evaluate']['dir'])")"
  csv="$(ls "$ev_dir"/*.csv | head -1)"
  if [ ! -f "$ev_dir/bop_eval/$(basename "$csv" .csv)/scores_bop19.json" ]; then
    BOP_TARGETS="$ev_dir/targets_subset.json" tools/bop_eval.sh "$csv" \
      > "$log_dir/${name}_${dataset}.bop_eval.log" 2>&1 || echo "official eval failed: $name"
  fi
}

if [[ " $steps " == *" tables "* ]]; then
  for ds in $datasets; do
    # the single-view FusedPose row the Gamma curve did not need
    run_row A6_k1_mean "$ds" experiment=A6 fusion=mean multiview.params.groups.n_views=1
    if [ ! -f "outputs/confidence/$ds/build.json" ]; then
      uv run python tools/build_confidence_table.py --dataset "$ds" 2>&1 | tee "outputs/confidence_build_${ds}.log"
    fi
  done
fi

if [[ " $steps " == *" fit "* ]]; then
  # gate on the report, the last file the fit writes (the model files come first)
  if [ ! -f "outputs/confidence/$tag/report.md" ]; then
    # shellcheck disable=SC2086
    uv run python tools/fit_confidence.py --datasets $datasets --tag "$tag" ${FIT_ARGS:-} 2>&1 | tee "outputs/confidence_fit_${tag}.log"
  fi
fi

if [[ " $steps " == *" rows "* ]]; then
  for ds in $datasets; do
    for k in $ks; do
      run_row "A8_k${k}" "$ds" experiment=A8 "multiview.params.groups.n_views=$k" \
        "confidence.params.model_h=models/confidence/$tag/model_h.json" \
        "confidence.params.model_f=models/confidence/$tag/model_f.json" \
        "confidence.params.thresholds=models/confidence/$tag/thresholds.json"
      if [ "$k" != 1 ]; then
        run_row "A8w_k${k}" "$ds" experiment=A8 "multiview.params.groups.n_views=$k" \
          "confidence.params.model_h=models/confidence/$tag/model_h.json" \
          "confidence.params.model_f=models/confidence/$tag/model_f.json" \
          "confidence.params.thresholds=models/confidence/$tag/thresholds.json" \
          "+multiview.params.weights.source=model_h" \
          "+multiview.params.weights.model_h=models/confidence/$tag/model_h.json"
      fi
    done
    # official bop_toolkit scores one after another: bop_eval.sh checks out a shared clone and
    # concurrent runs race on its git index (only T-LESS has BOP19 targets)
    if [ "$ds" = tless ]; then
      for k in $ks; do
        official "A8_k${k}" "$ds"
        [ "$k" != 1 ] && official "A8w_k${k}" "$ds"
      done
    fi
  done
fi

if [[ " $steps " == *" report "* ]]; then
  for ds in $datasets; do
    uv run python tools/confidence_report.py --dataset "$ds" --tag "$tag" \
      --plot "outputs/${ds}_delta" > "outputs/${ds}_delta_report.md"
    echo "report: outputs/${ds}_delta_report.md"
  done
fi
