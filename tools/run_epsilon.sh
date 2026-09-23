#!/usr/bin/env bash
# Milestone Epsilon (next-best-view, D14): the A9 rows from the cached refine stage. Every row is
# one tools/run.py invocation (D12) of the A9 chain — refine cache -> nbv stage (the active loop:
# associate -> fuse -> confidence after every unlocked View) -> evaluate on the episode's
# reference Views. Resumable: a row is skipped once its run manifest exists (FORCE=1 re-runs).
#
#   tools/run_epsilon.sh                         # every row on xyzibd, start groups 0..3
#   START_GROUPS="0" tools/run_epsilon.sh        # one start group
#   DATASET=tless BUDGETS="2 4" tools/run_epsilon.sh
#   STEPS=verdict tools/run_epsilon.sh           # only the Verdict-stopped loop rows
#   STEPS=ablation tools/run_epsilon.sh          # NBV with the entropy track weight (A9_nbve_*)
#   STEPS=oracle tools/run_epsilon.sh            # GT-scored next View, the ceiling (A9_oracle_*)
#
# Row naming: A9_<policy>_b<budget>_g<start group> for the AR-vs-views curve (fixed b1..b4 and
# b0 = every View; nbv / random b2..b4) and A9_<policy>_verdict_g<g> for the D14 loop proper
# (stop when no ObjectTrack requests a View, budget = the whole pool).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
dataset="${DATASET:-xyzibd}"
groups="${START_GROUPS:-0 1 2 3}"   # not GROUPS: bash owns that name
budgets="${BUDGETS:-2 3 4}"
steps="${STEPS:-curve ablation oracle verdict report}"
log_dir="outputs/runs"
mkdir -p "$log_dir"

run_row() {  # name, extra overrides...
  local name="$1"; shift
  local manifest="$log_dir/${name}_${dataset}/run_manifest.json"
  if [ -f "$manifest" ] && [ "${FORCE:-0}" = 0 ]; then
    echo "=== $name on $dataset (manifest exists, skipped)"; return
  fi
  echo "=== $name on $dataset ($*)"
  uv run python tools/run.py experiment=A9 "experiment.name=$name" "dataset=$dataset" "$@" \
    2>&1 | tee "$log_dir/${name}_${dataset}.log" | grep -v "^\[" | tail -3
}

if [[ " $steps " == *" curve "* ]]; then
  for g in $groups; do
    run_row "A9_fixed_b1_g$g" nbv.params.policy=fixed nbv.params.budget=1 "nbv.params.start_group=$g"
    for b in $budgets; do
      run_row "A9_nbv_b${b}_g$g" nbv.params.policy=nbv "nbv.params.budget=$b" "nbv.params.start_group=$g"
      run_row "A9_random_b${b}_g$g" nbv.params.policy=random "nbv.params.budget=$b" "nbv.params.start_group=$g"
      run_row "A9_fixed_b${b}_g$g" nbv.params.policy=fixed "nbv.params.budget=$b" "nbv.params.start_group=$g"
    done
    run_row "A9_fixed_b0_g$g" nbv.params.policy=fixed nbv.params.budget=0 "nbv.params.start_group=$g"
  done
fi

if [[ " $steps " == *" ablation "* ]]; then
  # E10: the track weight of the score — the binary entropy of the Confidence instead of 1 - p
  for g in $groups; do
    for b in $budgets; do
      run_row "A9_nbve_b${b}_g$g" nbv.params.policy=nbv "nbv.params.budget=$b" "nbv.params.start_group=$g" \
        nbv.params.score.weight=entropy
    done
  done
fi

if [[ " $steps " == *" oracle "* ]]; then
  # the ceiling of view choice: the GT-scored next View (E11), never a robot policy
  for g in $groups; do
    for b in $budgets; do
      run_row "A9_oracle_b${b}_g$g" nbv.params.policy=oracle "nbv.params.budget=$b" "nbv.params.start_group=$g"
    done
  done
fi

if [[ " $steps " == *" verdict "* ]]; then
  for g in $groups; do
    for policy in nbv random; do
      run_row "A9_${policy}_verdict_g$g" "nbv.params.policy=$policy" nbv.params.budget=0 \
        nbv.params.stop=verdict "nbv.params.start_group=$g"
    done
  done
fi

if [[ " $steps " == *" report "* ]]; then
  uv run python tools/nbv_report.py --dataset "$dataset" 2>&1 | tee "outputs/${dataset}_epsilon_report.log" | tail -20
fi
