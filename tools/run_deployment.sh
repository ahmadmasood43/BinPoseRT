#!/usr/bin/env bash
# Milestone Deployment (RQ-F): ICP schedule sweep, latency benchmark, and accuracy–latency Pareto.
# Assumes Delta/Epsilon caches are present on the GPU machine (segment + coarse_pose stages).
# All sweep rows share the A8 chain at k=4; y-axis is directly comparable to A8_k4 (official 75.9).
#
#   tools/run_deployment.sh                        # everything (slow: ~6 h unattended)
#   STEPS=equiv tools/run_deployment.sh            # Phase 3 equivalence check only
#   STEPS=sweep tools/run_deployment.sh            # ICP coordinate sweep (AR)
#   STEPS=bench tools/run_deployment.sh            # latency benchmark for every sweep row
#   STEPS=official tools/run_deployment.sh         # bop_toolkit for 4 official rows only
#   STEPS=xyzibd tools/run_deployment.sh           # XYZ-IBD cross-check row
#   STEPS=full_pipeline tools/run_deployment.sh    # D13's second budget (~2 min, once)
#   STEPS="sweep bench report" tools/run_deployment.sh
#
# FORCE=1 re-runs a row even if its manifest already exists.
# Ordering constraint: bench refuses to start while sweep rows are in flight (enforced below).
# Do NOT run bench and sweep concurrently: benchmark enforces load_avg < 2.0 and would starve.
#
# Official BOP eval costs ~3173 s/row (measured); it runs serially for 4 rows ≈ 3.5 h.
# Only T-LESS has BOP19 targets; XYZ-IBD cross-check reports core AR only.
# full_pipeline is D13's second latency budget (P3/P18/P19): one real tools/run.py invocation,
# segment through evaluate, on a 2-scene x 12-image subset -- separate from the ICP sweep entirely.

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"

steps="${STEPS:-equiv sweep bench official xyzibd full_pipeline report}"
log_dir="outputs/runs"
mkdir -p "$log_dir"

# 4 rows get official BOP eval (serially, overnight):
#   A10_exact        — reference (must reproduce A8_k4 per-GT errors)
#   A10_l3_i30_p3000 — ROI-only, default schedule
#   operating_point  — the chosen operating point (declare in the sweep step below)
#   cheapest_frontier — the cheapest Pareto-frontier point
OFFICIAL_ROWS="${OFFICIAL_ROWS:-A10_exact A10_l3_i30_p3000}"

run_row() {
  local name="$1" dataset="$2"; shift 2
  local manifest="$log_dir/${name}_${dataset}/run_manifest.json"
  if [ -f "$manifest" ] && [ "${FORCE:-0}" = 0 ]; then
    echo "=== $name on $dataset (manifest exists, skipped)"; return
  fi
  echo "=== $name on $dataset ($*)"
  uv run python tools/run.py "experiment.name=$name" "dataset=$dataset" \
    experiment=A10 multiview.params.groups.n_views=4 "$@" \
    2>&1 | tee "$log_dir/${name}_${dataset}.log" | grep -v "^\[" | tail -3
}

official() {
  local name="$1" dataset="$2"
  local manifest="$log_dir/${name}_${dataset}/run_manifest.json"
  [ "$dataset" != tless ] && return   # only tless has BOP19 targets
  local ev_dir csv
  ev_dir="$(python3 -c "import json;print(json.load(open('$manifest'))['stages']['evaluate']['dir'])")"
  csv="$(ls "$ev_dir"/*.csv 2>/dev/null | head -1)"
  [ -z "$csv" ] && { echo "no CSV for $name/$dataset"; return 1; }
  if [ ! -f "$ev_dir/bop_eval/$(basename "$csv" .csv)/scores_bop19.json" ]; then
    tools/bop_eval.sh "$csv" \
      > "$log_dir/${name}_${dataset}.bop_eval.log" 2>&1 || echo "official eval failed: $name"
  fi
}

# ---------------------------------------------------------------------------
# Phase 3: equivalence proof (requires A8 cached refine details on this machine)
# ---------------------------------------------------------------------------
if [[ " $steps " == *" equiv "* ]]; then
  echo "=== Phase 3: equivalence + hash guard ==="
  # OMP_NUM_THREADS=1: forces single-threaded Open3D ICP so the roi=none exact-equality leg
  # is deterministic across runs.  The original A8_k4 cache was also generated single-threaded
  # (the refine stage worker pool uses one process per object, not intra-process OMP threads).
  OMP_NUM_THREADS=1 uv run python tools/check_deployment.py --equivalence --hash-guard \
    --dataset tless --row A8_k4 \
    2>&1 | tee "$log_dir/check_deployment.log" || true
  # Phase 3 is INFORMATIONAL: failures here (stop-rule, missing caches) don't block the sweep.
  # The critical invariant (A8_k4 refine hash unchanged) is checked inside check_deployment.py.
  grep -E "^(PASS|FAIL)" "$log_dir/check_deployment.log" | tail -8
  # Abort only if the A8_k4 refine hash specifically changed — that would mean cache invalidation.
  if grep -q "FAIL.*A8_k4 refine hash" "$log_dir/check_deployment.log" 2>/dev/null; then
    echo "FATAL: A8_k4 refine hash changed — aborting; do not invalidate 13 h of GPU work" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Phase 4: ICP coordinate sweep (all rows serially; no GPU needed from here)
# ---------------------------------------------------------------------------
if [[ " $steps " == *" sweep "* ]]; then
  echo "=== Phase 4: ICP coordinate sweep ==="

  # Reference row: no roi override → reuses A8_k4 refine/evaluate caches (same YAML → same hash).
  run_row A10_exact tless

  # ROI-only row: roi=bbox, default schedule.  Measures the speedup cost in AR.
  run_row A10_l3_i30_p3000 tless \
    "+refiner.params.roi=bbox"

  # L axis: fewer correspondence levels
  run_row A10_l2_i30_p3000 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.15,0.04]"

  run_row A10_l1_i30_p3000 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.10]"

  # I axis: fewer ICP iterations
  run_row A10_l3_i15_p3000 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.max_iterations=15"

  run_row A10_l3_i8_p3000 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.max_iterations=8"

  # P axis: fewer model/scene points
  run_row A10_l3_i30_p1500 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.max_model_points=1500" \
    "refiner.params.max_scene_points=1500"

  run_row A10_l3_i30_p750 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.max_model_points=750" \
    "refiner.params.max_scene_points=750"

  # Combinations on the promising direction
  run_row A10_l2_i15_p1500 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.15,0.04]" \
    "refiner.params.max_iterations=15" \
    "refiner.params.max_model_points=1500" \
    "refiner.params.max_scene_points=1500"

  run_row A10_l1_i15_p1500 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.10]" \
    "refiner.params.max_iterations=15" \
    "refiner.params.max_model_points=1500" \
    "refiner.params.max_scene_points=1500"

  run_row A10_l1_i8_p750 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.10]" \
    "refiner.params.max_iterations=8" \
    "refiner.params.max_model_points=750" \
    "refiner.params.max_scene_points=750"

  # Control row: same config as A10_l1_i15_p750 but score_signal=pose_score
  # Separates "geometry worse" from "confidence ranking worse" on cheapest schedule.
  run_row A10_l1_i15_p750 tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.10]" \
    "refiner.params.max_iterations=15" \
    "refiner.params.max_model_points=750" \
    "refiner.params.max_scene_points=750"

  run_row A10_l1_i15_p750_posescore tless \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.10]" \
    "refiner.params.max_iterations=15" \
    "refiner.params.max_model_points=750" \
    "refiner.params.max_scene_points=750" \
    "evaluate.score_signal=pose_score"
fi

# ---------------------------------------------------------------------------
# Phase 4: latency benchmark — MUST run after sweep (load constraint)
# ---------------------------------------------------------------------------
if [[ " $steps " == *" bench "* ]]; then
  echo "=== Phase 4: latency benchmark (serially, load < 2.0) ==="
  # Abort if any sweep row is still running (pgrep by script arg, not by name, to avoid self-match)
  if pgrep -f "run.py.*experiment=A10" > /dev/null; then
    echo "ERROR: sweep rows still running; bench must not run concurrently" >&2
    exit 1
  fi
  # 1-minute load average lags the previous benchmark's own CPU usage; wait it out
  # rather than let benchmark.py's guard abort the whole loop on a stale reading.
  wait_for_quiet_load() {
    local waited=0
    while true; do
      load1="$(awk '{print $1}' /proc/loadavg)"
      if awk -v l="$load1" 'BEGIN{exit !(l < 2.0)}'; then return 0; fi
      if [ "$waited" -ge 300 ]; then
        echo "WARN: load still $load1 after ${waited}s; proceeding anyway" >&2
        return 0
      fi
      sleep 15; waited=$((waited + 15))
    done
  }
  for row in A10_exact A10_l3_i30_p3000 \
             A10_l2_i30_p3000 A10_l1_i30_p3000 \
             A10_l3_i15_p3000 A10_l3_i8_p3000 \
             A10_l3_i30_p1500 A10_l3_i30_p750 \
             A10_l2_i15_p1500 A10_l1_i15_p1500 \
             A10_l1_i8_p750 A10_l1_i15_p750 \
             A10_l1_i15_p750_posescore; do
    bench_out="outputs/tless_benchmark_${row}.json"
    if [ -f "$bench_out" ] && [ "${FORCE:-0}" = 0 ]; then
      echo "=== bench $row (exists, skipped)"; continue
    fi
    wait_for_quiet_load
    echo "=== bench $row"
    uv run python tools/benchmark.py --dataset tless --row "$row" \
      2>&1 | tee "$log_dir/bench_${row}_tless.log" | tail -3
  done
fi

# ---------------------------------------------------------------------------
# Phase 4: official BOP eval for 4 rows (serially, ~3.5 h unattended)
# ---------------------------------------------------------------------------
if [[ " $steps " == *" official "* ]]; then
  echo "=== Phase 4: official BOP eval (serialised) ==="
  for row in $OFFICIAL_ROWS; do
    official "$row" tless
  done
fi

# ---------------------------------------------------------------------------
# XYZ-IBD cross-check (one row, core AR only — no official eval for xyzibd)
# ---------------------------------------------------------------------------
if [[ " $steps " == *" xyzibd "* ]]; then
  echo "=== XYZ-IBD cross-check ==="
  operating_point="${OPERATING_POINT:-A10_l2_i15_p1500}"
  run_row "$operating_point" xyzibd \
    "+refiner.params.roi=bbox" \
    "refiner.params.corr_dist_factors=[0.15,0.04]" \
    "refiner.params.max_iterations=15" \
    "refiner.params.max_model_points=1500" \
    "refiner.params.max_scene_points=1500"
fi

# ---------------------------------------------------------------------------
# Full-pipeline budget (D13's second target, P18/P19): one real run, 2 scenes x 12 images.
# ---------------------------------------------------------------------------
if [[ " $steps " == *" full_pipeline "* ]]; then
  echo "=== Full-pipeline budget (D13's second target) ==="
  uv run python tools/benchmark.py --dataset tless --full-pipeline \
    2>&1 | tee "$log_dir/full_pipeline_bench.log" | tail -5
fi

# ---------------------------------------------------------------------------
# Phase 5: report
# ---------------------------------------------------------------------------
if [[ " $steps " == *" report "* ]]; then
  echo "=== Phase 5: deployment report ==="
  uv run python tools/deployment_report.py --dataset tless \
    2>&1 | tee "$log_dir/deployment_report.log" | tail -5
  echo "report: outputs/tless_deployment_report.md"
  echo "pareto: docs/figures/deployment_tless_pareto.png"
fi
