#!/usr/bin/env bash
# Gamma on XYZ-IBD val (the test split has no public GT): the GPU stages once (CNOS, FoundPose with
# onboarding) through the k=1 row, then the same curve/sweep rows as T-LESS from the caches.
#   tools/run_gamma_xyzibd.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p outputs/runs
echo "=== GPU stages (A6_k1_none on xyzibd, run_external=true)"
uv run python tools/run.py experiment.name=A6_k1_none dataset=xyzibd multiview=strided experiment=A6 \
  fusion=none multiview.params.groups.n_views=1 run_external=true \
  2>&1 | tee outputs/runs/A6_k1_none_xyzibd.log | grep -v "^\[" | tail -3
echo "=== curve + sweep"
DATASET=xyzibd MULTIVIEW=strided KS="1 2 3 4" tools/run_gamma.sh
