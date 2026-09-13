#!/usr/bin/env bash
# Official BOP evaluation of a results CSV with bop_toolkit in its own environment (D15).
#
#   tools/bop_eval.sh outputs/tless/test_primesense/runs/A0/<stamp>/A0-gt-foundpose_tless-test.csv
#
# Env: BOP_PATH (parent of tless/, default data/bop), BOP_EVAL_OUT (default outputs/bop_eval),
#      BOP_TOOLKIT_REF (git ref, default master). The toolkit env is created once with uv under
#      .venv-bop_toolkit (numpy<2, opencv-python, python 3.11 — incompatible with the core env).
set -euo pipefail

CSV=${1:?usage: tools/bop_eval.sh <results_csv>}
BOP_PATH=${BOP_PATH:-$PWD/data/bop}
OUT=${BOP_EVAL_OUT:-$PWD/outputs/bop_eval}
REF=${BOP_TOOLKIT_REF:-master}
ENV_DIR=${BOP_TOOLKIT_ENV:-$PWD/.venv-bop_toolkit}
TOOLKIT_DIR=${BOP_TOOLKIT_DIR:-$PWD/.bop_toolkit}

if [ ! -d "$TOOLKIT_DIR" ]; then
  git clone https://github.com/thodan/bop_toolkit "$TOOLKIT_DIR"
fi
git -C "$TOOLKIT_DIR" checkout -q "$REF"

if [ ! -d "$ENV_DIR" ]; then
  uv venv --python 3.11 "$ENV_DIR"
  uv pip install --python "$ENV_DIR/bin/python" -e "$TOOLKIT_DIR" "numpy<2" "opencv-python" "vispy" "pypng" "scipy" "matplotlib" "pytz" "imageio" "trimesh" "kiwisolver"
fi

mkdir -p "$OUT"
NAME=$(basename "$CSV")
cp "$CSV" "$OUT/$NAME"
export BOP_PATH
# bop_toolkit reads results from `results_path` and writes to `eval_path`; both are passed explicitly.
"$ENV_DIR/bin/python" "$TOOLKIT_DIR/scripts/eval_bop19_pose.py" \
  --renderer_type=vispy \
  --results_path="$OUT" \
  --eval_path="$OUT" \
  --result_filenames="$NAME"
echo "official scores: $OUT/${NAME%.csv}/scores_bop19.json"
