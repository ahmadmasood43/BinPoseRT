#!/usr/bin/env bash
# Official BOP evaluation of a BOP-format CSV with bop_toolkit in its own environment (D15).
# The core computes VSD/MSSD/MSPD itself for development; the numbers *reported* come from here.
#
#   tools/bop_eval.sh <result.csv> [<eval_dir>]
#
# <result.csv> must be named <method>_<dataset>-<split>.csv (e.g. binposert-A0_tless-test.csv), as
# written by the evaluate stage. Results land in <eval_dir> (default: next to the CSV, under
# bop_eval/). Needs data/bop/<dataset>/{models_eval,test_targets_bop19.json} and the test split.
set -euo pipefail
csv="${1:?usage: bop_eval.sh <result.csv> [<eval_dir>]}"
csv="$(readlink -f "$csv")"
eval_dir="${2:-$(dirname "$csv")/bop_eval}"
root="$(cd "$(dirname "$0")/.." && pwd)"
env_dir="$root/envs/bop_toolkit"
src_dir="$root/third_party/bop_toolkit"
BOP_TOOLKIT_COMMIT="${BOP_TOOLKIT_COMMIT:-cea62d651c7e395b2e1962b9749e4e89693c6ac4}"

if [ ! -d "$src_dir/.git" ]; then
  git clone -q https://github.com/thodan/bop_toolkit.git "$src_dir"
fi
git -C "$src_dir" checkout -q "$BOP_TOOLKIT_COMMIT"
if [ ! -x "$env_dir/.venv/bin/python" ]; then
  uv venv -q --python 3.10 "$env_dir/.venv"
  uv pip install -q --python "$env_dir/.venv/bin/python" -e "$src_dir" pyopengl-accelerate 2>/dev/null \
    || uv pip install -q --python "$env_dir/.venv/bin/python" -e "$src_dir"
fi

orig_name="$(basename "$csv" .csv)"
dataset_split="${orig_name##*_}"    # tless-test (after the last underscore)
dataset="${dataset_split%-*}"       # tless
# bop_toolkit splits the file name on underscores, so a method name may not contain any
# (experiment names such as A6_k2_mean do): underscores become hyphens for the evaluation.
method="${orig_name%_*}"
name="${method//_/-}_${dataset_split}"
mkdir -p "$eval_dir"
cp -f "$csv" "$eval_dir/$name.csv"

export BOP_PATH="$root/data/bop"
# BOP_TARGETS=<path> scores a subset of the targets (multi-view rows: the view-group images only);
# bop_toolkit resolves the name relative to the dataset directory, so the file is copied there.
targets_name="test_targets_bop19.json"
if [ -n "${BOP_TARGETS:-}" ]; then
  targets_name="targets_$(sha256sum "$BOP_TARGETS" | cut -c1-12).json"
  cp -f "$BOP_TARGETS" "$BOP_PATH/$dataset/$targets_name"
fi
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PATH="$env_dir/.venv/bin:$PATH"   # eval_bop19_pose.py shells out to a bare `python`
cd "$src_dir"
python scripts/eval_bop19_pose.py \
  --renderer_type vispy \
  --results_path "$eval_dir" \
  --eval_path "$eval_dir" \
  --result_filenames "$name.csv" \
  --targets_filename "$targets_name" \
  --num_workers "${BOP_EVAL_WORKERS:-1}"   # >1 races on a shared tmp dir in renderer_batch.py
if [ "$name" != "$orig_name" ] && [ ! -e "$eval_dir/$orig_name" ]; then
  ln -s "$name" "$eval_dir/$orig_name"   # found under the CSV's own name too
fi
echo "official scores: $eval_dir/$name/scores_bop19.json"
cat "$eval_dir/$name/scores_bop19.json"
