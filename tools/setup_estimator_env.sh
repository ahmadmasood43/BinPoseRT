#!/usr/bin/env bash
# Create the host-side environment for one external estimator without Docker (D15 fallback for a GPU
# machine where the NVIDIA container toolkit is unavailable). Mirrors docker/<name>/Dockerfile:
#   envs/<name>/.venv        uv-managed venv, python 3.10, docker/<name>/requirements.txt
#   third_party/<name>       upstream checkout at docker/<name>/upstream.env's commit, patched
#
#   tools/setup_estimator_env.sh cnos
set -euo pipefail
name="${1:?usage: setup_estimator_env.sh <cnos|foundpose|megapose>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
spec="$root/docker/$name"
env_dir="$root/envs/$name"
src_dir="$root/third_party/$name"

# shellcheck disable=SC1091
source "$spec/upstream.env"

if [ ! -d "$src_dir/.git" ]; then
  git clone --recurse-submodules "$UPSTREAM_URL" "$src_dir"
fi
git -C "$src_dir" fetch -q origin
git -C "$src_dir" checkout -q "$UPSTREAM_COMMIT"
git -C "$src_dir" submodule update --init --recursive -q
if [ -x "$spec/patch.sh" ]; then
  "$spec/patch.sh" "$src_dir"
fi

if [ ! -d "$env_dir/.venv" ]; then
  uv venv --python 3.10 "$env_dir/.venv"
fi
uv pip install --python "$env_dir/.venv/bin/python" --index-strategy unsafe-best-match \
  -r "$spec/requirements.txt"
echo "ready: $env_dir/.venv (upstream $src_dir @ $UPSTREAM_COMMIT)"
