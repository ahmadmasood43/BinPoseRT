#!/usr/bin/env bash
# Move caches between the laptop and a GPU machine (D1, D12). Code travels through git; this script
# moves only the two directories git ignores:
#
#   tools/sync.sh push-data    user@gpu      # data/bop/<dataset>  laptop -> GPU   (skip the re-download)
#   tools/sync.sh push-outputs user@gpu      # outputs/            laptop -> GPU   (local stages, e.g. GT masks)
#   tools/sync.sh pull-outputs user@gpu      # outputs/            GPU -> laptop   (adapter caches + runs/)
#
# After `pull-outputs`, every `tools/run.py experiment=… dataset=…` on the laptop finds the GPU stages
# cached (hashes exclude paths) and only re-runs what is missing. The remote repository directory
# defaults to ~/BinPoseRT; override with BINPOSERT_REMOTE_DIR. A third argument narrows the dataset
# (default: tless).
set -euo pipefail

CMD=${1:?usage: tools/sync.sh push-data|push-outputs|pull-outputs user@host [dataset]}
HOST=${2:?usage: tools/sync.sh push-data|push-outputs|pull-outputs user@host [dataset]}
DATASET=${3:-tless}
REMOTE=${BINPOSERT_REMOTE_DIR:-BinPoseRT}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RSYNC=(rsync -a --partial --info=progress2 --exclude '__pycache__' --exclude '*.part' --exclude '*.zip')

case "$CMD" in
  push-data)
    ssh "$HOST" "mkdir -p $REMOTE/data/bop"
    "${RSYNC[@]}" "$HERE/data/bop/$DATASET/" "$HOST:$REMOTE/data/bop/$DATASET/"
    ;;
  push-outputs)
    ssh "$HOST" "mkdir -p $REMOTE/outputs"
    "${RSYNC[@]}" "$HERE/outputs/$DATASET/" "$HOST:$REMOTE/outputs/$DATASET/"
    ;;
  pull-outputs)
    mkdir -p "$HERE/outputs/$DATASET"
    "${RSYNC[@]}" "$HOST:$REMOTE/outputs/$DATASET/" "$HERE/outputs/$DATASET/"
    echo
    echo "pulled. Re-run the experiment here to evaluate from the cache, e.g."
    echo "  uv run python tools/run.py experiment=A0 dataset=$DATASET"
    ;;
  *)
    echo "unknown command $CMD" >&2; exit 1 ;;
esac
