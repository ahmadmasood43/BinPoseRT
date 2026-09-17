#!/usr/bin/env bash
# Resume the XYZ-IBD download (curl -C - continues a partial file) and unpack with download_bop.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
src=https://huggingface.co/datasets/bop-benchmark/xyzibd/resolve/main
for part in val test_all; do
  zip="data/bop/xyzibd_${part}.zip"
  if [ ! -f "data/bop/xyzibd/.unpacked_${part}" ] && [ ! -f "$zip" ]; then
    curl -L -C - --retry 20 --retry-delay 10 -o "${zip}.part" "$src/xyzibd_${part}.zip"
    mv "${zip}.part" "$zip"
  fi
done
uv run python tools/download_bop.py xyzibd --parts base models val test_all
