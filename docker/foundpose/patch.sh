#!/usr/bin/env bash
# Minimal source patches applied to the FoundPose checkout (idempotent).
#  1. faiss k-means on CPU: faiss-cpu has no GPU k-means, and the adapter keeps samples on CPU.
set -euo pipefail
src="${1:?upstream dir}"
python3 - "$src" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "utils" / "cluster_util.py"
s = p.read_text()
old = 'gpu = True if samples.device.type == "cuda" else False'
new = 'gpu = False  # binposert: faiss-cpu build (docker/foundpose/patch.sh)'
if old in s:
    p.write_text(s.replace(old, new))
    print("patched", p)
else:
    print("already patched", p)
PY
