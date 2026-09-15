#!/usr/bin/env python
"""Failure gallery for an evaluate stage: worst-N GT by MSSD, GT (green) vs. prediction (red).

uv run python tools/gallery.py outputs/tless/test_primesense/evaluate/<hash> --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.viz import make_failure_gallery  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evaluate_dir", type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    manifest = json.loads((args.evaluate_dir / "run_manifest.json").read_text())
    dataset = make_dataset(manifest["config"], REPO)
    out = make_failure_gallery(args.evaluate_dir, dataset, n=args.n, out_dir=args.out)
    print(f"gallery: {out / 'gallery.png'}")


if __name__ == "__main__":
    main()
