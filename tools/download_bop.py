#!/usr/bin/env python
"""Download BOP dataset archives from the Hugging Face mirror and unpack them under data/bop/.

Examples::

    uv run python tools/download_bop.py tless --parts base models
    uv run python tools/download_bop.py tless --parts test_primesense_bop19   # 825 MB, GPU machine
    uv run python tools/download_bop.py xyzibd --parts base models val       # GPU machine only

Archive names follow the BOP convention ``<dataset>_<part>.zip``. Nothing is downloaded twice.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/bop-benchmark/{dataset}/resolve/main/{archive}"
DEFAULT_ROOT = Path("data/bop")


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")

    last = [-1]

    def hook(count: int, block: int, total: int) -> None:
        done = count * block
        pct = int(100.0 * done / total) if total > 0 else 0
        if pct // 5 != last[0]:  # one line per 5 %
            last[0] = pct // 5
            print(f"  {dst.name}: {done / 1e6:8.1f} MB ({pct:3d}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", help="BOP dataset name, e.g. tless, xyzibd, itodd, ipd")
    ap.add_argument("--parts", nargs="+", default=["base", "models"], help="archive suffixes")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    ds_dir = args.root / args.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        archive = f"{args.dataset}_{part}.zip"
        zip_path = args.root / archive
        marker = ds_dir / f".unpacked_{part}"
        if marker.exists():
            print(f"{archive}: already unpacked")
            continue
        if not zip_path.exists():
            url = HF_BASE.format(dataset=args.dataset, archive=archive)
            print(f"downloading {url}")
            download(url, zip_path)
        print(f"unpacking {archive} -> {ds_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            # BOP archives either contain '<dataset>/...' (base) or are relative to the dataset dir
            names = zf.namelist()
            prefixed = all(n.startswith(f"{args.dataset}/") for n in names if n)
            zf.extractall(args.root if prefixed else ds_dir)
        marker.touch()
        if not args.keep_zip:
            zip_path.unlink()
    print("done:", ds_dir)


if __name__ == "__main__":
    main()
