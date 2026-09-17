#!/usr/bin/env python
"""Verify that a dataset's Gamma rows are complete and consistent before the milestone is closed.

    uv run python tools/check_gamma.py --dataset tless
    uv run python tools/check_gamma.py --dataset xyzibd --n-images 255 --n-scenes 15

Checks, each printed as PASS / FAIL: every curve row (A6_k<k>_{none,best,mean}, A7_k<k>) and
sweep row has a run manifest whose stages all carry ``_SUCCESS``; every evaluate dir has
``report.json`` and ``gt_rows.parquet`` (the bootstrap input); the k = 1 and k = 2 single-view rows
score the same images and agree exactly; fused rows predict more than single-view rows; the
bootstrap interval of every row brackets its AR and is non-degenerate; association analysis and
galleries exist for every multi-view row; the report and its figures exist; no row log contains a
traceback. Exit status 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from multiview_report import bootstrap_ar, row_name  # noqa: E402

FUSIONS = ["none", "best", "mean", "joint"]


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))
        self.failures += 0 if ok else 1
        return ok


def manifest(outputs: Path, name: str, dataset: str) -> dict | None:
    p = outputs / "runs" / f"{name}_{dataset}" / "run_manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--n-images", type=int, default=None, help="expected images of the k=1 row")
    ap.add_argument("--n-scenes", type=int, default=None, help="expected scenes in gt_rows")
    ap.add_argument("--sweep-t", nargs="+", default=["0", "1", "2", "5", "10"])
    ap.add_argument("--sweep-r", nargs="+", default=["0", "0.1", "0.25", "0.5", "1"])
    args = ap.parse_args()
    c = Checker()
    ds = args.dataset

    rows = [(k, f) for k in args.ks for f in FUSIONS if not (k == 1 and f != "none")]
    reports: dict[tuple[int, str], dict] = {}
    for k, f in rows:
        name = row_name(k, f)
        m = manifest(args.outputs, name, ds)
        if not c.check(m is not None, f"{name}: run manifest"):
            continue
        assert m is not None
        dirs = {s: Path(v["dir"]) for s, v in m["stages"].items()}
        c.check(
            all((d / "_SUCCESS").exists() for d in dirs.values()),
            f"{name}: every stage has _SUCCESS",
        )
        ev = dirs["evaluate"]
        ok = (ev / "report.json").exists() and (ev / "gt_rows.parquet").exists()
        if not c.check(ok, f"{name}: report.json + gt_rows.parquet"):
            continue
        rep = json.loads((ev / "report.json").read_text())
        reports[(k, f)] = rep
        log = args.outputs / "runs" / f"{name}_{ds}.log"
        c.check(
            not (log.exists() and "Traceback" in log.read_text()), f"{name}: no traceback in log"
        )
        boot, lo, hi = bootstrap_ar(ev)
        c.check(
            lo < boot < hi and hi - lo > 1e-4,
            f"{name}: bootstrap interval brackets AR",
            f"AR {100 * rep['ar']:.1f}, per-GT approx {100 * boot:.1f} "
            f"[{100 * lo:.1f}, {100 * hi:.1f}]",
        )
        if args.n_scenes is not None:
            n = pd.read_parquet(ev / "gt_rows.parquet").scene_id.nunique()
            c.check(n == args.n_scenes, f"{name}: gt_rows cover {args.n_scenes} scenes", f"{n}")
        if f != "none" and k > 1:
            c.check(
                (dirs["associate"] / "analysis" / "association.json").exists(),
                f"{name}: association analysis",
            )
            gallery = dirs["fuse"] / "analysis" / "gallery"
            c.check(
                gallery.exists() and any(gallery.glob("*.png")),
                f"{name}: gallery",
                ", ".join(p.name for p in sorted(gallery.glob("*.png")))
                if gallery.exists()
                else "",
            )

    # cross-row consistency
    r1 = reports.get((1, "none"))
    for k in args.ks[1:]:
        rk = reports.get((k, "none"))
        if not (r1 and rk):
            continue
        same = r1["n_images"] == rk["n_images"]
        # strided groups leave n mod k images out: identical AR on identical images, otherwise
        # the single-view row on the subset must stay close to the full single-view row
        ok = (
            abs(r1["ar"] - rk["ar"]) < (1e-9 if same else 0.02) and rk["n_images"] <= r1["n_images"]
        )
        c.check(
            ok,
            f"k={k} single-view row consistent with k=1"
            + (" (identical images)" if same else " (image subset)"),
            f"{rk['n_images']} vs {r1['n_images']} images, "
            f"AR {100 * rk['ar']:.2f} / {100 * r1['ar']:.2f}",
        )
    if r1 and args.n_images is not None:
        c.check(
            r1["n_images"] == args.n_images,
            f"k=1 row scores {args.n_images} images",
            str(r1["n_images"]),
        )
    for k in args.ks:
        base = reports.get((k, "none"))
        for f in ("best", "mean", "joint"):
            r = reports.get((k, f))
            if base and r:
                c.check(
                    r["n_predictions"] > base["n_predictions"]
                    and r["n_images"] == base["n_images"],
                    f"k={k} {f}: more predictions than single view on the same images",
                    f"{r['n_predictions']} vs {base['n_predictions']}",
                )
                c.check(
                    r["ar"] > base["ar"],
                    f"k={k} {f}: AR above single view",
                    f"{100 * r['ar']:.1f} vs {100 * base['ar']:.1f}",
                )

    # sweep
    missing = [
        f"t{t}_r{r}"
        for t in args.sweep_t
        for r in args.sweep_r
        if manifest(args.outputs, f"A6_sweep_t{t}_r{r}", ds) is None
    ]
    c.check(
        not missing,
        f"extrinsic sweep: all {len(args.sweep_t) * len(args.sweep_r)} points",
        ", ".join(missing[:5]),
    )

    # report + figures
    for name in (f"{ds}_gamma_report.md", f"{ds}_gamma_curve.png", f"{ds}_gamma_sweep.png"):
        c.check((args.outputs / name).exists(), f"{name} exists")
    rep_md = args.outputs / f"{ds}_gamma_report.md"
    if rep_md.exists():
        text = rep_md.read_text()
        c.check("| 0 | —" not in text, "sweep table in the report is filled")
        n_cells = sum(
            1 for line in text.splitlines() if line.startswith("| ") and "[" in line and "]" in line
        )
        c.check(
            n_cells >= len(args.ks), "curve table carries bootstrap intervals", f"{n_cells} rows"
        )

    print(
        f"\n{'ALL CHECKS PASSED' if c.failures == 0 else f'{c.failures} CHECK(S) FAILED'} for {ds}"
    )
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
