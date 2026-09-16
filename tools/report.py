#!/usr/bin/env python
"""Aggregate experiment rows into one Markdown table from ``outputs/`` (D12: numbers only ever
come from run directories).

    uv run python tools/report.py --dataset tless A0 A1 A5 > outputs/tless/alpha_report.md

For each experiment the latest ``outputs/runs/<exp>_<dataset>/run_manifest.json`` locates the
evaluate stage; ``report.json`` gives the core AR and the visibility strata, and, when
``tools/bop_eval.sh`` was run on the BOP CSV, ``bop_eval/<name>/scores_bop19.json`` gives the
official numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def load_row(exp: str, dataset: str, outputs: Path) -> dict[str, Any] | None:
    manifest_path = outputs / "runs" / f"{exp}_{dataset}" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    ev = Path(manifest["stages"]["evaluate"]["dir"])
    report = json.loads((ev / "report.json").read_text())
    official: dict[str, Any] | None = None
    for p in ev.glob("bop_eval/*/scores_bop19.json"):
        official = json.loads(p.read_text())
    cfg = manifest["config"]
    return {
        "experiment": exp,
        "segmenter": cfg["segmenter"]["name"],
        "estimator": cfg["estimator"]["name"],
        "evaluate_dir": str(ev.relative_to(REPO)) if ev.is_relative_to(REPO) else str(ev),
        "evaluate_dir_abs": str(ev),
        "git_commit": manifest.get("git_commit"),
        "report": report,
        "official": official,
        "n_predictions": report["n_predictions"],
    }


def fmt(x: Any) -> str:
    return "—" if x is None else f"{100 * float(x):.1f}"


def render(rows: list[dict[str, Any]], dataset: str) -> str:
    out = [
        f"# Alpha results — {dataset}",
        "",
        "AR in %, BOP19 protocol. *core* = `binposert.evaluate` (development metric); *official* = "
        "`bop_toolkit` via `tools/bop_eval.sh` (reported number).",
        "",
        "| exp | seg | coarse pose | AR core | AR official | VSD | MSSD | MSPD | n pred | run |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep, off = r["report"], r["official"]
        ar_off = fmt(off["bop19_average_recall"]) if off else "—"
        vsd = fmt(off["bop19_average_recall_vsd"]) if off else fmt(rep["ar_vsd"])
        mssd = fmt(off["bop19_average_recall_mssd"]) if off else fmt(rep["ar_mssd"])
        mspd = fmt(off["bop19_average_recall_mspd"]) if off else fmt(rep["ar_mspd"])
        out.append(
            f"| {r['experiment']} | {r['segmenter']} | {r['estimator']} | {fmt(rep['ar'])} "
            f"| {ar_off} | {vsd} | {mssd} | {mspd} | {r['n_predictions']} "
            f"| `{r['evaluate_dir']}` |"
        )
    out += ["", "## AR by visible fraction (core metric, pooled over GT)", ""]
    bins = rows[0]["report"]["by_visibility"] if rows else []
    header = "| exp | " + " | ".join(f"[{b['lo']:.1f}, {b['hi']:.1f}) n" for b in bins) + " |"
    out += [header, "|---|" + "---|" * len(bins)]
    for r in rows:
        cells = [f"{fmt(b['ar'])} ({b['n_gt']})" for b in r["report"]["by_visibility"]]
        out.append(f"| {r['experiment']} | " + " | ".join(cells) + " |")
    out += ["", "Commits: " + ", ".join(f"{r['experiment']}={r['git_commit']}" for r in rows), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    args = ap.parse_args()
    rows = []
    for exp in args.experiments:
        row = load_row(exp, args.dataset, args.outputs)
        if row is None:
            print(f"warning: no run manifest for {exp} on {args.dataset}", file=sys.stderr)
            continue
        rows.append(row)
    print(render(rows, args.dataset))


if __name__ == "__main__":
    main()
