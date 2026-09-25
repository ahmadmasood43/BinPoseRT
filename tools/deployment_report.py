#!/usr/bin/env python
"""Deployment report: accuracy–latency Pareto from the A10 ICP schedule sweep.

    uv run python tools/deployment_report.py --dataset tless
    uv run python tools/deployment_report.py --dataset tless --plot outputs/tless_deployment

Reads:
  outputs/runs/<row>_<dataset>/run_manifest.json   — per-row stage dirs and config
  outputs/<dataset>_benchmark_<row>.json           — per-row latency (schema_version 2)
  <evaluate_dir>/gt_rows.parquet                   — for bootstrap AR
  <evaluate_dir>/bop_eval/.../scores_bop19.json    — official AR (when present)

Writes:
  outputs/tless_deployment_report.md           — human-readable results document
  outputs/tless_deployment_pareto.json         — one record per row (provenance anchor)
  outputs/tless_deployment_pareto.png          — Pareto figure
  docs/figures/deployment_tless_pareto.png     — curated copy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from multiview_report import bootstrap_ar  # noqa: E402
from report import load_row  # noqa: E402

# Row order matches the sweep (reference first, then by axis, then combinations)
SWEEP_ROWS = [
    ("A10_exact", "roi=none, L=3/I=30/P=3000 (reference)", False),
    ("A10_l3_i30_p3000", "roi=bbox, L=3/I=30/P=3000", False),
    ("A10_l2_i30_p3000", "roi=bbox, L=2/I=30/P=3000", False),
    ("A10_l1_i30_p3000", "roi=bbox, L=1/I=30/P=3000", False),
    ("A10_l3_i15_p3000", "roi=bbox, L=3/I=15/P=3000", False),
    ("A10_l3_i8_p3000", "roi=bbox, L=3/I=8/P=3000", False),
    ("A10_l3_i30_p1500", "roi=bbox, L=3/I=30/P=1500", False),
    ("A10_l3_i30_p750", "roi=bbox, L=3/I=30/P=750", False),
    ("A10_l2_i15_p1500", "roi=bbox, L=2/I=15/P=1500", False),
    ("A10_l1_i15_p1500", "roi=bbox, L=1/I=15/P=1500", False),
    ("A10_l1_i8_p750", "roi=bbox, L=1/I=8/P=750", False),
    ("A10_l1_i15_p750", "roi=bbox, L=1/I=15/P=750", False),
    (
        "A10_l1_i15_p750_posescore",
        "roi=bbox, L=1/I=15/P=750, score_signal=pose_score (control)",
        False,
    ),
]

XYZIBD_ROW = ("A10_l2_i15_p1500", "operating point cross-check, xyzibd")

TARGET_P95_MS = 200.0


def load_official_ar(evaluate_dir: Path) -> float | None:
    """Read the bop_toolkit top-level AR from scores_bop19.json, or None."""
    bop = evaluate_dir / "bop_eval"
    if not bop.exists():
        return None
    for scores in sorted(bop.rglob("scores_bop19.json")):
        try:
            data = json.loads(scores.read_text())
            return float(data["bop19_average_recall"])
        except Exception:
            continue
    return None


def load_bench(row: str, dataset: str, outputs: Path) -> dict[str, Any] | None:
    p = outputs / f"{dataset}_benchmark_{row}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def collect_rows(dataset: str, outputs: Path) -> list[dict[str, Any]]:
    records = []
    for name, label, _is_xyzibd in SWEEP_ROWS:
        base = load_row(name, dataset, outputs)
        if base is None:
            continue
        ev_dir = Path(base["evaluate_dir_abs"]) if base.get("evaluate_dir_abs") else None
        core_ar = official_ar = ar_lo = ar_hi = float("nan")
        if ev_dir and ev_dir.exists():
            try:
                core_ar, ar_lo, ar_hi = bootstrap_ar(ev_dir)
            except Exception:
                pass
            off = load_official_ar(ev_dir)
            if off is not None:
                official_ar = off

        bench = load_bench(name, dataset, outputs)
        p95 = float("nan")
        p50 = float("nan")
        if bench:
            tp = bench.get("per_track_total_ms", {})
            p95 = float(tp.get("p95_ms", float("nan")))
            p50 = float(tp.get("median_ms", float("nan")))

        records.append(
            {
                "row": name,
                "label": label,
                "core_ar": core_ar,
                "ar_lo": ar_lo,
                "ar_hi": ar_hi,
                "official_ar": official_ar,
                "p50_ms": p50,
                "p95_ms": p95,
            }
        )
    return records


def collect_xyzibd(outputs: Path) -> dict[str, Any] | None:
    """Core-AR-only cross-check row for xyzibd (no BOP19 GT for its test split, D4)."""
    name, label = XYZIBD_ROW
    base = load_row(name, "xyzibd", outputs)
    if base is None:
        return None
    ev_dir = Path(base["evaluate_dir_abs"])
    try:
        core_ar, ar_lo, ar_hi = bootstrap_ar(ev_dir)
    except Exception:
        core_ar = ar_lo = ar_hi = float("nan")
    return {"row": name, "label": label, "core_ar": core_ar, "ar_lo": ar_lo, "ar_hi": ar_hi}


def pareto_frontier(records: list[dict[str, Any]]) -> list[str]:
    """Names of rows on the AR–latency Pareto frontier (minimise p95, maximise AR)."""
    valid = [
        (r["row"], r["p95_ms"], r["core_ar"])
        for r in records
        if not np.isnan(r["p95_ms"]) and not np.isnan(r["core_ar"])
    ]
    frontier = []
    best_ar = -1.0
    for name, _p95, ar in sorted(valid, key=lambda x: x[1]):  # sort by p95 ascending
        if ar > best_ar:
            best_ar = ar
            frontier.append(name)
    return frontier


def plot_pareto(records: list[dict[str, Any]], frontier_names: list[str], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    frontier_set = set(frontier_names)

    def plotted_ar(r: dict[str, Any]) -> float:
        """The y-value actually drawn for this row: official AR when available, else core AR."""
        return r["official_ar"] if not np.isnan(r["official_ar"]) else r["core_ar"]

    for r in records:
        if np.isnan(r["p95_ms"]) or np.isnan(r["core_ar"]):
            continue
        is_frontier = r["row"] in frontier_set
        has_official = not np.isnan(r["official_ar"])
        ar_y = plotted_ar(r)
        marker = "o" if has_official else "^"
        color = "#1f77b4" if is_frontier else "#aec7e8"
        # The bootstrap CI (ar_lo/ar_hi) is computed on core_ar; it is only a valid error bar
        # when the plotted point IS core_ar. Official-AR points get no error bar (bop_toolkit
        # does not bootstrap its own metric here) rather than a CI around the wrong value.
        yerr = None
        if not has_official and not np.isnan(r["ar_lo"]):
            yerr = [[ar_y - r["ar_lo"]], [r["ar_hi"] - ar_y]]
        ax.errorbar(
            r["p95_ms"],
            ar_y,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=3,
            markersize=7 if is_frontier else 5,
        )
        if is_frontier:
            ax.annotate(
                r["row"], (r["p95_ms"], ar_y), textcoords="offset points", xytext=(4, 4), fontsize=7
            )

    # Reference row annotation (must match the y-coordinate the point above was drawn at)
    ref = next((r for r in records if r["row"] == "A10_exact"), None)
    if ref and not np.isnan(ref["p95_ms"]) and not np.isnan(ref["core_ar"]):
        ax.annotate(
            "A8_k4 / A10_exact\n(default)",
            (ref["p95_ms"], plotted_ar(ref)),
            textcoords="offset points",
            xytext=(4, -14),
            fontsize=7,
            color="#555",
        )

    # Pareto frontier step line
    if frontier_names:
        fp = [
            r
            for r in records
            if r["row"] in frontier_set and not np.isnan(r["p95_ms"]) and not np.isnan(r["core_ar"])
        ]
        fp_sorted = sorted(fp, key=lambda r: r["p95_ms"])
        xs = [r["p95_ms"] for r in fp_sorted]
        ys = [plotted_ar(r) for r in fp_sorted]
        ax.step(xs, ys, where="post", color="#1f77b4", linewidth=1.2, alpha=0.5)

    ax.axvline(TARGET_P95_MS, color="#e55", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.annotate(
        "200 ms\n(reported, not required)",
        (TARGET_P95_MS, ax.get_ylim()[0]),
        textcoords="offset points",
        xytext=(3, 4),
        color="#e55",
        fontsize=7,
    )

    ax.set_xlabel("update-path p95 per ObjectTrack (ms)")
    ax.set_ylabel("BOP AR (filled = official, open = core)")
    ax.set_title("A10 ICP schedule sweep — T-LESS accuracy–latency Pareto")
    legend_patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            label="Pareto frontier",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#aec7e8",
            label="non-frontier (core AR)",
            markersize=5,
        ),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"pareto plot: {out}")

    # Curated copy
    fig_dir = REPO / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "deployment_tless_pareto.png", dpi=150)
    plt.close(fig)


def write_report(
    records: list[dict[str, Any]],
    frontier_names: list[str],
    dataset: str,
    outputs: Path,
    xyzibd: dict[str, Any] | None = None,
) -> str:
    lines = []
    lines.append(f"# Deployment milestone — {dataset.upper()} results\n")
    lines.append("<!-- auto-generated by tools/deployment_report.py — do not edit by hand -->\n")
    lines.append(f"Dataset: {dataset}  \nChain: A10 (A8 chain at k=4, ICP schedule sweep)\n\n")
    lines.append("Every number here comes from files under `outputs/`.\n")

    lines.append("\n## Summary\n")
    lines.append("- ROI crop (F1) + Class A fixes applied to all rows except `A10_exact`.\n")
    lines.append("- 200 ms p95 target: **reported, not required** (P2).\n")
    lines.append("- Pareto figure: `docs/figures/deployment_tless_pareto.png`.\n")

    lines.append("\n## Sweep table\n")
    lines.append("| Row | Config | p50 ms | p95 ms | Core AR | Official AR | Frontier? |\n")
    lines.append("|---|---|---:|---:|---:|---:|:---:|\n")
    frontier_set = set(frontier_names)
    for r in records:
        p50 = f"{r['p50_ms']:.1f}" if not np.isnan(r["p50_ms"]) else "—"
        p95 = f"{r['p95_ms']:.1f}" if not np.isnan(r["p95_ms"]) else "—"
        core = f"{r['core_ar']:.4f}" if not np.isnan(r["core_ar"]) else "—"
        off = f"{r['official_ar']:.4f}" if not np.isnan(r["official_ar"]) else "—"
        front = "✓" if r["row"] in frontier_set else ""
        lines.append(f"| {r['row']} | {r['label']} | {p50} | {p95} | {core} | {off} | {front} |\n")

    lines.append(
        "\n*Core AR = mean AR from `gt_rows.parquet` (bootstrap 95 % CI computed but"
        " not shown here; see the JSON file). Official AR = `bop_toolkit` scores_bop19.*\n"
    )

    if xyzibd is not None and not np.isnan(xyzibd["core_ar"]):
        lines.append("\n## XYZ-IBD cross-check (core AR only, D4)\n")
        lines.append("| Row | Config | Core AR | 95% CI |\n")
        lines.append("|---|---|---:|---:|\n")
        lines.append(
            f"| {xyzibd['row']} | {xyzibd['label']} | {xyzibd['core_ar']:.4f} "
            f"| [{xyzibd['ar_lo']:.4f}, {xyzibd['ar_hi']:.4f}] |\n"
        )
        lines.append(
            "\nNo BOP19 GT for the xyzibd test split; official AR is not computable for this row.\n"
        )

    lines.append("\n## Latency protocol\n")
    lines.append(
        "D13 frozen protocol (`configs/benchmark.yaml`): 50 warm-up + 1000 timed"
        " ObjectTrack samples, load average < 2.0, `perf_counter` per stage.  "
        "No CUDA on the update path — `cuda_sync: not_applicable`.  "
        "n printed next to every percentile; p95 over 1000 samples is the 950th value.\n"
    )

    lines.append("\n## Limitations\n")
    lines.append("- XYZ-IBD cross-check row: core AR only (no BOP19 GT for xyzibd test split).\n")
    lines.append(
        "- Confidence model fitted on the default-schedule signal distribution;"
        " cheap-schedule rows apply it out of distribution (D11). "
        "See `A10_l1_i15_p750_posescore` control row for the geometry-only view.\n"
    )
    lines.append("- p95 is a property of this host (Pascal TITAN X, no sudo, 1 MB/s network).\n")

    text = "".join(lines)
    out = outputs / f"{dataset}_deployment_report.md"
    out.write_text(text)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Stem for output PNG (default: outputs/<dataset>_deployment_pareto)",
    )
    args = ap.parse_args()

    records = collect_rows(args.dataset, args.outputs)
    if not records:
        print(f"no rows found for {args.dataset} in {args.outputs}", file=sys.stderr)
        sys.exit(1)

    frontier = pareto_frontier(records)

    # Write provenance JSON
    json_out = args.outputs / f"{args.dataset}_deployment_pareto.json"
    json_out.write_text(json.dumps(records, indent=2))
    print(f"pareto json: {json_out}")

    plot_stem = args.plot or (args.outputs / f"{args.dataset}_deployment_pareto")
    plot_pareto(records, frontier, Path(str(plot_stem) + ".png"))

    xyzibd = collect_xyzibd(args.outputs) if args.dataset == "tless" else None

    report_path = write_report(records, frontier, args.dataset, args.outputs, xyzibd=xyzibd)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
