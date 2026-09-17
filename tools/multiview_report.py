#!/usr/bin/env python
"""Gamma analysis and report from ``outputs/`` only (D12): the AR-vs-views curve with bootstrap
error bars, the extrinsic-perturbation heat-map, association correctness against GT, symmetry-
branch statistics and the galleries.

    uv run python tools/multiview_report.py --dataset tless --ks 1 2 3 4 --plot outputs/tless_gamma

Rows are the runs ``tools/run_gamma.sh`` writes: ``A6_k<k>_<fusion>`` (fusion none | best | mean),
``A7_k<k>`` (mean + joint ICP) and ``A6_sweep_t<t>_r<r>``. Per multi-view row this writes
``<associate stage>/analysis/association.json`` and ``<fuse stage>/analysis/gallery/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from report import fmt, load_row  # noqa: E402

from binposert.evaluate.association import association_metrics, label_hypotheses  # noqa: E402
from binposert.pipeline.multiview_stage import TRACKS_FILE  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.viz.multiview_gallery import make_multiview_galleries  # noqa: E402

FUSIONS = [
    ("none", "single view (no fusion)"),
    ("best", "best single view"),
    ("mean", "mean"),
    ("joint", "mean + joint ICP"),
]


def row_name(k: int, fusion: str) -> str:
    return f"A7_k{k}" if fusion == "joint" else f"A6_k{k}_{fusion}"


def load_run(name: str, dataset: str, outputs: Path) -> dict[str, Any] | None:
    row = load_row(name, dataset, outputs)
    if row is None:
        return None
    manifest = json.loads(
        (outputs / "runs" / f"{name}_{dataset}" / "run_manifest.json").read_text()
    )
    stages = manifest["stages"]
    row["associate_dir"] = Path(stages["associate"]["dir"]) if "associate" in stages else None
    row["fuse_dir"] = Path(stages["fuse"]["dir"]) if "fuse" in stages else None
    row["config"] = manifest["config"]
    return row


def bootstrap_ar(
    evaluate_dir: Path, n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95 % interval of the per-GT AR approximation (mean of the three recall fractions),
    bootstrapping over scenes."""
    rows = pd.read_parquet(evaluate_dir / "gt_rows.parquet")
    cols = [c for c in ("ar_vsd", "ar_mssd", "ar_mspd") if c in rows and rows[c].notna().any()]
    rows = rows.assign(ar=rows[cols].mean(axis=1))
    per_scene = {int(cast(Any, s)): g.ar.to_numpy() for s, g in rows.groupby("scene_id")}
    scenes = sorted(per_scene)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(scenes, size=len(scenes), replace=True)
        stats.append(float(np.concatenate([per_scene[s] for s in pick]).mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(rows.ar.mean()), float(lo), float(hi)


def analyse_association(row: dict[str, Any], dataset: Any) -> dict[str, Any]:
    assoc = row["associate_dir"]
    out = assoc / "analysis"
    out.mkdir(exist_ok=True)
    cached = out / "association.json"
    if cached.exists():
        return json.loads(cached.read_text())
    tracks = pd.read_parquet(assoc / TRACKS_FILE)
    labels = label_hypotheses(tracks, dataset)
    tracks.assign(gt_index=labels.to_numpy()).to_parquet(
        out / "labelled_tracks.parquet", index=False
    )
    m = association_metrics(tracks, labels)
    summary = json.loads((assoc / "associate_summary.json").read_text())
    result = {**m.to_dict(), "summary": summary, "mixed": m.mixed[:200]}
    cached.write_text(json.dumps(result, indent=2))
    return result


def sweep_table(dataset: str, outputs: Path, ts: list[str], rs: list[str]) -> pd.DataFrame | None:
    cells = {}
    for t in ts:
        for r in rs:
            row = load_row(f"A6_sweep_t{t}_r{r}", dataset, outputs)
            if row is not None:
                rep = row["report"]
                # the sweep rows skip VSD: AR over the metrics that were computed
                vals = [rep[k] for k in ("ar_vsd", "ar_mssd", "ar_mspd") if rep[k] is not None]
                cells[(t, r)] = float(np.nanmean(vals))
    if not cells:
        return None
    table = pd.DataFrame(index=ts, columns=rs, dtype=float)
    for (t, r), v in cells.items():
        table.loc[t, r] = v
    table.index.name = "dt_mm \\ dtheta_deg"
    return table


def plot_curve(curve: dict[str, dict[int, dict[str, float]]], path: Path, dataset: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for fusion, label in FUSIONS:
        pts = curve.get(fusion, {})
        if not pts:
            continue
        ks = sorted(pts)
        ar = [100 * pts[k]["ar"] for k in ks]
        lo = [100 * (pts[k]["boot"] - pts[k]["lo"]) for k in ks]
        hi = [100 * (pts[k]["hi"] - pts[k]["boot"]) for k in ks]
        ax.errorbar(ks, ar, yerr=[lo, hi], marker="o", capsize=3, label=label)
    ax.set_xlabel("views per group")
    ax.set_ylabel("AR (BOP19 core, %)")
    ax.set_xticks(sorted({k for pts in curve.values() for k in pts}))
    ax.set_title(f"{dataset}: AR vs number of fused views (95 % bootstrap over scenes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_sweep(table: pd.DataFrame, path: Path, dataset: str, k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    vals = 100 * table.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="viridis", origin="lower")
    ax.set_xticks(range(len(table.columns)), [str(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)), [str(i) for i in table.index])
    ax.set_xlabel("rotation error per extra view (deg)")
    ax.set_ylabel("translation error per extra view (mm)")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="AR (MSSD/MSPD, %)")
    ax.set_title(f"{dataset}: {k}-view mean under extrinsic perturbation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def render(
    dataset: str,
    curve: dict[str, dict[int, dict[str, Any]]],
    assoc: dict[str, dict[str, Any]],
    fusion_stats: dict[str, dict[str, Any]],
    sweep: pd.DataFrame | None,
    sweep_k: int,
    plot: Path | None,
) -> str:
    ks = sorted({k for pts in curve.values() for k in pts})
    out = [
        f"# Gamma results — {dataset}",
        "",
        "AR in %, BOP19 protocol on the view-group images. *core* = `binposert.evaluate`; "
        "*official* = `bop_toolkit` on the same images (`tools/bop_eval.sh` with the row's "
        "`targets_subset.json`). Error bars: 95 % bootstrap over scenes of the per-GT recall "
        "approximation.",
        "",
        "## AR vs number of views",
        "",
        "| views | " + " | ".join(label for _, label in FUSIONS) + " |",
        "|---|" + "---|" * len(FUSIONS),
    ]
    for k in ks:
        cells = []
        for fusion, _ in FUSIONS:
            p = curve.get(fusion, {}).get(k)
            if p is None:
                cells.append("—")
                continue
            off = f" / **{fmt(p['official'])}**" if p.get("official") is not None else ""
            cells.append(f"{fmt(p['ar'])}{off} [{fmt(p['lo'])}, {fmt(p['hi'])}]")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "core / **official**, [bootstrap 95 % interval of core]. Rows with fewer images than "
        "k=1 (strided groups leave 50 mod k images out) are scored on their own images."
    )
    if plot is not None:
        out += ["", f"![AR vs views]({plot.name}_curve.png)"]

    out += [
        "",
        "## Per-metric breakdown",
        "",
        "| row | n img | n GT | AR | VSD | MSSD | MSPD | n pred |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fusion, label in FUSIONS:
        for k in ks:
            p = curve.get(fusion, {}).get(k)
            if p is None:
                continue
            r = p["report"]
            cells = [
                f"{label} k={k}",
                str(r.get("n_images")),
                str(r["n_gt"]),
                fmt(r["ar"]),
                fmt(r["ar_vsd"]),
                fmt(r["ar_mssd"]),
                fmt(r["ar_mspd"]),
                str(r["n_predictions"]),
            ]
            out.append("| " + " | ".join(cells) + " |")

    out += [
        "",
        "## Association correctness (vs GT instances)",
        "",
        "| views | groups | tracks | multi-view tracks | labelled hyps | track purity | "
        "member purity | completeness | mixed tracks | split instances |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in ks:
        a = assoc.get(str(k))
        if a is None:
            continue
        cells = [
            str(k),
            str(a["n_groups"]),
            str(a["n_tracks"]),
            str(a["n_multi_tracks"]),
            str(a["n_labelled"]),
            fmt(a["track_purity"]),
            fmt(a["member_purity"]),
            fmt(a["completeness"]),
            str(a["n_mixed_tracks"]),
            f"{a['n_split_instances']} (of {a['n_instances_multi_view']})",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out += [
        "",
        "A hypothesis is labelled with the nearest GT instance of its object in its View "
        "(translation within 0.5 d); purity and completeness are computed on labelled "
        "hypotheses only.",
    ]

    out += [
        "",
        "## Fusion statistics",
        "",
        "| row | tracks | multi-view | members aligned by symmetry | median dispersion mm / deg | "
        "joint accepted | joint reasons | median s/track |",
        "|---|---|---|---|---|---|---|---|",
    ]
    nan = float("nan")
    for name, s in fusion_stats.items():
        d_mm = s.get("median_dispersion_mm") or nan
        d_deg = s.get("median_dispersion_deg") or nan
        ja = s.get("joint_acceptance_rate")
        cells = [
            name,
            str(s["n_tracks"]),
            str(s.get("n_multi_view_tracks", "—")),
            str(s.get("n_members_aligned_by_symmetry", "—")),
            f"{d_mm:.2f} / {d_deg:.1f}",
            fmt(ja) if ja is not None else "—",
            str(s.get("joint_reasons", "—")),
            f"{s.get('median_seconds', nan):.3f}",
        ]
        out.append("| " + " | ".join(cells) + " |")

    if sweep is not None:
        out += [
            "",
            f"## Extrinsic perturbation sweep ({sweep_k} views, mean fusion, core AR without VSD)",
            "",
            "| δt mm \\ δθ ° | " + " | ".join(str(c) for c in sweep.columns) + " |",
            "|---|" + "---|" * len(sweep.columns),
        ]
        for t in sweep.index:
            out.append(
                f"| {t} | "
                + " | ".join(
                    fmt(sweep.loc[t, c]) if pd.notna(sweep.loc[t, c]) else "—"
                    for c in sweep.columns
                )
                + " |"
            )
        out.append("")
        out.append(
            "Every View but the first of a group is perturbed by a rigid transform of the given "
            "magnitude (random direction, seeded) in its camera frame before association and "
            "fusion."
        )
        if plot is not None:
            out.append(f"\n![sweep]({plot.name}_sweep.png)")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--sweep-k", type=int, default=4)
    ap.add_argument("--sweep-t", nargs="+", default=["0", "1", "2", "5", "10"])
    ap.add_argument("--sweep-r", nargs="+", default=["0", "0.1", "0.25", "0.5", "1"])
    ap.add_argument(
        "--plot", type=Path, default=None, help="prefix for <prefix>_curve.png / _sweep.png"
    )
    ap.add_argument("--gallery-n", type=int, default=12)
    args = ap.parse_args()

    dataset_cfg: dict[str, Any] | None = None
    curve: dict[str, dict[int, dict[str, Any]]] = {}
    assoc: dict[str, dict[str, Any]] = {}
    fusion_stats: dict[str, dict[str, Any]] = {}
    ds = None
    for fusion, _ in FUSIONS:
        for k in args.ks:
            row = load_run(row_name(k, fusion), args.dataset, args.outputs)
            if row is None:
                continue
            if ds is None:
                dataset_cfg = row["config"]
                ds = make_dataset(dataset_cfg, REPO)
            boot, lo, hi = bootstrap_ar(Path(row["evaluate_dir_abs"]))
            official = row["official"]["bop19_average_recall"] if row["official"] else None
            curve.setdefault(fusion, {})[k] = {
                "ar": row["report"]["ar"],
                "boot": boot,
                "lo": lo,
                "hi": hi,
                "official": official,
                "report": row["report"],
            }
            if row["associate_dir"] is not None and str(k) not in assoc and k > 1:
                assoc[str(k)] = analyse_association(row, ds)
            if row["fuse_dir"] is not None and fusion != "none":
                fusion_stats[row_name(k, fusion)] = json.loads(
                    (row["fuse_dir"] / "fuse_summary.json").read_text()
                )
                if k > 1:
                    gallery = row["fuse_dir"] / "analysis" / "gallery"
                    if not gallery.exists():
                        make_multiview_galleries(
                            ds,
                            row["associate_dir"],
                            row["fuse_dir"],
                            gallery,
                            assoc.get(str(k), {}).get("mixed", []),
                            n=args.gallery_n,
                        )
    sweep = sweep_table(args.dataset, args.outputs, args.sweep_t, args.sweep_r)
    if args.plot is not None:
        if curve:
            plot_curve(curve, args.plot.with_name(args.plot.name + "_curve.png"), args.dataset)
        if sweep is not None:
            plot_sweep(
                sweep,
                args.plot.with_name(args.plot.name + "_sweep.png"),
                args.dataset,
                args.sweep_k,
            )
    sys.stdout.write(
        render(args.dataset, curve, assoc, fusion_stats, sweep, args.sweep_k, args.plot)
    )


if __name__ == "__main__":
    main()
