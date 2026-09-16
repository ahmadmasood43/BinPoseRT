#!/usr/bin/env python
"""Beta analysis and report: refinement rows (A2 / A3 / A4) against the un-refined row they
started from (A1), from ``outputs/`` only (D12).

    uv run python tools/refine_report.py --before A1 A2 A3 A4 > docs/results_beta_tless.md

Per refinement row this writes ``<refine stage>/analysis/``: ``scored.parquet`` (every hypothesis
scored against GT before/after; cached), ``gate_stats.json``, ``sweep_val.parquet`` +
``gate_caps.json`` (alpha / beta sweep on the val scenes, choice, held-out check),
``strata.json`` (before/after AR by initial-error × visibility bin, BOP protocol) and the
``gallery/`` of gate mistakes. The Markdown on stdout summarises all rows.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from report import fmt, load_row  # noqa: E402

from binposert.evaluate.refinement import (  # noqa: E402
    choose_gate_caps,
    gate_statistics,
    gate_sweep,
    score_refinement,
    stratify_before_after,
)
from binposert.pipeline.refine_stage import DETAILS_FILE  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.refine import GateParams  # noqa: E402
from binposert.viz import make_refine_galleries  # noqa: E402

VAL_SCENES = [1, 6, 11, 16]
ALPHAS = [0.1, 0.15, 0.25, 0.4, 0.6, math.inf]
BETAS = [15.0, 30.0, 45.0, 90.0, math.inf]
# the silhouette and fitness checks are swept too: alpha / beta alone are not where the gate loses
EXTRA = {
    "min_iou": [0.0, 0.3, 0.5],
    "max_iou_drop": [0.05, 0.1, 0.2, 1.0],
    "min_fitness": [0.0, 0.3],
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return _json_safe(obj.item())
    return obj


def _dump(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)


def analyse_row(
    row: dict[str, Any],
    before_rows: pd.DataFrame,
    val_scenes: list[int],
    n_gallery: int,
    n_workers: int,
) -> dict[str, Any]:
    manifest = json.loads((Path(row["evaluate_dir_abs"]) / "run_manifest.json").read_text())
    refine_dir = Path(manifest["stages"]["refine"]["dir"])
    ana = refine_dir / "analysis"
    ana.mkdir(exist_ok=True)
    refiner_cfg = manifest["config"]["refiner"]
    gate_cfg = dict(refiner_cfg["params"].get("gate", {}))
    default = GateParams(**gate_cfg)
    dataset = make_dataset(manifest["config"], REPO)

    scored_path = ana / "scored.parquet"
    if scored_path.exists():
        scored = pd.read_parquet(scored_path)
    else:
        details = pd.read_parquet(refine_dir / DETAILS_FILE)
        scored = score_refinement(details, dataset, n_model_points=0, n_workers=n_workers)
        scored.to_parquet(scored_path, index=False)

    stats = gate_statistics(scored)
    _dump(ana / "gate_stats.json", stats)

    sweep = gate_sweep(scored, ALPHAS, BETAS, default, scenes=val_scenes, extra=EXTRA)
    sweep.to_parquet(ana / "sweep_val.parquet", index=False)
    choice = choose_gate_caps(sweep, default)
    heldout = [s for s in dataset.scene_ids if s not in val_scenes]
    chosen = dataclasses.replace(default, **choice["params"])
    held = {
        "scenes": heldout,
        "default": gate_statistics(scored[scored.scene_id.isin(heldout)], default),
        "chosen": gate_statistics(scored[scored.scene_id.isin(heldout)], chosen),
    }
    caps = {"val_scenes": val_scenes, "default": gate_cfg, "choice": choice, "heldout": held}
    _dump(ana / "gate_caps.json", caps)

    after_rows = pd.read_parquet(Path(row["evaluate_dir_abs"]) / "gt_rows.parquet")
    strata = {
        "all": stratify_before_after(before_rows, after_rows),
        "heldout": stratify_before_after(
            before_rows[before_rows.scene_id.isin(heldout)],
            after_rows[after_rows.scene_id.isin(heldout)],
        ),
    }
    _dump(ana / "strata.json", strata)

    galleries = make_refine_galleries(scored, dataset, ana / "gallery", n=n_gallery)
    summary = json.loads((refine_dir / "refine_summary.json").read_text())
    return {
        "refiner": refiner_cfg["name"],
        "variant": refiner_cfg["params"]["variant"],
        "refine_dir": str(refine_dir.relative_to(REPO))
        if refine_dir.is_relative_to(REPO)
        else str(refine_dir),
        "refine_summary": summary,
        "gate": stats,
        "caps": caps,
        "strata": strata,
        "galleries": {k: str(p.relative_to(REPO)) for k, p in galleries.items()},
    }


# ----------------------------------------------------------------------------- plot


def plot_strata(
    before: dict[str, Any], rows: list[tuple[dict[str, Any], dict[str, Any]]], path: Path
) -> None:
    """Before/after AR per initial-error bin and per visibility bin (grouped bars, one bar per
    row), plus the per-hypothesis success ladder coarse → depth init → ICP → gate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st0 = rows[0][1]["strata"]["all"]
    ib = [list(b) for b in st0["initial_bins"]]
    vb = [list(b) for b in st0["visibility_bins"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def grouped(ax: Any, keys: list[Any], labels: list[str], pick: Any, title: str) -> None:
        n_bars = len(rows) + 1
        width = 0.8 / n_bars
        x = np.arange(len(keys))
        first = rows[0][1]["strata"]["all"]["strata"]
        ax.bar(
            x - 0.4 + width / 2,
            [100 * pick(first, k)["ar_before"] for k in keys],
            width,
            color="0.6",
            label=f"{before['experiment']} (before)",
        )
        for i, (r, a) in enumerate(rows):
            vals = [100 * pick(a["strata"]["all"]["strata"], k)["ar_after"] for k in keys]
            ax.bar(
                x - 0.4 + width * (i + 1.5),
                vals,
                width,
                color=colors[i],
                label=f"{r['experiment']} {a['refiner']}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lbl}\n(n={pick(first, k)['n_gt']})" for lbl, k in zip(labels, keys, strict=True)],
            fontsize=8,
        )
        ax.set_ylabel("AR (core, %)")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    grouped(
        axes[0],
        [*ib, None],
        [_bin_label(b, "mm") for b in ib] + ["all"],
        lambda st, k: _find(st, k, None),
        "by initial error (MSSD of the un-refined prediction)",
    )
    grouped(
        axes[1],
        [*vb, None],
        [_bin_label(b, "") for b in vb] + ["all"],
        lambda st, k: _find(st, None, k),
        "by visible fraction",
    )
    axes[0].legend(fontsize=8, loc="upper right")

    ax = axes[2]
    steps = ["coarse", "init", "gate_off", "final"]
    names = ["coarse", "+ depth init", "+ ICP", "+ gate"]
    for i, (r, a) in enumerate(rows):
        ok = a["gate"]["success_0.1d"]
        ax.plot(
            names,
            [100 * ok[k] for k in steps],
            marker="o",
            color=colors[i],
            label=f"{r['experiment']} {a['refiner']}",
        )
    ax.set_ylabel("hypotheses within 0.1 d (%)")
    ax.set_ylim(0, 100)
    ax.set_title("per-hypothesis success along the refinement", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"{before['experiment']} → depth refinement: before → after AR by regime", fontsize=11
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- rendering


def _cell(entry: dict[str, Any]) -> str:
    if entry["n_gt"] == 0:
        return "—"
    return f"{fmt(entry['ar_before'])} → {fmt(entry['ar_after'])} ({entry['n_gt']})"


def _bin_label(b: list[float] | tuple[float, float] | None, unit: str) -> str:
    if b is None:
        return "all"
    lo, hi = b
    hi_s = "∞" if not math.isfinite(hi) else (f"{hi:g}" if unit == "mm" else f"{hi:.1f}")
    lo_s = f"{lo:g}" if unit == "mm" else f"{lo:.1f}"
    return f"[{lo_s}, {hi_s}) {unit}".strip()


def _find(strata: list[dict[str, Any]], initial: Any, visibility: Any) -> dict[str, Any]:
    key = (
        None if initial is None else list(initial),
        None if visibility is None else list(visibility),
    )
    for e in strata:
        if (e["initial"], e["visibility"]) == key:
            return e
    raise KeyError(key)


def render(
    before: dict[str, Any], rows: list[tuple[dict[str, Any], dict[str, Any]]], dataset: str
) -> str:
    out = [
        f"# Beta results — {dataset}",
        "",
        f"Depth refinement (D8) on top of {before['experiment']} "
        f"({before['segmenter']} masks + {before['estimator']} coarse poses). AR in %, BOP19 "
        "protocol; *core* = `binposert.evaluate`, *official* = `bop_toolkit` via "
        "`tools/bop_eval.sh`.",
        "",
        "## Rows",
        "",
        "| exp | refiner | AR core | AR official | VSD | MSSD | MSPD | rejected | "
        "median refine s | run |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _row_line(r: dict[str, Any], refiner: str, rej: str, secs: str) -> str:
        rep, off = r["report"], r["official"]
        ar_off = fmt(off["bop19_average_recall"]) if off else "—"
        vsd = fmt(off["bop19_average_recall_vsd"]) if off else fmt(rep["ar_vsd"])
        mssd = fmt(off["bop19_average_recall_mssd"]) if off else fmt(rep["ar_mssd"])
        mspd = fmt(off["bop19_average_recall_mspd"]) if off else fmt(rep["ar_mspd"])
        return (
            f"| {r['experiment']} | {refiner} | {fmt(rep['ar'])} | {ar_off} | {vsd} | {mssd} | "
            f"{mspd} | {rej} | {secs} | `{r['evaluate_dir']}` |"
        )

    out.append(_row_line(before, "— (before)", "—", "—"))
    for r, a in rows:
        g = a["gate"]
        out.append(
            _row_line(
                r,
                a["refiner"],
                f"{100 * g['rejection_rate']:.1f}%",
                f"{a['refine_summary']['median_seconds']:.2f}",
            )
        )

    out += [
        "",
        "## Before → after AR by initial error (core metric, pooled over GT, all scenes)",
        "",
        "Initial error = MSSD of the closest un-refined prediction to the GT instance. "
        "GT instances without an un-refined prediction have no initial error (*no pred*).",
        "",
    ]
    ib = [tuple(b) for b in rows[0][1]["strata"]["all"]["initial_bins"]]
    vb = [tuple(b) for b in rows[0][1]["strata"]["all"]["visibility_bins"]]
    header = "| exp | " + " | ".join(_bin_label(b, "mm") for b in ib) + " | all | no pred |"
    out += [header, "|---|" + "---|" * (len(ib) + 2)]
    for r, a in rows:
        st = a["strata"]["all"]
        cells = [_cell(_find(st["strata"], list(b), None)) for b in ib]
        cells.append(_cell(_find(st["strata"], None, None)))
        out.append(f"| {r['experiment']} | " + " | ".join(cells) + f" | {st['n_no_prediction']} |")

    out += ["", "## Before → after AR by initial error × visibility (all scenes)", ""]
    for r, a in rows:
        st = a["strata"]["all"]["strata"]
        out += [
            f"**{r['experiment']} ({a['refiner']})**",
            "",
            "| initial \\ visibility | " + " | ".join(_bin_label(b, "") for b in vb) + " | all |",
            "|---|" + "---|" * (len(vb) + 1),
        ]
        for b in [*ib, None]:
            cells = [_cell(_find(st, list(b) if b else None, list(v))) for v in vb]
            cells.append(_cell(_find(st, list(b) if b else None, None)))
            out.append(f"| {_bin_label(b, 'mm')} | " + " | ".join(cells) + " |")
        out.append("")

    out += [
        "## Gate",
        "",
        "Per-hypothesis view (MSSD/MSPD of the same hypothesis before and after, valid GT only). "
        "*precision* = share of gate rejections where the candidate was no closer to GT than the "
        "coarse pose; *rejected good* = rejections of a candidate that was closer and within "
        "0.1 d; "
        "*accepted worse* = acceptances that moved away from GT; *broke* / *fixed* = acceptances "
        "that crossed the 0.1 d success line downwards / upwards. *rf* = mean per-hypothesis "
        "recall fraction (MSSD + MSPD thresholds) of the coarse poses, of all candidates (gate "
        "off) and of the gated result.",
        "",
        "| exp | n | rejected | reasons | precision strict / success | rejected good | "
        "avoided break | accepted worse | broke | fixed | rf coarse | rf gate off | rf final | "
        "success coarse → init → gate off → final |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, a in rows:
        g = a["gate"]
        reasons = ", ".join(
            f"{k} {v}" for k, v in sorted(g["reasons"].items(), key=lambda kv: -kv[1])
        )
        rg, acc, rf, ok = (
            g["rejected_by_gate"],
            g["accepted"],
            g["recall_fraction"],
            g["success_0.1d"],
        )
        out.append(
            f"| {r['experiment']} | {g['n']} | {g['n'] - g['n_accepted']} "
            f"({100 * g['rejection_rate']:.1f}%) | {reasons} | {fmt(rg['precision'])} / "
            f"{fmt(rg['precision_success'])} | {rg['rejected_good']} | {rg['avoided_break']} | "
            f"{acc['worsened']} | {acc['broke']} | {acc['fixed']} | "
            f"{fmt(rf['coarse'])} | {fmt(rf['gate_off'])} | {fmt(rf['final'])} | "
            f"{fmt(ok['coarse'])} → {fmt(ok['init'])} → {fmt(ok['gate_off'])} → "
            f"{fmt(ok['final'])} |"
        )
    out += [
        "",
        "Per check (gate rejections only): n · strict precision · success precision · "
        "rejected good · avoided break",
        "",
    ]
    for r, a in rows:
        per = a["gate"]["rejected_by_gate"]["per_reason"]
        cells = [
            f"{k}: {v['n']} · {fmt(v['precision'])} · {fmt(v['precision_success'])} · "
            f"{v['rejected_good']} · {v['avoided_break']}"
            for k, v in sorted(per.items(), key=lambda kv: -kv[1]["n"])
        ]
        out.append(f"- {r['experiment']}: " + "; ".join(cells))

    val = rows[0][1]["caps"]["val_scenes"]
    fields = list(rows[0][1]["caps"]["choice"]["params"])
    out += [
        "",
        f"## Gate thresholds swept on val scenes {val}",
        "",
        f"Grid over {', '.join(fields)} (∞ = check off). Objective = rf final over the val scenes; "
        "the default stays unless the best combination beats it by ≥ 0.1 pt. Held-out = the other "
        "scenes, default → chosen.",
        "",
        "| exp | default | objective | best | objective | gain | chosen | held-out rf | "
        "held-out success | held-out net successes | held-out rejected |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _p(d: dict[str, Any]) -> str:
        return "(" + ", ".join(f"{float(d[f]):g}" for f in fields) + ")"

    def net(st: dict[str, Any]) -> int:
        rg = st["rejected_by_gate"]
        return int(rg["avoided_break"] - rg["rejected_good"])

    for r, a in rows:
        c, ch = a["caps"], a["caps"]["choice"]
        hd, hc = c["heldout"]["default"], c["heldout"]["chosen"]
        out.append(
            f"| {r['experiment']} | {_p(c['default'])} | {fmt(ch['default_objective'])} | "
            f"{_p(ch['best'])} | {fmt(ch['best_objective'])} | "
            f"{100 * ch['gain_over_default']:+.2f} | "
            f"{_p(ch['params'])}{' *changed*' if ch['changed'] else ''} | "
            f"{fmt(hd['recall_fraction']['final'])} → {fmt(hc['recall_fraction']['final'])} | "
            f"{fmt(hd['success_0.1d']['final'])} → {fmt(hc['success_0.1d']['final'])} | "
            f"{net(hd):+d} → {net(hc):+d} | "
            f"{100 * hd['rejection_rate']:.1f}% → {100 * hc['rejection_rate']:.1f}% |"
        )

    out += ["", "## Galleries", ""]
    for r, a in rows:
        for kind, path in a["galleries"].items():
            out.append(f"- {r['experiment']} {kind}: `{path}`")
    out += [
        "",
        "Commits: "
        + ", ".join(
            f"{r['experiment']}={r['git_commit']}" for r in [before, *[r for r, _ in rows]]
        ),
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+", help="refinement rows, e.g. A2 A3 A4")
    ap.add_argument("--before", default="A1", help="the un-refined row they build on")
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--val-scenes", type=int, nargs="+", default=VAL_SCENES)
    ap.add_argument("--n-gallery", type=int, default=20)
    ap.add_argument("--n-workers", type=int, default=0, help="0 = all cores but one")
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="where to write the strata figure (default: outputs/<dataset>_beta_strata.png)",
    )
    args = ap.parse_args()
    import os

    n_workers = args.n_workers or max(1, (os.cpu_count() or 2) - 1)

    before = load_row(args.before, args.dataset, args.outputs)
    if before is None:
        sys.exit(f"no run manifest for {args.before} on {args.dataset}")
    before_rows = pd.read_parquet(Path(before["evaluate_dir_abs"]) / "gt_rows.parquet")
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for exp in args.experiments:
        row = load_row(exp, args.dataset, args.outputs)
        if row is None:
            print(f"warning: no run manifest for {exp} on {args.dataset}", file=sys.stderr)
            continue
        print(f"analysing {exp} ...", file=sys.stderr)
        rows.append(
            (row, analyse_row(row, before_rows, args.val_scenes, args.n_gallery, n_workers))
        )
    if not rows:
        sys.exit("nothing to report")
    plot = args.plot or args.outputs / f"{args.dataset}_beta_strata.png"
    plot_strata(before, rows, plot)
    print(f"strata figure: {plot}", file=sys.stderr)
    print(render(before, rows, args.dataset))


if __name__ == "__main__":
    main()
