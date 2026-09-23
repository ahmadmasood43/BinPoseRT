#!/usr/bin/env python
"""Epsilon analysis and report from ``outputs/`` only (D12): AR vs Views used per policy with
bootstrap error bars, the paired NBV − random difference per episode, what the NBV score picks
(angle to the Views already used), the Verdict-stopped loop, and the cost of a step.

    uv run python tools/nbv_report.py --dataset xyzibd

Rows are the runs ``tools/run_epsilon.sh`` writes: ``A9_<policy>_b<budget>_g<start group>`` and
``A9_<policy>_verdict_g<g>``. Every policy of one start group is evaluated on the same reference
Views, so rows of one group are paired by (scene, GT instance). Writes
``outputs/<dataset>_epsilon_report.md``, ``outputs/<dataset>_epsilon_curve.{png,json}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from report import fmt, load_row  # noqa: E402

from binposert.pipeline.run import make_dataset  # noqa: E402

ROW_RE = re.compile(
    r"^A9_(?P<policy>nbve|nbv|random|fixed|oracle)_(?:b(?P<budget>\d+)|(?P<verdict>verdict))_g(?P<g>\d+)$"
)
POLICIES = [
    ("fixed", "fixed order (Gamma's groups)"),
    ("random", "random next"),
    ("nbv", "NBV (D14, weight 1 − Confidence)"),
    ("nbve", "NBV (weight = entropy of Confidence)"),
    ("oracle", "oracle (GT-scored next View, ceiling)"),
]
AR_COLS = ("ar_vsd", "ar_mssd", "ar_mspd")


def find_rows(dataset: str, outputs: Path) -> list[dict[str, Any]]:
    rows = []
    for d in sorted((outputs / "runs").glob(f"A9_*_{dataset}")):
        name = d.name[: -len(dataset) - 1]
        m = ROW_RE.match(name)
        if not m or not (d / "run_manifest.json").exists():
            continue
        row = load_row(name, dataset, outputs)
        if row is None:
            continue
        manifest = json.loads((d / "run_manifest.json").read_text())
        nbv_dir = Path(manifest["stages"]["nbv"]["dir"])
        row.update(
            {
                "name": name,
                "policy": m["policy"],
                "budget": int(m["budget"]) if m["budget"] else None,
                "verdict_loop": m["verdict"] is not None,
                "g": int(m["g"]),
                "nbv_dir": nbv_dir,
                "nbv_summary": json.loads((nbv_dir / "nbv_summary.json").read_text()),
                "episodes": json.loads((nbv_dir / "episodes.json").read_text()),
            }
        )
        rows.append(row)
    return rows


def gt_ar(row: dict[str, Any]) -> pd.DataFrame:
    """Per-GT rows of the evaluate stage with the per-GT AR approximation and the episode key."""
    t = pd.read_parquet(Path(row["evaluate_dir_abs"]) / "gt_rows.parquet")
    cols = [c for c in AR_COLS if c in t and t[c].notna().any()]
    t = t.assign(ar=t[cols].mean(axis=1), g=row["g"])
    return t


def bootstrap(
    values_by_episode: dict[Any, np.ndarray], n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Mean of the pooled values and its 95 % interval, bootstrapping over episodes."""
    keys = sorted(values_by_episode)
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([values_by_episode[k] for k in keys])
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        stats.append(float(np.concatenate([values_by_episode[keys[i]] for i in pick]).mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(pooled.mean()), float(lo), float(hi)


def curve_points(rows: list[dict[str, Any]], n_boot: int) -> dict[str, dict[str, dict[str, Any]]]:
    """``{policy: {x-key: point}}``: per (policy, budget) pooled over start groups — the report's
    AR averaged over its rows, the per-GT bootstrap over (scene, g) episodes, the Views used."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["policy"], "verdict" if r["verdict_loop"] else f"b{r['budget']}")
        groups.setdefault(key, []).append(r)
    for (policy, xkey), rs in sorted(groups.items()):
        per_episode: dict[tuple[int, int], np.ndarray] = {}
        for r in rs:
            t = gt_ar(r)
            for s, grp in t.groupby("scene_id"):
                per_episode[(int(cast(Any, s)), r["g"])] = grp.ar.to_numpy()
        boot, lo, hi = bootstrap(per_episode, n_boot, 0)
        views = [e["n_views_used"] for r in rs for e in r["episodes"]]
        out.setdefault(policy, {})[xkey] = {
            "rows": [r["name"] for r in rs],
            "n_start_groups": len(rs),
            "ar": float(np.mean([r["report"]["ar"] for r in rs])),
            "ar_mssd": float(np.mean([r["report"]["ar_mssd"] for r in rs])),
            "ar_vsd": float(np.mean([r["report"]["ar_vsd"] for r in rs])),
            "ar_mspd": float(np.mean([r["report"]["ar_mspd"] for r in rs])),
            "boot": boot,
            "lo": lo,
            "hi": hi,
            "views_used": float(np.mean(views)),
            "n_episodes": len(per_episode),
            "n_gt": int(sum(len(v) for v in per_episode.values())),
        }
    return out


def paired_difference(
    rows: list[dict[str, Any]], a: str, b: str, n_boot: int
) -> dict[str, dict[str, Any]]:
    """Per budget, the per-episode AR of policy ``a`` minus policy ``b`` on the same reference
    Views and GT (paired by (scene, g)), with a bootstrap interval over episodes and the share
    of episodes ``a`` wins."""
    by: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for r in rows:
        if r["verdict_loop"] or r["policy"] not in (a, b):
            continue
        t = gt_ar(r)
        for s, grp in t.groupby("scene_id"):
            by.setdefault((f"b{r['budget']}", int(cast(Any, s)), r["g"]), {})[r["policy"]] = (
                grp.ar.to_numpy()
            )
    out: dict[str, dict[str, Any]] = {}
    for xkey in sorted({k[0] for k in by}):
        eps = [v for k, v in by.items() if k[0] == xkey and a in v and b in v]
        if not eps:
            continue
        diffs = np.asarray([float(v[a].mean() - v[b].mean()) for v in eps])
        rng = np.random.default_rng(1)
        stats = [
            float(diffs[rng.choice(len(diffs), size=len(diffs), replace=True)].mean())
            for _ in range(n_boot)
        ]
        lo, hi = np.percentile(stats, [2.5, 97.5])
        out[xkey] = {
            "n_episodes": len(eps),
            "mean_diff": float(diffs.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "wins": float((diffs > 0).mean()),
            "ties": float((diffs == 0).mean()),
        }
    return out


def angle_analysis(rows: list[dict[str, Any]], dataset: Any) -> dict[str, dict[str, float]]:
    """What a policy unlocks: the angle (deg) between the chosen View's optical axis and the
    nearest already-used View's, and the distance between their centres (mm), averaged over
    every choice of every episode of the policy."""
    axes: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def cam(scene_id: int, image_id: int) -> tuple[np.ndarray, np.ndarray]:
        key = (scene_id, image_id)
        if key not in axes:
            v, _ = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
            axes[key] = (v.T_world_camera[:3, 2], v.T_world_camera[:3, 3])
        return axes[key]

    out: dict[str, dict[str, float]] = {}
    for policy, _ in POLICIES:
        angles, dists, n_fallback, n_choices, seconds = [], [], 0, 0, []
        for r in rows:
            # the every-View row would swamp the statistic with its 16 forced choices
            if r["policy"] != policy or r["budget"] == 0:
                continue
            for e in r["episodes"]:
                for s in e["steps"]:
                    if s["chosen"] is None:
                        continue
                    n_choices += 1
                    n_fallback += int(s["reason"] == "fallback")
                    if s["seconds_score"] > 0:
                        seconds.append(s["seconds_score"] / max(1, len(s["candidates"])))
                    z_c, c_c = cam(e["scene_id"], int(s["chosen"]))
                    best_a, best_d = np.inf, np.inf
                    for u in s["image_ids"]:
                        z_u, c_u = cam(e["scene_id"], int(u))
                        best_a = min(
                            best_a, float(np.degrees(np.arccos(np.clip(z_c @ z_u, -1, 1))))
                        )
                        best_d = min(best_d, float(np.linalg.norm(c_c - c_u)))
                    angles.append(best_a)
                    dists.append(best_d)
        if n_choices:
            out[policy] = {
                "n_choices": n_choices,
                "fallback_share": n_fallback / n_choices,
                "angle_to_nearest_used_deg": float(np.mean(angles)),
                "distance_to_nearest_used_mm": float(np.mean(dists)),
                "seconds_per_candidate": float(np.mean(seconds)) if seconds else float("nan"),
            }
    return out


def oracle_analysis(rows: list[dict[str, Any]], outputs: Path, dataset_name: str) -> dict[str, Any]:
    """What the oracle's per-candidate gain follows, after the first View: the number of
    successful single-view hypotheses the candidate image holds (Delta's labelled table), the
    number of hypotheses in it, and the NBV score of the same candidate — within-scene Spearman
    correlations averaged over episodes. Says whether view choice is about geometry (what NBV
    can see) or about what the detector will find in the image (what it cannot)."""
    from scipy.stats import spearmanr

    labelled = outputs / "confidence" / dataset_name / "hypotheses.parquet"
    oracles = {(r["g"], r["budget"]): r for r in rows if r["policy"] == "oracle"}
    nbvs = {(r["g"], r["budget"]): r for r in rows if r["policy"] == "nbv"}
    if not oracles or not labelled.exists():
        return {}
    h = pd.read_parquet(labelled)
    n_ok = h[h.success.astype(bool)].groupby(["scene_id", "image_id"]).size()
    n_all = h.groupby(["scene_id", "image_id"]).size()
    recs = []
    for key, ro in oracles.items():
        rn = nbvs.get(key)
        nbv_scores: dict[tuple[int, int], float] = {}
        if rn is not None:
            for e in rn["episodes"]:
                for c in e["steps"][0]["candidates"]:
                    nbv_scores[(e["scene_id"], c["image_id"])] = c["score"]
        for e in ro["episodes"]:
            s = e["scene_id"]
            for c in e["steps"][0]["candidates"]:
                i = c["image_id"]
                recs.append(
                    {
                        "episode": (key[0], s),
                        "gain": c["score"],
                        "n_ok": int(n_ok.get((s, i), 0)),
                        "n_hyp": int(n_all.get((s, i), 0)),
                        "nbv": nbv_scores.get((s, i), float("nan")),
                    }
                )
    d = pd.DataFrame(recs).drop_duplicates(subset=["episode", "gain", "n_ok", "n_hyp", "nbv"])

    def within(a: str, b: str) -> float:
        rs = []
        for _, g in d.groupby("episode"):
            g = g.dropna(subset=[a, b])
            if g[a].nunique() > 1 and g[b].nunique() > 1:
                rs.append(float(spearmanr(g[a], g[b]).correlation))
        return float(np.nanmean(rs)) if rs else float("nan")

    return {
        "n_episodes": int(d["episode"].nunique()),
        "gain_vs_successful_hypotheses_in_image": within("gain", "n_ok"),
        "gain_vs_hypotheses_in_image": within("gain", "n_hyp"),
        "gain_vs_nbv_score": within("gain", "nbv"),
        "nbv_score_vs_successful_hypotheses": within("nbv", "n_ok"),
    }


def verdict_loop_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not r["verdict_loop"]:
            continue
        s = r["nbv_summary"]
        eps = r["episodes"]
        out.append(
            {
                "name": r["name"],
                "policy": r["policy"],
                "g": r["g"],
                "views_used_mean": s["views_used"]["mean"],
                "views_used_min": s["views_used"]["min"],
                "views_used_max": s["views_used"]["max"],
                "stops": s["stops"],
                "ar": r["report"]["ar"],
                "verdicts_final": s["verdicts"],
                "verdicts_first": {
                    k: int(sum(e["steps"][0]["verdicts"].get(k, 0) for e in eps))
                    for k in ("accept", "reject", "request_view")
                },
            }
        )
    return out


def plot_curve(
    curve: dict[str, dict[str, dict[str, Any]]],
    diffs: dict[str, dict[str, dict[str, Any]]],
    path: Path,
    dataset: str,
) -> None:
    """Left: AR vs Views used per policy (the every-View row as a dashed ceiling line). Right:
    the paired per-episode difference to random next, per budget, with its 95 % interval — the
    scene-to-scene spread on the left hides differences the pairing resolves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    colours = {p: f"C{i}" for i, (p, _) in enumerate(POLICIES)}
    for policy, label in POLICIES:
        pts = {k: v for k, v in curve.get(policy, {}).items() if k not in ("verdict", "b0")}
        if pts:
            order = sorted(pts, key=lambda k: pts[k]["views_used"])
            x = [pts[k]["views_used"] for k in order]
            ar = [100 * pts[k]["ar"] for k in order]
            ax.plot(x, ar, marker="o", label=label, color=colours[policy])
        v = curve.get(policy, {}).get("verdict")
        if v:
            ax.scatter(
                [v["views_used"]],
                [100 * v["ar"]],
                marker="*",
                s=180,
                zorder=5,
                color=colours[policy],
                edgecolor="k",
                label=f"{label}, Verdict-stopped",
            )
        b0 = curve.get(policy, {}).get("b0")
        if b0:
            ax.axhline(100 * b0["ar"], ls="--", color="grey", lw=1)
            ax.annotate(
                f"every View ({b0['views_used']:.0f}): {100 * b0['ar']:.1f}",
                (1.0, 100 * b0["ar"]),
                textcoords="offset points",
                xytext=(2, 3),
                fontsize=8,
                color="grey",
            )
    ax.set_xlabel("Views used per scene (mean)")
    ax.set_ylabel("AR (BOP19 core, %)")
    ax.set_title(f"{dataset}: AR vs Views used")
    ax.set_xlim(0.7, max(ax.get_xlim()[1], 8.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    width = 0.8 / max(1, len(diffs))
    budgets = sorted({k for d in diffs.values() for k in d})
    for j, (policy, d) in enumerate(diffs.items()):
        xs = [i + (j - (len(diffs) - 1) / 2) * width for i, k in enumerate(budgets) if k in d]
        ys = [100 * d[k]["mean_diff"] for k in budgets if k in d]
        lo = [100 * (d[k]["mean_diff"] - d[k]["lo"]) for k in budgets if k in d]
        hi = [100 * (d[k]["hi"] - d[k]["mean_diff"]) for k in budgets if k in d]
        label = dict(POLICIES).get(policy, policy)
        bx.bar(
            xs, ys, width=width, yerr=[lo, hi], capsize=3, color=colours.get(policy), label=label
        )
    bx.axhline(0, color="k", lw=0.8)
    bx.set_xticks(range(len(budgets)))
    bx.set_xticklabels([b.replace("b", "budget ") for b in budgets])
    bx.set_ylabel("Δ AR vs random next (pt, paired per episode)")
    bx.set_title("paired difference to random, 95 % bootstrap")
    bx.grid(alpha=0.3, axis="y")
    bx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def render(
    dataset: str,
    curve: dict[str, dict[str, dict[str, Any]]],
    diff_nr: dict[str, dict[str, Any]],
    diff_nf: dict[str, dict[str, Any]],
    diff_er: dict[str, dict[str, Any]],
    diff_or: dict[str, dict[str, Any]],
    angles: dict[str, dict[str, float]],
    loops: list[dict[str, Any]],
    oracle: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    out = [
        f"# Epsilon results — {dataset}",
        "",
        "Every row: cached CNOS → FoundPose → pt2plane refinement, then the active loop (D14)",
        "— associate → weighted SE(3) mean → Model F Confidence after every unlocked View —",
        "evaluated on the same reference Views per start group (the strided 4-view group of",
        "Gamma / Delta). AR = core evaluator (BOP19 protocol), mean over the start groups of the",
        "row; the interval is a 95 % bootstrap over (scene, start group) episodes of the per-GT",
        "recall approximation.",
        "",
        "## AR vs Views used",
        "",
        "| policy | budget | rows (start groups) | Views used | AR | AR_VSD | AR_MSSD | AR_MSPD "
        "| per-GT AR [95 %] | episodes / GT |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for policy, label in POLICIES:
        for xkey, p in sorted(curve.get(policy, {}).items(), key=lambda kv: kv[1]["views_used"]):
            out.append(
                f"| {label} | {xkey} | {p['n_start_groups']} | {p['views_used']:.2f} | "
                f"**{fmt(p['ar'])}** | {fmt(p['ar_vsd'])} | {fmt(p['ar_mssd'])} | "
                f"{fmt(p['ar_mspd'])} | {fmt(p['boot'])} [{fmt(p['lo'])}, {fmt(p['hi'])}] | "
                f"{p['n_episodes']} / {p['n_gt']} |"
            )
    out += ["", f"![AR vs views]({dataset}_epsilon_curve.png)", ""]
    for title, diff, other in (
        ("NBV − random next", diff_nr, "random"),
        ("NBV − fixed order", diff_nf, "fixed"),
        ("NBV (entropy weight) − random next", diff_er, "random"),
        ("oracle − random next (the ceiling)", diff_or, "random"),
    ):
        if not diff:
            continue
        out += [
            f"## Paired difference: {title}",
            "",
            f"Per episode (scene, start group), the per-GT AR of the first policy minus {other}",
            "on the same",
            "reference Views and GT instances; interval = 95 % bootstrap over episodes; wins =",
            "share of episodes the first policy is ahead.",
            "",
            "| budget | episodes | mean Δ AR (pt) | 95 % | first wins | ties |",
            "|---|---|---|---|---|---|",
        ]
        for xkey, d in sorted(diff.items()):
            out.append(
                f"| {xkey} | {d['n_episodes']} | {100 * d['mean_diff']:+.2f} | "
                f"[{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] | {100 * d['wins']:.0f} % | "
                f"{100 * d['ties']:.0f} % |"
            )
        out.append("")
    if angles:
        out += [
            "## What the policies unlock",
            "",
            "Angle between the chosen View's optical axis and the nearest View already used, and",
            "the distance between the camera centres, averaged over every choice; the NBV score's",
            "cost per candidate View (all tracks, silhouettes at a quarter resolution).",
            "",
            "| policy | choices | fallback | angle to nearest used (°) | centre distance (mm) "
            "| s / candidate |",
            "|---|---|---|---|---|---|",
        ]
        for policy, label in POLICIES:
            a = angles.get(policy)
            if a:
                out.append(
                    f"| {label} | {a['n_choices']} | {100 * a['fallback_share']:.0f} % | "
                    f"{a['angle_to_nearest_used_deg']:.1f} | "
                    f"{a['distance_to_nearest_used_mm']:.0f} | {a['seconds_per_candidate']:.2f} |"
                )
        out.append("")
    if oracle:
        out += [
            "## What the oracle's gain follows",
            "",
            "After the first View, the oracle scores every candidate by the annotated instances a",
            "successful FusedPose would match once that View is added. Within-scene Spearman",
            "correlation of that gain with properties of the candidate image, averaged over",
            f"{oracle['n_episodes']} episodes:",
            "",
            "| correlate | ρ |",
            "|---|---|",
            "| successful single-view hypotheses in the image (detector + estimator, needs GT) | "
            f"{oracle['gain_vs_successful_hypotheses_in_image']:.2f} |",
            "| hypotheses in the image (needs the image) | "
            f"{oracle['gain_vs_hypotheses_in_image']:.2f} |",
            "| the NBV score of the candidate (geometry only) | "
            f"{oracle['gain_vs_nbv_score']:.2f} |",
            "| NBV score vs successful hypotheses | "
            f"{oracle['nbv_score_vs_successful_hypotheses']:.2f} |",
            "",
        ]
    if loops:
        out += [
            "## The Verdict-stopped loop (D14)",
            "",
            "1 View → fuse → Verdict; while any ObjectTrack says `request_view` unlock the",
            "policy's next View; stop at accept / reject everywhere or exhaustion (published",
            "thresholds, τ_acc 0.823 / τ_rej 0.675).",
            "",
            "| row | policy | g | Views used mean [min, max] | stops | AR "
            "| Verdicts after 1 View (acc / rej / req) | final |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for lp in loops:
            vf, vl = lp["verdicts_first"], lp["verdicts_final"]
            out.append(
                f"| {lp['name']} | {lp['policy']} | {lp['g']} | {lp['views_used_mean']:.2f} "
                f"[{lp['views_used_min']}, {lp['views_used_max']}] | {lp['stops']} | "
                f"{fmt(lp['ar'])} | "
                f"{vf['accept']} / {vf['reject']} / {vf['request_view']} | "
                f"{vl.get('accept', 0)} / {vl.get('reject', 0)} / {vl.get('request_view', 0)} |"
            )
        out.append("")
    out += [
        "## Rows",
        "",
        "| row | policy | budget | g | Views used | AR | evaluate dir |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["name"]):
        out.append(
            f"| {r['name']} | {r['policy']} | {'verdict' if r['verdict_loop'] else r['budget']} | "
            f"{r['g']} | {r['nbv_summary']['views_used']['mean']:.2f} | {fmt(r['report']['ar'])} | "
            f"`{r['evaluate_dir']}` |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    outputs = Path(args.outputs)
    rows = find_rows(args.dataset, outputs)
    if not rows:
        sys.exit(f"no A9 rows for {args.dataset} under {outputs / 'runs'}")
    print(f"{len(rows)} A9 rows")
    curve = curve_points(rows, args.n_boot)
    diff_nr = paired_difference(rows, "nbv", "random", args.n_boot)
    diff_nf = paired_difference(rows, "nbv", "fixed", args.n_boot)
    diff_er = paired_difference(rows, "nbve", "random", args.n_boot)
    diff_or = paired_difference(rows, "oracle", "random", args.n_boot)
    cfg = json.loads(
        (outputs / "runs" / f"{rows[0]['name']}_{args.dataset}" / "run_manifest.json").read_text()
    )["config"]
    dataset = make_dataset(cfg, REPO)
    angles = angle_analysis(rows, dataset)
    loops = verdict_loop_table(rows)
    oracle = oracle_analysis(rows, outputs, args.dataset)
    fig = outputs / f"{args.dataset}_epsilon_curve.png"
    diffs = {
        p: d
        for p, d in (
            ("fixed", paired_difference(rows, "fixed", "random", args.n_boot)),
            ("nbv", diff_nr),
            ("nbve", diff_er),
            ("oracle", diff_or),
        )
        if d
    }
    plot_curve(curve, diffs, fig, args.dataset)
    with open(outputs / f"{args.dataset}_epsilon_curve.json", "w") as f:
        json.dump(
            {
                "curve": curve,
                "nbv_minus_random": diff_nr,
                "nbv_minus_fixed": diff_nf,
                "nbve_minus_random": diff_er,
                "oracle_minus_random": diff_or,
                "choices": angles,
                "verdict_loop": loops,
                "oracle_analysis": oracle,
            },
            f,
            indent=1,
            default=str,
        )
    md = render(
        args.dataset, curve, diff_nr, diff_nf, diff_er, diff_or, angles, loops, oracle, rows
    )
    path = outputs / f"{args.dataset}_epsilon_report.md"
    path.write_text(md)
    print(md)
    print(f"wrote {path} and {fig}")


if __name__ == "__main__":
    main()
