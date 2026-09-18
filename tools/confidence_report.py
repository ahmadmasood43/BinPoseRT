#!/usr/bin/env python
"""Delta analysis and report of one dataset from ``outputs/`` only: the A8 rows (A6 chain +
confidence stage, BOP score = Confidence) against the Gamma mean rows, the Model-H-weighted
variant A8w, the Verdict bands with their measured success rates on the full pipeline output, and
the confident-failure galleries.

    uv run python tools/confidence_report.py --dataset tless --tag v1 --plot outputs/tless_delta

Rows are the runs ``tools/run_delta.sh`` writes (``A8_k<k>``, ``A8w_k<k>``) next to Gamma's
``A6_k<k>_mean``. FusedPoses of the A8 rows are labelled once per row (world-frame MSSD against
the group's ground truth) and cached under ``<confidence stage>/analysis/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from multiview_report import bootstrap_ar, load_run  # noqa: E402

from binposert.confidence import (  # noqa: E402
    VerdictThresholds,
    label_fused,
    risk_coverage,
    summary,
    verdict_rates,
)
from binposert.pipeline.multiview_stage import FUSED_FILE, TRACKS_FILE  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.pipeline.stages import default_workers  # noqa: E402
from binposert.viz.confidence_gallery import make_confidence_galleries  # noqa: E402


def load_delta_run(
    name: str, dataset: str, outputs: Path, tag: str | None = None
) -> dict[str, Any] | None:
    """A row's manifest, evaluate report and stage dirs; a confidence row fitted with another
    tag's model files is refused rather than reported against this tag's thresholds."""
    row = load_run(name, dataset, outputs)
    if row is None:
        return None
    manifest = json.loads(
        (outputs / "runs" / f"{name}_{dataset}" / "run_manifest.json").read_text()
    )
    st = manifest["stages"]
    row["confidence_dir"] = Path(st["confidence"]["dir"]) if "confidence" in st else None
    row["fuse_hash"] = st["fuse"]["hash"] if "fuse" in st else None
    if tag is not None and row["confidence_dir"] is not None:
        model_f = str(manifest["config"].get("confidence", {}).get("params", {}).get("model_f"))
        if f"models/confidence/{tag}/" not in model_f:
            raise SystemExit(
                f"{name} on {dataset} was run with {model_f}, not the {tag} models; re-run the row"
            )
    return row


def paired_delta_ar(
    ev_a: Path, ev_b: Path, exclude_scenes: set[int], n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """AR(b) − AR(a) on the same ground truth, held-out scenes only, with a paired scene-bootstrap
    95 % interval: the per-row intervals overlap even when every scene moves the same way."""
    keys = ["scene_id", "image_id", "object_id", "gt_index"]
    cols = ["ar_vsd", "ar_mssd", "ar_mspd"]
    a = pd.read_parquet(ev_a / "gt_rows.parquet")
    b = pd.read_parquet(ev_b / "gt_rows.parquet")
    a = a[~a.scene_id.isin(exclude_scenes)]
    b = b[~b.scene_id.isin(exclude_scenes)]
    m = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    use = [c for c in cols if m[f"{c}_a"].notna().any() and m[f"{c}_b"].notna().any()]
    m["ar_a"] = m[[f"{c}_a" for c in use]].mean(axis=1)
    m["ar_b"] = m[[f"{c}_b" for c in use]].mean(axis=1)
    per_scene = {
        int(sid): (g.ar_a.to_numpy(), g.ar_b.to_numpy()) for sid, g in m.groupby("scene_id")
    }
    scenes = sorted(per_scene)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(scenes, size=len(scenes), replace=True)
        aa = np.concatenate([per_scene[s][0] for s in pick])
        bb = np.concatenate([per_scene[s][1] for s in pick])
        deltas.append(float(bb.mean() - aa.mean()))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_gt": int(len(m)),
        "ar_a": float(m.ar_a.mean()),
        "ar_b": float(m.ar_b.mean()),
        "delta": float(m.ar_b.mean() - m.ar_a.mean()),
        "lo": float(lo),
        "hi": float(hi),
    }


def labelled_fused(row: dict[str, Any], dataset: Any, n_workers: int) -> pd.DataFrame:
    """The row's scored FusedPoses with labels, cached next to the confidence stage."""
    conf_dir = row["confidence_dir"]
    cache = conf_dir / "analysis" / "labelled_fused.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    fused = pd.read_parquet(conf_dir / FUSED_FILE)
    lab = label_fused(fused, dataset, n_model_points=0, n_workers=n_workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(cache, index=False)
    return lab


def failure_anatomy(lab: pd.DataFrame, tau_accept: float) -> dict[str, Any]:
    """What the accepted failures are: near misses just over the 0.1 d line, poses on the wrong
    copy or object (translation off by more than half a diameter), or the rest (an orientation the
    silhouette cannot tell apart — a flip or a symmetry the model does not declare)."""
    acc = lab[(lab.confidence >= tau_accept) & (~lab.success.astype(bool))]
    n = int(len(acc))
    if n == 0:
        return {"n": 0}
    e = acc["mssd_mm"] / acc["diameter"]
    near = e < 0.2
    wrong_copy = (~near) & (acc["t_err_mm"] / acc["diameter"] > 0.5)
    rest = ~(near | wrong_copy)
    return {
        "n": n,
        "near_miss": float(near.mean()),
        "wrong_copy": float(wrong_copy.mean()),
        "orientation": float(rest.mean()),
        "median_mssd_d": float(e.median()),
    }


def fmt(x: Any, pct: bool = True, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * float(x):.{digits}f}" if pct else f"{float(x):.3f}"


def render(
    dataset: str,
    tag: str,
    ks: list[int],
    rows: dict[str, dict[str, Any]],
    bands: dict[str, dict[str, Any]],
    calib: dict[str, dict[str, Any]],
    fit_scenes: list[int],
    galleries: dict[str, str],
    thresholds: VerdictThresholds | None,
    anatomy: dict[str, dict[str, Any]] | None = None,
    paired: dict[str, dict[str, float]] | None = None,
) -> str:
    out = [
        f"# Delta on {dataset} — confidence stage (models `{tag}`)",
        "",
        "Rows: `A6_k<k>_mean` (Gamma: product fusion weights, BOP score = FoundPose pose_score), "
        "`A8_k<k>` (the same fused poses, BOP score = Confidence from Model F), `A8w_k<k>` "
        "(Model H as the fusion weight, BOP score = Confidence). Core AR with a scene-bootstrap "
        "95 % interval; official = bop_toolkit on the group images.",
        "",
        "## AR by view count",
        "",
        "| k | row | n pred | AR core | 95 % CI | AR official | Δ vs A6 (core) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ks:
        base = rows.get(f"A6_k{k}_mean")
        for name in (f"A6_k{k}_mean", f"A8_k{k}", f"A8w_k{k}"):
            r = rows.get(name)
            if r is None:
                continue
            rep = r["report"]
            off = r["official"]["bop19_average_recall"] if r["official"] else None
            d = rep["ar"] - base["report"]["ar"] if base and name != f"A6_k{k}_mean" else None
            out.append(
                f"| {k} | `{name}` | {rep['n_predictions']} | {fmt(rep['ar'])} | "
                f"[{fmt(r['ci'][1])}, {fmt(r['ci'][2])}] | {fmt(off)} | "
                f"{'—' if d is None else f'{100 * d:+.1f}'} |"
            )
    if paired:
        out += [
            "",
            "## Confidence ranking, held-out scenes only (paired)",
            "",
            f"AR of the row minus AR of `A6_k<k>_mean` on the same ground truth, val scenes "
            f"{fit_scenes} excluded, with a paired scene-bootstrap 95 % interval (core "
            "evaluator, mean of the per-GT recall fractions).",
            "",
            "| row | n GT | AR A6 mean | AR row | Δ | 95 % CI of Δ |",
            "|---|---|---|---|---|---|",
        ]
        for name, d in paired.items():
            out.append(
                f"| `{name}` | {d['n_gt']} | {fmt(d['ar_a'])} | {fmt(d['ar_b'])} | "
                f"{100 * d['delta']:+.1f} | [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] |"
            )
    if thresholds is not None:
        out += [
            "",
            "## Verdict bands on the full pipeline output",
            "",
            f"τ_acc = {thresholds.tau_accept:.3f}, τ_rej = {thresholds.tau_reject:.3f} (chosen on "
            f"the val scenes {fit_scenes}; `held-out` rows exclude them). Success = MSSD < 0.1 d "
            "of the FusedPose against the nearest ground truth of its group.",
            "",
            "| row | rows | n | success % | accept % | accept precision % | reject % | "
            "reject precision % | request_view % | request_view success % | ROC-AUC | Brier | "
            "ECE % | AURC % |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, b in bands.items():
            for subset, r in b.items():
                c = calib[name][subset]
                out.append(
                    f"| `{name}` | {subset} | {r['n']} | {fmt(c['base_rate'])} | "
                    f"{fmt(r['accept_rate'])} | {fmt(r['accept_precision'])} | "
                    f"{fmt(r['reject_rate'])} | {fmt(r['reject_precision'])} | "
                    f"{fmt(r['request_view_rate'])} | "
                    f"{fmt(r['request_view_success_rate'])} | {fmt(c['roc_auc'])} | "
                    f"{fmt(c['brier'], False)} | {fmt(c['ece'])} | {fmt(c['aurc'])} |"
                )
    if anatomy:
        out += [
            "",
            "## Accepted failures (held-out): what they are",
            "",
            "near miss = MSSD in (0.1, 0.2) d; wrong copy = translation off by > 0.5 d (another "
            "copy or a look-alike object under the same label); orientation = the rest (a flip "
            "the silhouette cannot see).",
            "",
            "| row | accepted failures | near miss % | wrong copy % | orientation % | "
            "median MSSD / d |",
            "|---|---|---|---|---|---|",
        ]
        for name, a in anatomy.items():
            if a["n"]:
                out.append(
                    f"| `{name}` | {a['n']} | {fmt(a['near_miss'])} | {fmt(a['wrong_copy'])} | "
                    f"{fmt(a['orientation'])} | {a['median_mssd_d']:.2f} |"
                )
    if galleries:
        out += ["", "## Galleries", ""]
        for name, p in galleries.items():
            out.append(f"- {name}: `{p}`")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--plot", type=Path, default=None, help="prefix for figures and galleries")
    ap.add_argument("--gallery-ks", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--n-workers", type=int, default=0)
    args = ap.parse_args()
    n_workers = args.n_workers or default_workers()

    fit_report = args.outputs / "confidence" / args.tag / "report.json"
    fit_scenes: list[int] = []
    if fit_report.exists():
        fit_scenes = json.loads(fit_report.read_text())["fit_scenes"].get(args.dataset, [])
    thr_path = REPO / "models" / "confidence" / args.tag / "thresholds.json"
    thresholds = VerdictThresholds.load(thr_path) if thr_path.exists() else None

    rows: dict[str, dict[str, Any]] = {}
    for k in args.ks:
        for name in (f"A6_k{k}_mean", f"A8_k{k}", f"A8w_k{k}"):
            r = load_delta_run(name, args.dataset, args.outputs, args.tag)
            if r is not None:
                r["ci"] = bootstrap_ar(Path(r["evaluate_dir_abs"]))
                rows[name] = r
    paired: dict[str, dict[str, float]] = {}
    for k in args.ks:
        base = rows.get(f"A6_k{k}_mean")
        if base is None:
            continue
        for name in (f"A8_k{k}", f"A8w_k{k}"):
            if name in rows:
                paired[name] = paired_delta_ar(
                    Path(base["evaluate_dir_abs"]),
                    Path(rows[name]["evaluate_dir_abs"]),
                    set(fit_scenes),
                )
    dataset = None
    bands: dict[str, dict[str, Any]] = {}
    calib: dict[str, dict[str, Any]] = {}
    galleries: dict[str, str] = {}
    anatomy: dict[str, dict[str, Any]] = {}
    for name, r in rows.items():
        if r["confidence_dir"] is None or thresholds is None:
            continue
        if dataset is None:
            dataset = make_dataset(r["config"], REPO)
        lab = labelled_fused(r, dataset, n_workers)
        y, p = lab["success"].to_numpy(bool), lab["confidence"].to_numpy(float)
        held = ~lab["scene_id"].isin(fit_scenes).to_numpy()
        bands[name], calib[name] = {}, {}
        for subset, sel in (("all", np.ones(len(lab), bool)), ("held-out", held)):
            if sel.sum() == 0:
                continue
            bands[name][subset] = verdict_rates(y[sel], p[sel], thresholds)
            calib[name][subset] = summary(y[sel], p[sel])
        anatomy[name] = failure_anatomy(lab[held], thresholds.tau_accept)
        if args.plot is not None and name in {f"A8_k{k}" for k in args.gallery_ks}:
            tracks = pd.read_parquet(Path(r["associate_dir"]) / TRACKS_FILE)
            scored = lab[held].copy()
            written = make_confidence_galleries(
                dataset, scored, tracks, args.plot.parent / f"{args.plot.name}_gallery_{name}"
            )
            galleries.update({f"{name} {k}": v for k, v in written.items()})
            rc = risk_coverage(y[held], p[held])
            (args.plot.parent / f"{args.plot.name}_risk_coverage_{name}.json").write_text(
                json.dumps({"coverage": rc.coverage.tolist(), "risk": rc.risk.tolist()})
            )
    print(
        render(
            args.dataset,
            args.tag,
            args.ks,
            rows,
            bands,
            calib,
            fit_scenes,
            galleries,
            thresholds,
            anatomy,
            paired,
        )
    )


if __name__ == "__main__":
    main()
