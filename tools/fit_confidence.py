#!/usr/bin/env python
"""Fit and evaluate the ConfidenceModels (Delta, D11) on the labelled tables of
``tools/build_confidence_table.py``; write the fitted model files, the calibration analysis and
the figures.

    uv run python tools/fit_confidence.py --datasets tless xyzibd \\
        --fit-scenes tless=1,6,11,16 xyzibd=0,20,40,60 --tag v1

Split discipline: the listed scenes of every dataset are the *fit* (val) rows — Model H, Model F,
the regularisation strength, the recalibration and the Verdict thresholds are chosen on them and
nothing else; every number reported comes from the other scenes (*eval*), per dataset and pooled.
Extra rows fit on one dataset alone and evaluate on the other (cross-dataset).

In-sample optimism is handled with scene-grouped out-of-fold (OOF) predictions of the fit rows:
each model's logit is recalibrated (Platt) on its OOF logits, Model F is fitted on the OOF Model H
aggregates of its members (so it sees Model H as the eval rows will), and the Verdict thresholds
are chosen on the OOF, recalibrated Model F probabilities. Writes ``models/confidence/<tag>/``
(model_h.json, model_f.json, thresholds.json, card.md) and ``outputs/confidence/<tag>/``
(report.json, report.md, figures).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.confidence import (  # noqa: E402
    ConfidenceModel,
    FeatureSchema,
    choose_thresholds,
    platt_scaling,
    reliability_table,
    risk_coverage,
    summary,
    verdict_rates,
)
from binposert.confidence.model import _sigmoid  # noqa: E402
from binposert.confidence.schema import (  # noqa: E402
    FUSED_EXTRAS,
    HYPOTHESIS_EXTRAS,
    SCHEMA_VERSION,
    SIGNALS,
)
from binposert.pipeline.confidence_stage import P_H_COLUMNS, TRACK_KEYS  # noqa: E402

H_KEYS = ["scene_id", "image_id", "object_id", "detection_id", "hypothesis_id"]
F_KEYS = ["row", *TRACK_KEYS]
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
METRIC_COLUMNS = [
    "n",
    "base_rate",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "ece_quantile",
    "aurc",
    "e_aurc",
    "coverage_at_risk_05",
    "coverage_at_risk_10",
]


# ----------------------------------------------------------------------------- data


def load_tables(outputs: Path, datasets: list[str], fit_scenes: dict[str, set[int]]) -> dict:
    hyps, fused, members = [], [], []
    for ds in datasets:
        d = outputs / "confidence" / ds
        h = pd.read_parquet(d / "hypotheses.parquet")
        f = pd.read_parquet(d / "fused.parquet")
        m = pd.read_parquet(d / "members.parquet")
        m.insert(0, "dataset", ds)
        for t in (h, f):
            t["role"] = np.where(t["scene_id"].isin(fit_scenes.get(ds, set())), "fit", "eval")
        hyps.append(h)
        fused.append(f)
        members.append(m)
    return {
        "hypotheses": pd.concat(hyps, ignore_index=True),
        "fused": pd.concat(fused, ignore_index=True),
        "members": pd.concat(members, ignore_index=True),
    }


def fused_with_extras(
    fused: pd.DataFrame, members: pd.DataFrame, hyps: pd.DataFrame, p_h: np.ndarray
) -> pd.DataFrame:
    """Join the Model H aggregates of every track's members onto the fused rows."""
    h = hyps[["dataset", *H_KEYS]].copy()
    h["p_h"] = p_h
    m = members.merge(h, on=["dataset", *H_KEYS], how="left", validate="many_to_one")
    if m["p_h"].isna().any():
        raise SystemExit("members.parquet names hypotheses that are not in hypotheses.parquet")
    m["rejected"] = m["rejection_reason"].notna().astype(float)
    g = m.groupby(["dataset", *F_KEYS])
    agg = pd.DataFrame(
        {
            "p_h_mean": g["p_h"].mean(),
            "p_h_min": g["p_h"].min(),
            "p_h_max": g["p_h"].max(),
            "rejected": g["rejected"].mean(),
        }
    ).reset_index()
    out = fused.drop(columns=[c for c in P_H_COLUMNS if c in fused]).merge(
        agg, on=["dataset", *F_KEYS], how="left", validate="one_to_one"
    )
    if out[P_H_COLUMNS].isna().any(axis=1).any():
        raise SystemExit("some FusedPoses have no members")
    out.index = fused.index  # a left merge keeps the row order; keep the labels too
    return out


# ----------------------------------------------------------------------------- fitting


def select_c(schema: FeatureSchema, fit: pd.DataFrame, grid: tuple[float, ...]) -> dict[str, Any]:
    """Scene-grouped 5-fold cross-validation on the fit rows: mean log-loss per C."""
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold

    groups = fit["dataset"].astype(str) + "/" + fit["scene_id"].astype(str)
    y = fit["success"].to_numpy(dtype=bool)
    n_splits = min(5, groups.nunique())
    losses: dict[float, list[float]] = {c: [] for c in grid}
    for tr, te in GroupKFold(n_splits=n_splits).split(fit, y, groups):
        for c in grid:
            m = ConfidenceModel.fit(schema, fit.iloc[tr], y[tr], C=c)
            p = np.clip(m.predict_proba(fit.iloc[te]), 1e-6, 1 - 1e-6)
            losses[c].append(float(log_loss(y[te], p)))
    table = {str(c): float(np.mean(v)) for c, v in losses.items()}
    best = min(grid, key=lambda c: table[str(c)])
    return {"grid": table, "best": float(best), "n_splits": n_splits}


def scene_groups(table: pd.DataFrame) -> pd.Series:
    return table["dataset"].astype(str) + "/" + table["scene_id"].astype(str)


def oof_logits(
    schema: FeatureSchema,
    fit: pd.DataFrame,
    y: np.ndarray,
    C: float,
    n_splits: int = 5,
    kind: str = "logistic",
) -> np.ndarray:
    """Raw logit of every fit row from a model that did not see the row's scene."""
    from sklearn.model_selection import GroupKFold

    groups = scene_groups(fit)
    out = np.zeros(len(fit), dtype=np.float64)
    for tr, te in GroupKFold(n_splits=min(n_splits, groups.nunique())).split(fit, y, groups):
        m = ConfidenceModel.fit(schema, fit.iloc[tr], y[tr], C=C, kind=kind)
        out[te] = m.raw_logits(fit.iloc[te])
    return out


def loso_analysis(
    tables: dict[str, pd.DataFrame], C_h: float, C_f: float, accept: float, reject: float
) -> dict[str, Any]:
    """Diagnostic, not the published model: leave-one-scene-out over *all* scenes of every
    dataset, every prediction out-of-fold, Platt fitted on the pooled OOF logits. The metric rows
    are honest out-of-fold numbers. Thresholds are *not* chosen and scored on the same rows: the
    scenes are split into two halves (alternating in scene order, per dataset), thresholds chosen
    on the OOF probabilities of one half are scored on the other, both ways, so the reported
    band precisions are out of sample for the threshold choice too. The per-scene Model H AUC is
    kept for the scene-difficulty table."""
    hyps, fused, members = tables["hypotheses"], tables["fused"], tables["members"]
    schema_h, schema_f = FeatureSchema.hypothesis(), FeatureSchema.fused()
    y_h = hyps["success"].to_numpy(dtype=bool)
    z_h = oof_logits(schema_h, hyps, y_h, C_h, n_splits=10**6)
    a, b = platt_scaling(z_h, y_h)
    p_h = _sigmoid(a * z_h + b)
    fused_x = fused_with_extras(fused, members, hyps, p_h)
    y_f = fused_x["success"].to_numpy(dtype=bool)
    z_f = oof_logits(schema_f, fused_x, y_f, C_f, n_splits=10**6)
    a_f, b_f = platt_scaling(z_f, y_f)
    p_f = _sigmoid(a_f * z_f + b_f)
    out: dict[str, Any] = {"h": {"pooled": summary(y_h, p_h)}, "f": {"pooled": summary(y_f, p_f)}}
    for ds in hyps["dataset"].unique():
        sel_h = (hyps["dataset"] == ds).to_numpy()
        sel_f = (fused_x["dataset"] == ds).to_numpy()
        out["h"][f"dataset={ds}"] = summary(y_h[sel_h], p_h[sel_h])
        out["f"][f"dataset={ds}"] = summary(y_f[sel_f], p_f[sel_f])
    for k in sorted(fused_x["k"].unique()):
        sel_f = (fused_x["k"] == k).to_numpy()
        out["f"][f"k={k}"] = summary(y_f[sel_f], p_f[sel_f])
    # per-scene difficulty (Model H, one fold per scene)
    per_scene = []
    for (ds, sid), idx in hyps.groupby(["dataset", "scene_id"]).indices.items():
        per_scene.append(
            {
                "dataset": str(ds),
                "scene_id": int(sid),
                "n": int(len(idx)),
                "base_rate": float(y_h[idx].mean()),
                "roc_auc": summary(y_h[idx], p_h[idx])["roc_auc"],
                "brier": float(np.mean((p_h[idx] - y_h[idx]) ** 2)),
            }
        )
    out["per_scene_h"] = per_scene
    # threshold transfer between two halves of the scenes
    halves: dict[str, int] = {}
    for ds in sorted(hyps["dataset"].unique()):
        scenes = sorted(int(x) for x in hyps.loc[hyps["dataset"] == ds, "scene_id"].unique())
        for i, sid in enumerate(scenes):
            halves[f"{ds}/{sid}"] = i % 2
    half_f = np.asarray([halves[k] for k in scene_groups(fused_x)])
    out["threshold_transfer"] = {}
    for chosen_on, scored_on in ((0, 1), (1, 0)):
        thr = choose_thresholds(y_f[half_f == chosen_on], p_f[half_f == chosen_on], accept, reject)
        rates = verdict_rates(y_f[half_f == scored_on], p_f[half_f == scored_on], thr)
        entry: dict[str, Any] = {"thresholds": thr.to_dict(), "pooled": rates}
        for ds in sorted(fused_x["dataset"].unique()):
            sel = (half_f == scored_on) & (fused_x["dataset"] == ds).to_numpy()
            entry[f"dataset={ds}"] = verdict_rates(y_f[sel], p_f[sel], thr)
        out["threshold_transfer"][f"half {chosen_on} → half {scored_on}"] = entry
    return out


def evaluate_sets(
    model: ConfidenceModel, table: pd.DataFrame, by: list[str]
) -> dict[str, dict[str, Any]]:
    """Calibration summary on ``table`` (already restricted to eval rows), pooled and per group."""
    out: dict[str, dict[str, Any]] = {}
    if len(table) == 0:
        return out
    p = model.predict_proba(table)
    y = table["success"].to_numpy(dtype=bool)
    out["pooled"] = summary(y, p)
    for col in by:
        for key, idx in table.groupby(col).indices.items():
            if len(idx) >= 20 and len(set(y[idx])) == 2:
                out[f"{col}={key}"] = summary(y[idx], p[idx])
    return out


def ablate_signals(
    tables: dict[str, pd.DataFrame],
    fit_mask_h: pd.Series,
    fit_mask_f: pd.Series,
    C_h: float,
    C_f: float,
    full_h: dict[str, Any],
    full_f: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refit without one signal, under the published protocol (fit on the fit rows, Platt on
    out-of-fold logits), and score the held-out rows. The signal is removed from *both* models:
    Model F reads its members through Model H's probabilities, so an ablation that only dropped
    the signal from Model F's own features would let it flow in through ``p_h_*``."""
    hyps, fused = tables["hypotheses"], tables["fused"]
    ev_h_mask = (~fit_mask_h) & (hyps["role"] == "eval")
    ev_f_mask = (~fit_mask_f) & (fused["role"] == "eval")
    y_ev_h = hyps.loc[ev_h_mask, "success"].to_numpy(dtype=bool)
    y_ev_f = fused.loc[ev_f_mask, "success"].to_numpy(dtype=bool)
    rows_h, rows_f = [], []
    for sig in SIGNALS:
        keep = tuple(x for x in SIGNALS if x != sig)
        schema_h = FeatureSchema("hypothesis", extras=HYPOTHESIS_EXTRAS, signals=keep)
        schema_f = FeatureSchema("fused", extras=FUSED_EXTRAS, signals=keep)
        pair = fit_pair(
            tables, fit_mask_h, fit_mask_f, C_h, C_f, f"without {sig}", schema_h, schema_f
        )
        p_h = pair["model_h"].predict_proba(hyps[ev_h_mask])
        fx = pair["fused"]
        p_f = pair["model_f"].predict_proba(fx[ev_f_mask.to_numpy()])
        rows_h.append(_ablation_row(sig, summary(y_ev_h, p_h), full_h))
        rows_f.append(_ablation_row(sig, summary(y_ev_f, p_f), full_f))

    def key(r: dict[str, Any]) -> float:
        return float(r["delta_roc_auc"])

    return sorted(rows_h, key=key), sorted(rows_f, key=key)


def _ablation_row(sig: str, sm: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal": sig,
        "roc_auc": sm["roc_auc"],
        "delta_roc_auc": sm["roc_auc"] - full["roc_auc"],
        "brier": sm["brier"],
        "delta_brier": sm["brier"] - full["brier"],
        "aurc": sm["aurc"],
        "delta_aurc": sm["aurc"] - full["aurc"],
    }


def fit_pair(
    tables: dict[str, pd.DataFrame],
    fit_mask_h: pd.Series,
    fit_mask_f: pd.Series,
    C_h: float | None,
    C_f: float | None,
    label: str,
    schema_h: FeatureSchema | None = None,
    schema_f: FeatureSchema | None = None,
    weight_source: str = "",
) -> dict[str, Any]:
    """Model H on the masked hypotheses (recalibrated on its OOF logits), its OOF aggregates onto
    the fused fit rows and its final aggregates onto the rest, Model F on the masked fused rows
    (recalibrated the same way). Returns the models (raw and recalibrated), the fused table with
    extras, the OOF recalibrated Model F probabilities of the fit rows and the fit metadata."""
    hyps, fused, members = tables["hypotheses"], tables["fused"], tables["members"]
    schema_h = schema_h or FeatureSchema.hypothesis()
    schema_f = schema_f or FeatureSchema.fused()
    fit_h = hyps[fit_mask_h]
    y_h = fit_h["success"].to_numpy(dtype=bool)
    c_sel_h = select_c(schema_h, fit_h, C_GRID) if C_h is None else {"best": C_h}
    prov = {"fit": label, "fit_scenes": scene_list(fit_h)}
    if weight_source:
        prov["weight_source"] = weight_source
    raw_h = ConfidenceModel.fit(
        schema_h, fit_h, y_h, C=c_sel_h["best"], provenance={**prov, "model": "H"}
    )
    z_oof_h = oof_logits(schema_h, fit_h, y_h, c_sel_h["best"])
    model_h = raw_h.recalibrated(*platt_scaling(z_oof_h, y_h))
    # Model H probabilities: OOF (recalibrated) on the fit rows, the final model elsewhere
    p_h_all = model_h.predict_proba(hyps)
    a, b = model_h.calibration
    p_h_all[fit_mask_h.to_numpy()] = _sigmoid(a * z_oof_h + b)
    fused_x = fused_with_extras(fused, members, hyps, p_h_all)
    fit_f = fused_x[fit_mask_f]
    y_f = fit_f["success"].to_numpy(dtype=bool)
    c_sel_f = select_c(schema_f, fit_f, C_GRID) if C_f is None else {"best": C_f}
    raw_f = ConfidenceModel.fit(
        schema_f, fit_f, y_f, C=c_sel_f["best"], provenance={**prov, "model": "F"}
    )
    z_oof_f = oof_logits(schema_f, fit_f, y_f, c_sel_f["best"])
    model_f = raw_f.recalibrated(*platt_scaling(z_oof_f, y_f))
    a_f, b_f = model_f.calibration
    return {
        "model_h": model_h,
        "model_f": model_f,
        "raw_h": raw_h,
        "raw_f": raw_f,
        "fused": fused_x,
        "p_h": p_h_all,
        "p_oof_f": _sigmoid(a_f * z_oof_f + b_f),
        "y_fit_f": y_f,
        "c_h": c_sel_h,
        "c_f": c_sel_f,
        "n_fit_h": int(len(fit_h)),
        "n_fit_f": int(len(fit_f)),
    }


def scene_list(table: pd.DataFrame) -> dict[str, list[int]]:
    return {
        str(ds): sorted(int(s) for s in g["scene_id"].unique())
        for ds, g in table.groupby("dataset")
    }


# ----------------------------------------------------------------------------- figures


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 4), squeeze=False)
    for ax, (name, (y, p)) in zip(axes[0], curves.items(), strict=True):
        bins = reliability_table(y, p, 10)
        xs = [b.confidence for b in bins if b.n]
        ys = [b.accuracy for b in bins if b.n]
        ns = [b.n for b in bins if b.n]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(xs, ys, "o-", label=f"model (ECE {100 * _ece(bins):.1f} %)")
        for x, yv, n in zip(xs, ys, ns, strict=True):
            ax.annotate(
                str(n), (x, yv), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center"
            )
        ax2 = ax.twinx()
        ax2.hist(p, bins=np.linspace(0, 1, 11), alpha=0.15, color="grey")
        ax2.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("observed success rate")
        ax.set_title(f"{name} (n = {len(y)})")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _ece(bins: list[Any]) -> float:
    total = sum(b.n for b in bins)
    return sum(b.n / total * abs(b.accuracy - b.confidence) for b in bins if b.n) if total else 0.0


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (y, p) in curves.items():
        rc = risk_coverage(y, p)
        ax.plot(rc.coverage, rc.risk, label=f"{name} (AURC {100 * rc.aurc:.1f})")
        ax.axhline(1 - y.mean(), color="grey", lw=0.5, ls=":")
    ax.set_xlabel("coverage (share of poses accepted, most confident first)")
    ax.set_ylabel("risk (failure rate among accepted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- report


def metric_row(name: str, s: dict[str, Any]) -> str:
    def f(k: str, pct: bool = True) -> str:
        v = s.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{100 * v:.1f}" if pct else f"{v:.3f}"

    return (
        f"| {name} | {s['n']} | {f('base_rate')} | {f('roc_auc')} | {f('pr_auc')} | "
        f"{f('brier', False)} | {f('ece')} | {f('ece_quantile')} | {f('aurc')} | {f('e_aurc')} | "
        f"{f('coverage_at_risk_05')} | {f('coverage_at_risk_10')} |"
    )


METRIC_HEADER = (
    "| rows | n | success % | ROC-AUC | PR-AUC | Brier | ECE % | ECE-q % | AURC % | E-AURC % | "
    "cov @ 5 % risk | cov @ 10 % risk |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def render(rep: dict[str, Any], tag: str) -> str:
    out = [f"# Confidence models `{tag}` — calibration analysis", ""]
    out += [
        "Fit rows (val scenes): "
        + ", ".join(f"{ds} {v}" for ds, v in rep["fit_scenes"].items())
        + f"; Model H {rep['n_fit_h']} hypotheses, Model F {rep['n_fit_f']} FusedPoses. "
        "Every metric below is on the other scenes (eval) unless the row says `fit`.",
        "",
        f"Regularisation (scene-grouped CV on fit rows, log-loss): H C = {rep['c_h']['best']}, "
        f"F C = {rep['c_f']['best']}.",
        "",
        "## Model H — per refined PoseHypothesis",
        "",
        METRIC_HEADER,
    ]
    out += [metric_row(k, v) for k, v in rep["h_eval"].items()]
    out += [metric_row(f"fit ({k})", v) for k, v in rep["h_fit"].items() if k == "pooled"]
    out += ["", "## Model F — per FusedPose", "", METRIC_HEADER]
    out += [metric_row(k, v) for k, v in rep["f_eval"].items()]
    out += [metric_row(f"fit ({k})", v) for k, v in rep["f_fit"].items() if k == "pooled"]
    out += [
        "",
        "## Cross-dataset (fit on one, evaluate on all rows of the other)",
        "",
        METRIC_HEADER,
    ]
    for name, cross in rep["cross"].items():
        out += [metric_row(f"H {name}", cross["h"]), metric_row(f"F {name}", cross["f"])]
    cal = rep["calibration"]
    out += [
        "",
        "## Recalibration (Platt on out-of-fold logits of the fit rows)",
        "",
        f"H: logit → {cal['h'][0]:.3f}·logit {cal['h'][1]:+.3f}; "
        f"F: logit → {cal['f'][0]:.3f}·logit {cal['f'][1]:+.3f}.",
        "",
        METRIC_HEADER,
        metric_row("H raw (in-sample fit)", cal["h_raw_eval"]),
        metric_row("H recalibrated (published)", rep["h_eval"]["pooled"]),
        metric_row("F raw (in-sample fit)", cal["f_raw_eval"]),
        metric_row("F recalibrated (published)", rep["f_eval"]["pooled"]),
        metric_row("F OOF on fit rows (thresholds chosen here)", cal["f_oof_fit"]),
    ]
    mlp = rep["mlp"]
    out += [
        "",
        "## MLP comparator (D11: only if logistic under-fits)",
        "",
        METRIC_HEADER,
        metric_row("H logistic", rep["h_eval"]["pooled"]),
        metric_row("H MLP", mlp["h"]),
        metric_row("F logistic", rep["f_eval"]["pooled"]),
        metric_row("F MLP", mlp["f"]),
        "",
        f"Decision: {mlp['decision']}",
        "",
        "## Verdict thresholds (chosen on fit rows of Model F)",
        "",
    ]
    thr = rep["thresholds"]
    out += [
        f"τ_acc = {thr['tau_accept']:.3f} (target accept precision {thr['accept_precision']:.2f}"
        f"{'' if thr['accept_satisfiable'] else ', NOT satisfiable'}), "
        f"τ_rej = {thr['tau_reject']:.3f} (target reject precision {thr['reject_precision']:.2f}"
        f"{'' if thr['reject_satisfiable'] else ', NOT satisfiable'}).",
        "",
        "| rows | n | accept % | reject % | request_view % | accept precision % | "
        "reject precision % | request_view success % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rep["verdicts"].items():
        out.append(
            f"| {name} | {r['n']} | {100 * r['accept_rate']:.1f} | {100 * r['reject_rate']:.1f} | "
            f"{100 * r['request_view_rate']:.1f} | {_pct(r['accept_precision'])} | "
            f"{_pct(r['reject_precision'])} | {_pct(r['request_view_success_rate'])} |"
        )
    loso = rep["loso"]
    out += [
        "",
        "## Diagnostic: leave-one-scene-out over every scene (not the published model)",
        "",
        "Every prediction is from a model that never saw its scene; Platt is fitted on the "
        "pooled out-of-fold logits (two parameters). Thresholds are chosen on the out-of-fold "
        "probabilities of one half of the scenes (alternating in scene order, per dataset) and "
        "scored on the other half, both ways — the band precisions below are out of sample for "
        "the threshold choice as well.",
        "",
        METRIC_HEADER,
    ]
    out += [metric_row(f"LOSO H {k}", v) for k, v in loso["h"].items()]
    out += [metric_row(f"LOSO F {k}", v) for k, v in loso["f"].items()]
    out += [
        "",
        "| thresholds | scored on | n | accept % | reject % | request_view % | "
        "accept precision % | reject precision % | request_view success % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in loso["threshold_transfer"].items():
        t = entry["thresholds"]
        for sub, r in entry.items():
            if sub == "thresholds":
                continue
            out.append(
                f"| {name} (τ_acc {t['tau_accept']:.3f}, τ_rej {t['tau_reject']:.3f}) | {sub} | "
                f"{r['n']} | {100 * r['accept_rate']:.1f} | {100 * r['reject_rate']:.1f} | "
                f"{100 * r['request_view_rate']:.1f} | {_pct(r['accept_precision'])} | "
                f"{_pct(r['reject_precision'])} | {_pct(r['request_view_success_rate'])} |"
            )
    out += [
        "",
        "Per-scene Model H (leave-one-scene-out); fit scenes marked with *:",
        "",
        "| dataset | scene | n | success % | ROC-AUC | Brier |",
        "|---|---|---|---|---|---|",
    ]
    fit_sc = {ds: set(v) for ds, v in rep["fit_scenes"].items()}
    for r in sorted(loso["per_scene_h"], key=lambda r: (r["dataset"], r["scene_id"])):
        star = "*" if r["scene_id"] in fit_sc.get(r["dataset"], set()) else ""
        out.append(
            f"| {r['dataset']} | {r['scene_id']}{star} | {r['n']} | {100 * r['base_rate']:.1f} | "
            f"{_pct(r['roc_auc'])} | {r['brier']:.3f} |"
        )
    out += ["", "## Feature importance (standardised logistic coefficients)", ""]
    out += ["| Model H feature | coef | Model F feature | coef |", "|---|---|---|---|"]
    ch = sorted(rep["coefficients_h"].items(), key=lambda kv: -abs(kv[1]))
    cf = sorted(rep["coefficients_f"].items(), key=lambda kv: -abs(kv[1]))
    for i in range(max(len(ch), len(cf))):
        a = f"`{ch[i][0]}` | {ch[i][1]:+.2f}" if i < len(ch) else " | "
        b = f"`{cf[i][0]}` | {cf[i][1]:+.2f}" if i < len(cf) else " | "
        out.append(f"| {a} | {b} |")
    out += [
        "",
        "## Ablate one signal (refit without it on fit rows; eval rows, pooled)",
        "",
        "| signal | H ΔROC-AUC | H ΔBrier | H ΔAURC | F ΔROC-AUC | F ΔBrier | F ΔAURC |",
        "|---|---|---|---|---|---|---|",
    ]
    ab_f = {r["signal"]: r for r in rep["ablation_f"]}
    for r in rep["ablation_h"]:
        q = ab_f[r["signal"]]
        out.append(
            f"| `{r['signal']}` | {100 * r['delta_roc_auc']:+.2f} | {r['delta_brier']:+.4f} | "
            f"{100 * r['delta_aurc']:+.2f} | {100 * q['delta_roc_auc']:+.2f} | "
            f"{q['delta_brier']:+.4f} | {100 * q['delta_aurc']:+.2f} |"
        )
    out += ["", "## Figures", ""]
    out += [f"![{name}]({Path(p).name})" for name, p in rep["figures"].items()]
    return "\n".join(out) + "\n"


def _pct(v: Any) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}"


def card(rep: dict[str, Any], tag: str, model_h: ConfidenceModel, model_f: ConfidenceModel) -> str:
    h, f = rep["h_eval"]["pooled"], rep["f_eval"]["pooled"]
    thr = rep["thresholds"]
    return "\n".join(
        [
            f"# Confidence models `{tag}`",
            "",
            f"Feature schema v{SCHEMA_VERSION} (QualitySignals v1); logistic regressions, "
            "JSON weights.",
            f"Fitted {rep['date']} on val scenes "
            + ", ".join(f"{d} {v}" for d, v in rep["fit_scenes"].items())
            + ".",
            "",
            f"- **Model H** (per refined PoseHypothesis): {rep['n_fit_h']} fit rows, "
            f"C = {rep['c_h']['best']}; eval ROC-AUC {100 * h['roc_auc']:.1f}, "
            f"Brier {h['brier']:.3f}, ECE {100 * h['ece']:.1f} %.",
            f"- **Model F** (per FusedPose): {rep['n_fit_f']} fit rows, C = {rep['c_f']['best']}; "
            f"eval ROC-AUC {100 * f['roc_auc']:.1f}, Brier {f['brier']:.3f}, "
            f"ECE {100 * f['ece']:.1f} %.",
            f"- **Verdict**: accept ≥ {thr['tau_accept']:.3f}, reject ≤ {thr['tau_reject']:.3f}"
            f" (targets {thr['accept_precision']:.2f} / {thr['reject_precision']:.2f}).",
            "",
            "Success = MSSD < 0.1 · diameter against the nearest same-object ground truth. Signals"
            " come from CNOS + FoundPose + point-to-plane ICP on T-LESS test_primesense and XYZ-IBD"
            " val; a different segmenter, estimator or refiner changes the signal distributions and"
            " needs a refit.",
            "",
            "Files: model_h.json, model_f.json, thresholds.json; analysis in "
            "outputs/confidence/<tag>/report.md.",
            "",
        ]
    )


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datasets", nargs="+", default=["tless", "xyzibd"])
    ap.add_argument(
        "--fit-scenes",
        nargs="+",
        default=["tless=1,6,11,16", "xyzibd=0,20,40,60"],
        help="<dataset>=<scene ids> whose rows are the val (fit) rows",
    )
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--models-root", type=Path, default=REPO / "models" / "confidence")
    ap.add_argument(
        "--weight-source",
        default="product",
        help="fusion weighting of the tables' tracks, recorded in the models' provenance so the "
        "confidence stage can warn when a row was associated with another weighting; pass '' "
        "to omit the field (the v1 models were published without it and their fingerprints "
        "must not change)",
    )
    ap.add_argument("--accept-precision", type=float, default=0.95)
    ap.add_argument("--reject-precision", type=float, default=0.90)
    args = ap.parse_args()
    t0 = time.perf_counter()
    fit_scenes = {
        spec.split("=")[0]: {int(s) for s in spec.split("=")[1].split(",")}
        for spec in args.fit_scenes
    }
    tables = load_tables(args.outputs, args.datasets, fit_scenes)
    hyps, fused = tables["hypotheses"], tables["fused"]
    print(
        f"loaded {len(hyps)} hypotheses, {len(fused)} FusedPoses "
        f"({(hyps.role == 'fit').sum()} / {(fused.role == 'fit').sum()} fit rows)"
    )
    report_dir = args.outputs / "confidence" / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.models_root / args.tag

    # main models: pooled fit rows of every dataset
    main_fit = fit_pair(
        tables,
        hyps.role == "fit",
        fused.role == "fit",
        None,
        None,
        "pooled",
        weight_source=args.weight_source,
    )
    model_h, model_f, fused_x = main_fit["model_h"], main_fit["model_f"], main_fit["fused"]
    ev_h, ev_f = hyps[hyps.role == "eval"], fused_x[fused_x.role == "eval"]
    fit_h, fit_f = hyps[hyps.role == "fit"], fused_x[fused_x.role == "fit"]
    rep: dict[str, Any] = {
        "tag": args.tag,
        "date": time.strftime("%Y-%m-%d"),
        "fit_scenes": {k: sorted(v) for k, v in fit_scenes.items()},
        "n_fit_h": main_fit["n_fit_h"],
        "n_fit_f": main_fit["n_fit_f"],
        "c_h": main_fit["c_h"],
        "c_f": main_fit["c_f"],
        "h_eval": evaluate_sets(model_h, ev_h, ["dataset"]),
        "h_fit": evaluate_sets(model_h, fit_h, []),
        "f_eval": evaluate_sets(model_f, ev_f, ["dataset", "k"]),
        "f_fit": evaluate_sets(model_f, fit_f, []),
        "coefficients_h": model_h.coefficients(),
        "coefficients_f": model_f.coefficients(),
    }
    # per-dataset x k rows for the F model
    for ds, g in ev_f.groupby("dataset"):
        for k, gk in g.groupby("k"):
            if len(gk) >= 20:
                rep["f_eval"][f"{ds} k={k}"] = summary(gk["success"], model_f.predict_proba(gk))
    print("H eval:", {k: round(v["roc_auc"], 3) for k, v in rep["h_eval"].items()})
    print("F eval:", {k: round(v["roc_auc"], 3) for k, v in rep["f_eval"].items()})

    # MLP comparator, under the published protocol (same rows, same C, Platt on OOF logits)
    y_fit_h, y_fit_f = fit_h["success"].to_numpy(bool), fit_f["success"].to_numpy(bool)
    c_h, c_f = main_fit["c_h"]["best"], main_fit["c_f"]["best"]
    sch_h, sch_f = FeatureSchema.hypothesis(), FeatureSchema.fused()
    mlp_h = ConfidenceModel.fit(sch_h, fit_h, y_fit_h, kind="mlp", C=c_h).recalibrated(
        *platt_scaling(oof_logits(sch_h, fit_h, y_fit_h, c_h, kind="mlp"), y_fit_h)
    )
    mlp_f = ConfidenceModel.fit(sch_f, fit_f, y_fit_f, kind="mlp", C=c_f).recalibrated(
        *platt_scaling(oof_logits(sch_f, fit_f, y_fit_f, c_f, kind="mlp"), y_fit_f)
    )
    mh = summary(ev_h["success"], mlp_h.predict_proba(ev_h))
    mf = summary(ev_f["success"], mlp_f.predict_proba(ev_f))
    lh, lf = rep["h_eval"]["pooled"], rep["f_eval"]["pooled"]
    gain_h, gain_f = mh["roc_auc"] - lh["roc_auc"], mf["roc_auc"] - lf["roc_auc"]
    under_fit = (gain_h >= 0.02 and mh["brier"] < lh["brier"]) or (
        gain_f >= 0.02 and mf["brier"] < lf["brier"]
    )
    rep["mlp"] = {
        "h": mh,
        "f": mf,
        "gain_roc_auc_h": gain_h,
        "gain_roc_auc_f": gain_f,
        "decision": (
            "MLP gains ≥ 2 pt ROC-AUC with a lower Brier — logistic under-fits; revisit"
            if under_fit
            else f"logistic kept (MLP ROC-AUC gain H {100 * gain_h:+.1f}, "
            f"F {100 * gain_f:+.1f} pt; no clear under-fit)"
        ),
    }

    # cross-dataset
    rep["cross"] = {}
    if len(args.datasets) > 1:
        for src in args.datasets:
            others = [d for d in args.datasets if d != src]
            pair = fit_pair(
                tables,
                (hyps.role == "fit") & (hyps.dataset == src),
                (fused.role == "fit") & (fused.dataset == src),
                main_fit["c_h"]["best"],
                main_fit["c_f"]["best"],
                f"{src} only",
            )
            th = hyps[hyps.dataset.isin(others)]
            tf = pair["fused"][pair["fused"].dataset.isin(others)]
            rep["cross"][f"{src} → {'+'.join(others)}"] = {
                "h": summary(th["success"], pair["model_h"].predict_proba(th)),
                "f": summary(tf["success"], pair["model_f"].predict_proba(tf)),
            }

    # thresholds on the OOF recalibrated probabilities of the fit rows, rates on eval rows
    p_oof_f = main_fit["p_oof_f"]
    thr = choose_thresholds(y_fit_f, p_oof_f, args.accept_precision, args.reject_precision)
    rep["thresholds"] = thr.to_dict()
    rep["calibration"] = {
        "h": list(model_h.calibration),
        "f": list(model_f.calibration),
        "h_raw_eval": summary(ev_h["success"], main_fit["raw_h"].predict_proba(ev_h)),
        "f_raw_eval": summary(ev_f["success"], main_fit["raw_f"].predict_proba(ev_f)),
        "f_oof_fit": summary(y_fit_f, p_oof_f),
    }
    p_ev_f = model_f.predict_proba(ev_f)
    rep["verdicts"] = {"eval pooled": verdict_rates(ev_f["success"], p_ev_f, thr)}
    for ds, g in ev_f.groupby("dataset"):
        rep["verdicts"][f"eval {ds}"] = verdict_rates(g["success"], model_f.predict_proba(g), thr)
    for k, g in ev_f.groupby("k"):
        rep["verdicts"][f"eval k={k}"] = verdict_rates(g["success"], model_f.predict_proba(g), thr)
    rep["verdicts"]["fit pooled (OOF)"] = verdict_rates(y_fit_f, p_oof_f, thr)
    # the same thresholds applied to Model H on single-view hypotheses
    rep["verdicts"]["eval hypotheses (Model H)"] = verdict_rates(
        ev_h["success"], model_h.predict_proba(ev_h), thr
    )

    # leave-one-scene-out diagnostic over every scene
    rep["loso"] = loso_analysis(
        tables,
        main_fit["c_h"]["best"],
        main_fit["c_f"]["best"],
        args.accept_precision,
        args.reject_precision,
    )
    print("LOSO H:", {k: round(v["roc_auc"], 3) for k, v in rep["loso"]["h"].items()})

    # ablations (the signal is removed from Model H and Model F together)
    rep["ablation_h"], rep["ablation_f"] = ablate_signals(
        tables,
        hyps.role == "fit",
        fused.role == "fit",
        main_fit["c_h"]["best"],
        main_fit["c_f"]["best"],
        lh,
        lf,
    )

    # figures
    figs: dict[str, str] = {}
    curves_h = {
        str(ds): (g["success"].to_numpy(bool), model_h.predict_proba(g))
        for ds, g in ev_h.groupby("dataset")
    }
    curves_f = {
        str(ds): (g["success"].to_numpy(bool), model_f.predict_proba(g))
        for ds, g in ev_f.groupby("dataset")
    }
    for name, curves, title in (
        ("reliability_h", curves_h, "Model H reliability (eval scenes)"),
        ("reliability_f", curves_f, "Model F reliability (eval scenes)"),
    ):
        path = report_dir / f"{name}.png"
        plot_reliability(curves, path, title)
        figs[name] = str(path)
    rc_curves = {f"F {ds}": c for ds, c in curves_f.items()} | {
        f"H {ds}": c for ds, c in curves_h.items()
    }
    plot_risk_coverage(rc_curves, report_dir / "risk_coverage.png", "Risk–coverage (eval scenes)")
    figs["risk_coverage"] = str(report_dir / "risk_coverage.png")
    per_k = {
        f"k={k}": (g["success"].to_numpy(bool), model_f.predict_proba(g))
        for k, g in ev_f.groupby("k")
    }
    plot_risk_coverage(
        per_k, report_dir / "risk_coverage_k.png", "Model F risk–coverage by view count"
    )
    figs["risk_coverage_k"] = str(report_dir / "risk_coverage_k.png")
    rep["figures"] = figs

    # persist
    model_h.save(model_dir / "model_h.json")
    model_f.save(model_dir / "model_f.json")
    thr.save(model_dir / "thresholds.json")
    (model_dir / "card.md").write_text(card(rep, args.tag, model_h, model_f))
    fused_x["p_f"] = model_f.predict_proba(fused_x)
    fused_x["verdict"] = [v.value for v in thr.verdicts(fused_x["p_f"].to_numpy())]
    fused_x.to_parquet(report_dir / "fused_scored.parquet", index=False)
    h_scored = hyps.copy()
    h_scored["p_h"] = main_fit["p_h"]
    h_scored.to_parquet(report_dir / "hypotheses_scored.parquet", index=False)
    rep["seconds"] = time.perf_counter() - t0
    (report_dir / "report.json").write_text(json.dumps(rep, indent=2, default=_json_safe))
    (report_dir / "report.md").write_text(render(rep, args.tag))
    print(f"wrote {model_dir} and {report_dir} ({rep['seconds']:.0f} s)")


def _json_safe(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    main()
