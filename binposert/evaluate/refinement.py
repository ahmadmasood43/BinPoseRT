"""Refinement analysis (Beta; DECISIONS.md §4): before/after AR stratified by initial-error and
visibility bins, gate statistics (rejection rate, precision of rejection) and an offline sweep of
the gate caps over the measurements a refine stage stores in ``refine_details.parquet``.

Two data sources, two protocols:

* :func:`stratify_before_after` joins the per-GT rows of two evaluate stages (un-refined and
  refined) — the BOP protocol, VSD included. The initial error of a GT instance is the MSSD of the
  closest un-refined prediction.
* :func:`score_refinement` scores every stored hypothesis against its closest GT with MSSD and MSPD
  (no VSD, no top-n matching): a per-hypothesis view in which the coarse, candidate and final poses
  of the *same* hypothesis are comparable, so a rejection can be judged right or wrong and the gate
  re-decided under other caps (:func:`gate_sweep`) without re-running anything.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from binposert.data import BopDataset
from binposert.evaluate.localisation import MSPD_THRESH, MSSD_THRESH, valid_ground_truth
from binposert.evaluate.metrics import mspd, mssd
from binposert.evaluate.stratified import VISIBILITY_BINS
from binposert.refine.gate import GateParams, gate_reason
from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.transforms import translation_distance

INITIAL_ERROR_BINS_MM: list[tuple[float, float]] = [(0, 5), (5, 10), (10, 20), (20, math.inf)]
GT_KEYS = ["scene_id", "image_id", "object_id", "gt_index"]
# Reasons decided before the gate ran (no candidate worth judging); the gate cannot re-decide them.
PRE_GATE_REASONS = ("no_depth", "depth_window", "no_overlap", "icp_failed")
GATE_REASONS = (
    "fitness",
    "displacement_translation",
    "displacement_rotation",
    "silhouette_iou",
    "silhouette_iou_drop",
)
SCORE_COLUMNS = [
    "gt_index",
    "gt_valid",
    "visible_fraction",
    "diameter",
    "image_width",
    "coarse_mssd_mm",
    "init_mssd_mm",
    "cand_mssd_mm",
    "coarse_mspd_px",
    "cand_mspd_px",
    "coarse_t_err_mm",
    "coarse_r_err_deg",
]


# ----------------------------------------------------------------------------- per-hypothesis


def _T(row: Any, prefix: str) -> npt.NDArray[np.float64]:
    return np.array(
        [[float(row[f"{prefix}T_{i}{j}"]) for j in range(4)] for i in range(4)], dtype=np.float64
    )


def _T_init(T_coarse: npt.NDArray[np.float64], z_shift_mm: float) -> npt.NDArray[np.float64]:
    """The pose ICP started from: the coarse pose shifted along its ray by the depth init."""
    T = T_coarse.copy()
    tz = float(T_coarse[2, 3])
    if np.isfinite(z_shift_mm) and z_shift_mm != 0.0 and tz > 0:
        T[:3, 3] = T_coarse[:3, 3] * (tz + z_shift_mm) / tz
    return T


def _score_scene(
    scene_id: int, details: pd.DataFrame, dataset: BopDataset, n_model_points: int
) -> list[dict[str, Any]]:
    """Closest GT of each hypothesis' object (by coarse MSSD); coarse, depth-initialised and
    candidate errors to it."""
    pts_cache: dict[int, npt.NDArray[np.float64]] = {}
    out: list[dict[str, Any]] = []
    for image_id_raw, d_img in details.groupby("image_id", sort=True):
        image_id = int(cast(Any, image_id_raw))
        view, gts = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
        width = view.image_size[1]
        for _, row in d_img.iterrows():
            oid = int(row["object_id"])
            model = dataset.load_model(oid)
            if oid not in pts_cache:
                pts_cache[oid] = (
                    model.sample_points(n_model_points, seed=oid)
                    if n_model_points > 0
                    else model.vertices
                )
            pts = pts_cache[oid]
            gts_obj = [g for g in gts if g.object_id == oid]
            rec: dict[str, Any] = {k: row[k] for k in ("scene_id", "image_id", "hypothesis_id")}
            rec["object_id"] = oid
            rec["detection_id"] = int(row["detection_id"])
            if not gts_obj:
                out.append(rec | dict.fromkeys(SCORE_COLUMNS, float("nan")))
                continue
            valid = {
                g.gt_index for g in valid_ground_truth(dataset, scene_id, image_id, oid, gts_obj)
            }
            T_c, T_r = _T(row, "coarse_"), _T(row, "cand_")
            T_i = _T_init(T_c, float(row["z_shift_mm"]) if "z_shift_mm" in row else 0.0)
            e_c = [mssd(T_c, g.T_camera_object, pts, model.symmetry) for g in gts_obj]
            j = int(np.argmin(e_c))
            g = gts_obj[j]
            rec.update(
                gt_index=g.gt_index,
                gt_valid=g.gt_index in valid,
                visible_fraction=g.visible_fraction,
                diameter=model.diameter,
                image_width=width,
                coarse_mssd_mm=e_c[j],
                init_mssd_mm=mssd(T_i, g.T_camera_object, pts, model.symmetry),
                cand_mssd_mm=mssd(T_r, g.T_camera_object, pts, model.symmetry),
                coarse_mspd_px=mspd(T_c, g.T_camera_object, pts, model.symmetry, view.K),
                cand_mspd_px=mspd(T_r, g.T_camera_object, pts, model.symmetry, view.K),
                coarse_t_err_mm=translation_distance(T_c, g.T_camera_object),
                coarse_r_err_deg=sym_aware_rotation_distance_deg(
                    T_c, g.T_camera_object, model.symmetry
                ),
            )
            out.append(rec)
    return out


def score_refinement(
    details: pd.DataFrame, dataset: BopDataset, n_model_points: int = 0, n_workers: int = 1
) -> pd.DataFrame:
    """``refine_details.parquet`` rows + :data:`SCORE_COLUMNS` (GT errors of the coarse and the
    candidate pose) + the outcome columns of :func:`add_outcomes` for the stored gate decision.
    Hypotheses whose object has no GT in the image get NaN scores. Scenes run in parallel when
    ``n_workers > 1`` (spawned processes)."""
    scene_ids = sorted(int(s) for s in details["scene_id"].unique())
    jobs = [(sid, details[details.scene_id == sid], dataset, n_model_points) for sid in scene_ids]
    if n_workers > 1 and len(jobs) > 1:
        from binposert.pipeline.pool import scene_pool

        with scene_pool(min(n_workers, len(jobs))) as pool:
            results = pool.starmap(_score_scene, jobs)
    else:
        results = [_score_scene(*job) for job in jobs]
    keys = ["scene_id", "image_id", "object_id", "detection_id", "hypothesis_id"]
    scores = pd.DataFrame([r for rows in results for r in rows], columns=keys + SCORE_COLUMNS)
    merged = details.merge(scores, on=keys, how="left", validate="one_to_one")
    return add_outcomes(merged)


def recall_fraction(
    mssd_mm: npt.ArrayLike, mspd_px: npt.ArrayLike, diameter: npt.ArrayLike, width: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Per-hypothesis share of the BOP MSSD and MSPD thresholds passed (mean of the two): the
    VSD-less, matching-less counterpart of a GT instance's recall fraction."""
    e_s = np.asarray(mssd_mm, dtype=float)[:, None]
    e_p = np.asarray(mspd_px, dtype=float)[:, None]
    thr_s = np.asarray(diameter, dtype=float)[:, None] * MSSD_THRESH[None, :]
    thr_p = np.asarray(width, dtype=float)[:, None] / 640.0 * MSPD_THRESH[None, :]
    return 0.5 * ((e_s < thr_s).mean(axis=1) + (e_p < thr_p).mean(axis=1))


def redecide(scored: pd.DataFrame, params: GateParams) -> pd.Series:
    """The gate's reason for every row under ``params`` (None = accepted), from the stored
    measurements; pre-gate rejections keep their reason."""
    cols = [
        "fitness",
        "displacement_mm",
        "displacement_deg",
        "iou_coarse",
        "iou_refined",
        "diameter",
    ]
    values = scored[cols].to_numpy(dtype=float)
    stored = scored["reason"].to_numpy(dtype=object)
    reasons: list[str | None] = []
    for k in range(len(scored)):
        if isinstance(stored[k], str) and stored[k] in PRE_GATE_REASONS:
            reasons.append(stored[k])
        else:
            f, d_t, d_r, iou_c, iou_r, diameter = (float(v) for v in values[k])
            reasons.append(gate_reason(f, d_t, d_r, iou_c, iou_r, diameter, params))
    return pd.Series(reasons, index=scored.index, dtype=object)


def add_outcomes(scored: pd.DataFrame, params: GateParams | None = None) -> pd.DataFrame:
    """Outcome columns for the stored decision (``params=None``) or a re-decided one: ``accepted``,
    ``reason``, ``final_*`` errors, per-pose recall fractions (``*_rf``) and successes at
    ``MSSD < 0.1 d`` (``*_ok``), ``improved`` (candidate closer to GT than the coarse pose)."""
    s = scored.copy()
    if params is not None:
        s["reason"] = redecide(s, params)
        s["accepted"] = s["reason"].isna()
    acc = s["accepted"].to_numpy(dtype=bool)
    s["final_mssd_mm"] = np.where(acc, s["cand_mssd_mm"], s["coarse_mssd_mm"])
    s["final_mspd_px"] = np.where(acc, s["cand_mspd_px"], s["coarse_mspd_px"])
    for name in ("coarse", "cand", "final"):
        s[f"{name}_rf"] = recall_fraction(
            s[f"{name}_mssd_mm"], s[f"{name}_mspd_px"], s["diameter"], s["image_width"]
        )
        s[f"{name}_ok"] = s[f"{name}_mssd_mm"] < 0.1 * s["diameter"]
    s["improved"] = s["cand_mssd_mm"] < s["coarse_mssd_mm"]
    return s


def gate_statistics(scored: pd.DataFrame, params: GateParams | None = None) -> dict[str, Any]:
    """Rejection rate, reasons and the *precision of rejection* — how often the gate's rejection
    was the right call, i.e. the candidate was no closer to GT than the coarse pose — plus the
    mirror numbers for acceptances and the "gate off" alternative (every candidate accepted).
    Only hypotheses matched to a valid GT count (those are the ones AR sees)."""
    s = add_outcomes(scored, params) if params is not None else scored
    s = s[s["gt_valid"].fillna(False).astype(bool)]
    n = int(len(s))
    acc = s[s["accepted"]]
    rej = s[~s["accepted"]]
    by_gate = rej[rej["reason"].isin(GATE_REASONS)]
    pre = rej[rej["reason"].isin(PRE_GATE_REASONS)]

    def _mean(x: pd.Series) -> float:
        return float(x.mean()) if len(x) else float("nan")

    per_reason = {}
    for reason, grp in by_gate.groupby("reason"):
        per_reason[str(reason)] = {
            "n": int(len(grp)),
            "precision": _mean(~grp["improved"]),
            "precision_success": _mean(~grp["cand_ok"]),
            "rejected_good": int((grp["improved"] & grp["cand_ok"]).sum()),
            "avoided_break": int((~grp["cand_ok"] & grp["coarse_ok"]).sum()),
        }
    return {
        "n": n,
        "n_accepted": int(len(acc)),
        "rejection_rate": len(rej) / n if n else float("nan"),
        "reasons": {str(k): int(v) for k, v in rej["reason"].value_counts().items()},
        "accepted": {
            "n": int(len(acc)),
            "improved": int(acc["improved"].sum()),
            "worsened": int(
                (~acc["improved"] & (acc["cand_mssd_mm"] > acc["coarse_mssd_mm"])).sum()
            ),
            "fixed": int((acc["cand_ok"] & ~acc["coarse_ok"]).sum()),
            "broke": int((~acc["cand_ok"] & acc["coarse_ok"]).sum()),
        },
        "rejected_by_gate": {
            "n": int(len(by_gate)),
            # strict: the candidate was no closer to GT than the coarse pose
            "precision": _mean(~by_gate["improved"]),
            # at the success level: the candidate was not within 0.1 d (rejecting it cost nothing)
            "precision_success": _mean(~by_gate["cand_ok"]),
            "rejected_good": int((by_gate["improved"] & by_gate["cand_ok"]).sum()),
            "avoided_break": int((~by_gate["cand_ok"] & by_gate["coarse_ok"]).sum()),
            "per_reason": per_reason,
        },
        "rejected_pre_gate": {"n": int(len(pre))},
        "recall_fraction": {
            "coarse": _mean(s["coarse_rf"]),
            "gate_off": _mean(s["cand_rf"]),
            "final": _mean(s["final_rf"]),
        },
        "success_0.1d": {
            "coarse": _mean(s["coarse_ok"]),
            "init": _mean(s["init_mssd_mm"] < 0.1 * s["diameter"]),
            "gate_off": _mean(s["cand_ok"]),
            "final": _mean(s["final_ok"]),
        },
    }


def gate_sweep(
    scored: pd.DataFrame,
    alphas: list[float],
    betas: list[float],
    base: GateParams,
    scenes: list[int] | None = None,
    extra: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Re-decide every hypothesis under each (alpha, beta) — crossed with the grids in ``extra``
    (other :class:`GateParams` fields, e.g. ``{"min_iou": [0.0, 0.5]}``), the remaining caps from
    ``base`` — and score the final poses: one row per combination with ``objective`` (mean final
    recall fraction over valid-GT hypotheses of ``scenes``, all if None), ``success_0.1d``,
    ``rejection_rate``, ``rejection_precision`` and ``net_successes`` (saved minus lost)."""
    s = scored if scenes is None else scored[scored["scene_id"].isin(scenes)]
    grids: dict[str, list[float]] = {"alpha": alphas, "beta_deg": betas, **(extra or {})}
    names = list(grids)
    rows = []
    for values in itertools.product(*(grids[n] for n in names)):
        p = dataclasses.replace(base, **dict(zip(names, values, strict=True)))
        st = gate_statistics(s, p)
        rg = st["rejected_by_gate"]
        rows.append(
            {
                **dict(zip(names, values, strict=True)),
                "n": st["n"],
                "objective": st["recall_fraction"]["final"],
                "success_0.1d": st["success_0.1d"]["final"],
                "rejection_rate": st["rejection_rate"],
                "rejection_precision": rg["precision"],
                "net_successes": rg["avoided_break"] - rg["rejected_good"],
            }
        )
    return pd.DataFrame(rows)


def choose_gate_caps(
    sweep: pd.DataFrame, default: GateParams, min_gain: float = 0.001
) -> dict[str, Any]:
    """The swept combination with the best objective, unless it beats the default by less than
    ``min_gain`` (then the default stays: no change without evidence). Returns the chosen values
    of every swept field under ``params`` plus the bookkeeping."""
    fields = [c for c in sweep.columns if c in {f.name for f in dataclasses.fields(GateParams)}]
    obj = sweep["objective"].to_numpy(dtype=float)
    k = int(np.nanargmax(obj))
    at_default = np.ones(len(sweep), dtype=bool)
    for f in fields:
        at_default &= np.isclose(sweep[f].to_numpy(dtype=float), getattr(default, f))
    default_obj = float(obj[at_default][0]) if at_default.any() else float("nan")
    gain = float(obj[k]) - default_obj
    keep = not (np.isfinite(gain) and gain >= min_gain)
    best = {f: float(sweep[f].iloc[k]) for f in fields}
    chosen = {f: getattr(default, f) for f in fields} if keep else best
    return {
        "params": chosen,
        "objective": default_obj if keep else float(obj[k]),
        "default_objective": default_obj,
        "best": best,
        "best_objective": float(obj[k]),
        "gain_over_default": gain,
        "changed": not keep,
    }


# ----------------------------------------------------------------------------- per-GT (BOP rows)


def _pooled(rows: pd.DataFrame) -> dict[str, float]:
    entry: dict[str, float] = {}
    for col in ("ar_vsd", "ar_mssd", "ar_mspd", "success_0.1d"):
        vals = rows[col].to_numpy(dtype=float) if len(rows) and col in rows else np.array([])
        vals = vals[np.isfinite(vals)]
        entry[col] = float(vals.mean()) if len(vals) else float("nan")
    finite = [entry[c] for c in ("ar_vsd", "ar_mssd", "ar_mspd") if np.isfinite(entry[c])]
    entry["ar"] = float(np.mean(finite)) if finite else float("nan")
    return entry


def _in_bin(v: pd.Series, lo: float, hi: float, last: bool) -> pd.Series:
    return (v >= lo) & ((v <= hi) if last and np.isfinite(hi) else (v < hi))


def stratify_before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    initial_bins: list[tuple[float, float]] | None = None,
    visibility_bins: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Join the ``gt_rows`` of an un-refined and a refined evaluate stage on the GT instance and
    pool AR before/after per initial-error bin (MSSD of the closest un-refined prediction, mm) ×
    visibility bin, with the marginals (``initial=None`` / ``visibility=None`` = all). GT instances
    without an un-refined prediction have no initial error and are counted separately."""
    initial_bins = INITIAL_ERROR_BINS_MM if initial_bins is None else initial_bins
    visibility_bins = VISIBILITY_BINS if visibility_bins is None else visibility_bins
    cols = ["ar_vsd", "ar_mssd", "ar_mspd", "success_0.1d"]
    b = before[GT_KEYS + ["visible_fraction", "mssd_mm"] + cols]
    a = after[GT_KEYS + cols]
    j = b.merge(a, on=GT_KEYS, suffixes=("_before", "_after"), validate="one_to_one")
    has_init = np.isfinite(j["mssd_mm"])
    strata: list[dict[str, Any]] = []
    init_sel: list[tuple[tuple[float, float] | None, pd.Series]] = [(None, has_init)]
    for k, (lo, hi) in enumerate(initial_bins):
        init_sel.append(
            ((lo, hi), has_init & _in_bin(j["mssd_mm"], lo, hi, k == len(initial_bins) - 1))
        )
    vis_sel: list[tuple[tuple[float, float] | None, pd.Series]] = [
        (None, pd.Series(True, index=j.index))
    ]
    for k, (lo, hi) in enumerate(visibility_bins):
        vis_sel.append(
            ((lo, hi), _in_bin(j["visible_fraction"], lo, hi, k == len(visibility_bins) - 1))
        )
    for ib, isel in init_sel:
        for vb, vsel in vis_sel:
            sel = j[isel & vsel]
            entry: dict[str, Any] = {
                "initial": None if ib is None else [float(ib[0]), float(ib[1])],
                "visibility": None if vb is None else [float(vb[0]), float(vb[1])],
                "n_gt": int(len(sel)),
            }
            for when in ("before", "after"):
                pooled = _pooled(sel.rename(columns={f"{c}_{when}": c for c in cols}))
                entry.update({f"{k}_{when}": v for k, v in pooled.items()})
            entry["delta_ar"] = entry["ar_after"] - entry["ar_before"]
            strata.append(entry)
    return {
        "n_gt": int(len(j)),
        "n_no_prediction": int((~has_init).sum()),
        "initial_bins": [list(bn) for bn in initial_bins],
        "visibility_bins": [list(bn) for bn in visibility_bins],
        "strata": strata,
    }
