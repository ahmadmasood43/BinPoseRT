"""Refinement analysis (Beta): gate re-decision from stored measurements, precision of rejection,
alpha/beta sweep, stratified before/after tables and the gate-mistake galleries."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from binposert.evaluate.refinement import (
    GATE_REASONS,
    INITIAL_ERROR_BINS_MM,
    PRE_GATE_REASONS,
    add_outcomes,
    choose_gate_caps,
    gate_statistics,
    gate_sweep,
    recall_fraction,
    redecide,
    score_refinement,
    stratify_before_after,
)
from binposert.pipeline.run import run
from binposert.refine import GateParams, gate_reason
from binposert.viz import make_refine_galleries, select_failures

REPO = Path(__file__).resolve().parents[1]
DEFAULT = GateParams()


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        return compose(config_name="config", overrides=list(overrides))


# ----------------------------------------------------------------------------- gate decision


def test_gate_reason_checks_in_fixed_order():
    d = 100.0
    ok = dict(
        fitness=0.9, displacement_mm=1.0, displacement_deg=1.0, iou_coarse=0.8, iou_refined=0.85
    )

    def reason(**kw):
        return gate_reason(**{**ok, **kw}, diameter=d, params=DEFAULT)

    assert reason() is None
    assert reason(fitness=0.1) == "fitness"
    assert reason(fitness=0.1, displacement_mm=99) == "fitness"  # fitness is checked first
    assert reason(displacement_mm=26) == "displacement_translation"  # > 0.25 d
    assert reason(displacement_mm=24.9) is None
    assert reason(displacement_deg=31) == "displacement_rotation"
    assert reason(iou_refined=0.4) == "silhouette_iou"
    assert reason(iou_coarse=0.9, iou_refined=0.75) == "silhouette_iou_drop"
    assert reason(iou_coarse=0.9, iou_refined=0.81) is None
    wide = GateParams(alpha=math.inf, beta_deg=math.inf)
    assert gate_reason(0.9, 1e6, 179.0, 0.8, 0.85, d, wide) is None


def _measurements(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scene_id": rng.integers(1, 5, n),
            "fitness": rng.uniform(0.0, 1.0, n),
            "displacement_mm": rng.uniform(0.0, 60.0, n),
            "displacement_deg": rng.uniform(0.0, 60.0, n),
            "iou_coarse": rng.uniform(0.3, 1.0, n),
            "iou_refined": rng.uniform(0.3, 1.0, n),
            "diameter": rng.uniform(60.0, 200.0, n),
            "image_width": 640,
            "gt_valid": True,
        }
    )


def _stored(df: pd.DataFrame, params: GateParams) -> pd.DataFrame:
    """What a refine stage would have stored under ``params`` (its decision is gate_reason)."""
    out = df.copy()
    out["reason"] = [
        gate_reason(
            *(
                float(r[c])
                for c in (
                    "fitness",
                    "displacement_mm",
                    "displacement_deg",
                    "iou_coarse",
                    "iou_refined",
                    "diameter",
                )
            ),
            params,
        )
        for _, r in df.iterrows()
    ]
    out["accepted"] = out["reason"].isna()
    return out


def test_redecide_reproduces_the_stored_decision_and_keeps_pre_gate_reasons():
    rng = np.random.default_rng(0)
    df = _stored(_measurements(rng, 300), DEFAULT)
    assert 0 < df["accepted"].sum() < len(df)
    again = redecide(df, DEFAULT)
    assert again.isna().equals(df["reason"].isna())
    assert (again.dropna() == df["reason"].dropna()).all()
    # pre-gate rejections have no gate measurements and stay rejected under any caps
    df.loc[df.index[:3], ["reason", "accepted"]] = [
        ["no_depth", False],
        ["icp_failed", False],
        ["no_overlap", False],
    ]
    df.loc[df.index[:3], ["fitness", "iou_refined"]] = np.nan
    wide = GateParams(
        alpha=math.inf, beta_deg=math.inf, min_iou=0.0, max_iou_drop=1.0, min_fitness=0.0
    )
    r = redecide(df, wide)
    assert list(r.iloc[:3]) == ["no_depth", "icp_failed", "no_overlap"]
    assert r.iloc[3:].isna().all()
    assert set(PRE_GATE_REASONS).isdisjoint(GATE_REASONS)


# ----------------------------------------------------------------------------- outcomes


def _scored_rows() -> pd.DataFrame:
    """Four archetypes, diameter 100: accepted+improved, accepted+worse, rejected+correct,
    rejected+good (candidate closer and within 0.1 d); plus a pre-gate rejection."""
    rows = [
        # coarse, cand (mm), accepted, reason
        (20.0, 5.0, True, None),
        (5.0, 30.0, True, None),
        (8.0, 40.0, False, "silhouette_iou"),
        (25.0, 6.0, False, "displacement_translation"),
        (30.0, 30.0, False, "no_depth"),
    ]
    df = pd.DataFrame(rows, columns=["coarse_mssd_mm", "cand_mssd_mm", "accepted", "reason"])
    df["init_mssd_mm"] = df["coarse_mssd_mm"]
    df["coarse_mspd_px"] = df["coarse_mssd_mm"]
    df["cand_mspd_px"] = df["cand_mssd_mm"]
    df["diameter"] = 100.0
    df["image_width"] = 640
    df["gt_valid"] = True
    df["scene_id"] = [1, 1, 2, 2, 3]
    for c in ("fitness", "displacement_mm", "displacement_deg", "iou_coarse", "iou_refined"):
        df[c] = 0.9 if c in ("fitness", "iou_coarse", "iou_refined") else 1.0
    return add_outcomes(df)


def test_add_outcomes_final_pose_follows_the_decision():
    s = _scored_rows()
    assert list(s["final_mssd_mm"]) == [5.0, 30.0, 8.0, 25.0, 30.0]
    assert list(s["improved"]) == [True, False, False, True, False]
    assert list(s["coarse_ok"]) == [False, True, True, False, False]
    assert list(s["cand_ok"]) == [True, False, False, True, False]
    # thresholds 0.05 d .. 0.5 d (strict): 4.9 mm passes all ten, 49.9 mm only the last
    rf = recall_fraction([4.9, 49.9], [4.9, 49.9], [100.0, 100.0], [640, 640])
    assert rf[0] == pytest.approx(1.0) and rf[1] == pytest.approx(0.1)


def test_gate_statistics_precision_of_rejection():
    st = gate_statistics(_scored_rows())
    assert st["n"] == 5 and st["n_accepted"] == 2 and st["rejection_rate"] == pytest.approx(0.6)
    assert st["reasons"] == {"silhouette_iou": 1, "displacement_translation": 1, "no_depth": 1}
    assert st["accepted"] == {"n": 2, "improved": 1, "worsened": 1, "fixed": 1, "broke": 1}
    rg = st["rejected_by_gate"]
    assert rg["n"] == 2 and rg["precision"] == pytest.approx(0.5)
    assert rg["rejected_good"] == 1 and rg["avoided_break"] == 1
    assert rg["per_reason"]["silhouette_iou"]["precision"] == 1.0
    assert rg["per_reason"]["displacement_translation"]["rejected_good"] == 1
    assert st["rejected_pre_gate"]["n"] == 1
    assert st["success_0.1d"]["coarse"] == pytest.approx(0.4)
    assert st["success_0.1d"]["init"] == pytest.approx(0.4)
    assert st["success_0.1d"]["gate_off"] == pytest.approx(0.4)
    assert st["success_0.1d"]["final"] == pytest.approx(0.4)
    # only hypotheses matched to a valid GT count
    s = _scored_rows()
    s.loc[0, "gt_valid"] = False
    assert gate_statistics(s)["n"] == 4


def test_gate_sweep_and_cap_choice():
    rng = np.random.default_rng(1)
    df = _measurements(rng, 400)
    df["coarse_mssd_mm"] = rng.uniform(2.0, 40.0, len(df))
    # candidates that moved a lot are bad, small moves are good: a translation cap should pay off
    df["cand_mssd_mm"] = np.where(df["displacement_mm"] > 20, 60.0, 2.0)
    df["init_mssd_mm"] = df["coarse_mssd_mm"]
    df["coarse_mspd_px"] = df["coarse_mssd_mm"]
    df["cand_mspd_px"] = df["cand_mssd_mm"]
    df = _stored(df, DEFAULT)
    df = add_outcomes(df)
    base = GateParams(min_iou=0.0, max_iou_drop=1.0, min_fitness=0.0)
    sweep = gate_sweep(df, [0.05, 0.1, 0.25, math.inf], [30.0, math.inf], base, scenes=[1, 2])
    assert len(sweep) == 8 and (sweep["n"] == df["scene_id"].isin([1, 2]).sum()).all()
    off = sweep[(sweep.alpha == math.inf) & (sweep.beta_deg == math.inf)].iloc[0]
    assert off["rejection_rate"] == 0.0
    # alpha = 0.1 x (60..200 mm) rejects the 20 mm+ moves without losing many good ones
    best = sweep.loc[sweep["objective"].idxmax()]
    assert best["alpha"] in (0.1, 0.25) and best["objective"] > off["objective"]
    choice = choose_gate_caps(sweep, base)
    assert choice["best"]["alpha"] == best["alpha"] and choice["changed"]
    assert choice["params"] == choice["best"]
    assert choice["gain_over_default"] == pytest.approx(
        best["objective"]
        - sweep[(sweep.alpha == 0.25) & (sweep.beta_deg == 30.0)]["objective"].iloc[0]
    )
    # a tiny gain does not move the default
    tie = sweep.copy()
    tie["objective"] = 0.5
    tie.loc[0, "objective"] = 0.5004
    kept = choose_gate_caps(tie, base)
    assert not kept["changed"] and kept["params"] == {"alpha": 0.25, "beta_deg": 30.0}
    # extra fields are crossed into the grid and chosen alongside
    ext = gate_sweep(df, [0.25, math.inf], [30.0], base, extra={"min_fitness": [0.0, 0.5]})
    assert len(ext) == 4 and set(ext["min_fitness"]) == {0.0, 0.5}
    assert set(choose_gate_caps(ext, base)["params"]) == {"alpha", "beta_deg", "min_fitness"}


# ----------------------------------------------------------------------------- per-GT strata


def _gt_rows(mssd: list[float], ar: list[float], vis: list[float]) -> pd.DataFrame:
    n = len(mssd)
    return pd.DataFrame(
        {
            "scene_id": 1,
            "image_id": 1,
            "object_id": 1,
            "gt_index": range(n),
            "visible_fraction": vis,
            "mssd_mm": mssd,
            "mspd_px": mssd,
            "success_0.1d": [float(m < 10) for m in mssd],
            "ar_mssd": ar,
            "ar_mspd": ar,
            "ar_vsd": ar,
        }
    )


def test_stratify_before_after_bins_by_initial_error_and_visibility():
    before = _gt_rows(
        [3, 7, 15, 30, math.inf], [0.9, 0.7, 0.4, 0.1, 0.0], [0.9, 0.9, 0.5, 0.2, 0.9]
    )
    after = _gt_rows([2, 4, 30, 30, math.inf], [1.0, 0.8, 0.2, 0.1, 0.0], [0.9, 0.9, 0.5, 0.2, 0.9])
    out = stratify_before_after(before, after)
    assert out["n_gt"] == 5 and out["n_no_prediction"] == 1
    assert out["initial_bins"] == [list(b) for b in INITIAL_ERROR_BINS_MM]
    by = {
        (
            tuple(e["initial"]) if e["initial"] else None,
            tuple(e["visibility"]) if e["visibility"] else None,
        ): e
        for e in out["strata"]
    }
    assert by[(None, None)]["n_gt"] == 4  # the no-prediction instance has no initial error
    assert by[((0, 5), None)]["n_gt"] == 1 and by[((0, 5), None)]["ar_before"] == pytest.approx(0.9)
    assert by[((0, 5), None)]["ar_after"] == pytest.approx(1.0)
    assert by[((5, 10), None)]["delta_ar"] == pytest.approx(0.1)
    assert by[((10, 20), None)]["delta_ar"] == pytest.approx(-0.2)
    assert by[((20, math.inf), None)]["n_gt"] == 1
    assert by[((0, 5), (0.6, 1.0))]["n_gt"] == 1 and by[((0, 5), (0.1, 0.3))]["n_gt"] == 0
    assert by[(None, (0.1, 0.3))]["n_gt"] == 1 and by[(None, (0.3, 0.6))][
        "ar_before"
    ] == pytest.approx(0.4)
    assert by[(None, (0.6, 1.0))]["success_0.1d_after"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- fixture end to end


def test_scoring_and_galleries_on_the_fixture(tmp_path):
    base = (
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "estimator.params.t_sigma_mm=4",
        "estimator.params.r_sigma_deg=6",
        "evaluate.with_vsd=false",
    )
    before = run(_compose(*base), repo_root=REPO)
    after = run(_compose(*base, "stages=[segment,coarse_pose,refine,evaluate]"), repo_root=REPO)
    from binposert.pipeline.run import make_context

    ctx = make_context(_compose(*base), repo_root=REPO)
    details = pd.read_parquet(after.refs["refine"].dir / "refine_details.parquet")
    scored = score_refinement(details, ctx.dataset, n_model_points=500, n_workers=2)
    assert len(scored) == len(details) == 12
    assert scored["gt_valid"].all() and np.isfinite(scored["coarse_mssd_mm"]).all()
    assert np.isfinite(scored["init_mssd_mm"]).all()
    # refinement of a noisy GT pose converges: every accepted candidate is closer than its coarse
    # pose; the one occluded instance ICP wanders off on is caught by the gate (candidate worse)
    acc = scored[scored["accepted"]]
    assert len(acc) >= 11 and acc["improved"].all()
    assert (acc["final_mssd_mm"] == acc["cand_mssd_mm"]).all()
    assert (~scored[~scored["accepted"]]["improved"]).all()
    assert (scored["coarse_t_err_mm"] > 0).all() and (scored["coarse_r_err_deg"] > 0).all()
    st = gate_statistics(scored)
    assert (
        st["n_accepted"] >= 11 and st["recall_fraction"]["final"] > st["recall_fraction"]["coarse"]
    )
    assert st["rejected_by_gate"]["n"] == 0 or st["rejected_by_gate"]["precision"] == 1.0

    rows_b = pd.read_parquet(before.refs["evaluate"].dir / "gt_rows.parquet")
    rows_a = pd.read_parquet(after.refs["evaluate"].dir / "gt_rows.parquet")
    strata = stratify_before_after(rows_b, rows_a)
    total = next(e for e in strata["strata"] if e["initial"] is None and e["visibility"] is None)
    assert total["n_gt"] == 12 and total["delta_ar"] > 0.1

    # galleries: force one gate mistake of each kind so both tiles get drawn
    forced = scored[scored["accepted"]].copy()
    i0, i1 = forced.index[:2]
    forced.loc[i0, ["coarse_mssd_mm", "cand_mssd_mm"]] = [1.0, 30.0]  # accepted, made worse
    forced.loc[i1, ["accepted", "reason"]] = [False, "silhouette_iou"]  # rejected a good one
    forced = add_outcomes(forced)
    sel = select_failures(forced, n=5)
    assert len(sel["accepted_worse"]) == 1 and len(sel["rejected_good"]) == 1
    written = make_refine_galleries(forced, ctx.dataset, tmp_path / "gallery", n=5)
    for kind, path in written.items():
        assert path.exists()
        items = json.loads(path.with_suffix(".json").read_text())["items"]
        assert len(items) == 1 and items[0]["accepted"] == (kind == "accepted_worse")
