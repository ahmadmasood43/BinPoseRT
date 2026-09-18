"""ConfidenceModels (D11): schema, synthetic pass/fail signals, calibration metrics, Verdict
thresholds and the labelled tables on the mini fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from binposert.confidence import (
    ConfidenceModel,
    FeatureSchema,
    VerdictThresholds,
    brier,
    choose_thresholds,
    ece,
    label_fused,
    label_hypotheses,
    pr_auc,
    reliability_table,
    risk_coverage,
    roc_auc,
    summary,
    verdict_rates,
)
from binposert.confidence.schema import MISSING_SUFFIX, SCHEMA_VERSION
from binposert.data import BopDataset
from binposert.pipeline.artefacts import (
    HypothesisRecord,
    hypothesis_to_row,
    transform_to_columns,
)
from binposert.transforms import rotvec_T
from binposert.types import PoseHypothesis, QualitySignals, Stage, Verdict

# ----------------------------------------------------------------------------- synthetic signals


def synthetic_hypotheses(n: int, seed: int = 0, missing_rate: float = 0.1) -> pd.DataFrame:
    """Rows whose signals separate successes from failures imperfectly, with NaN patches that are
    themselves informative (a missing ICP fitness goes with failure more often than not)."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.55
    icp = np.where(y, rng.normal(0.85, 0.08, n), rng.normal(0.45, 0.18, n)).clip(0, 1)
    iou = np.where(y, rng.normal(0.8, 0.1, n), rng.normal(0.4, 0.2, n)).clip(0, 1)
    seg = np.where(y, rng.normal(0.6, 0.15, n), rng.normal(0.45, 0.15, n)).clip(0, 1)
    diameter = rng.choice([60.0, 110.0, 300.0], n)
    rmse = np.where(y, rng.normal(0.02, 0.005, n), rng.normal(0.05, 0.02, n)).clip(0.001) * diameter
    disp = np.where(y, rng.normal(0.04, 0.02, n), rng.normal(0.15, 0.08, n)).clip(0) * diameter
    missing = rng.random(n) < missing_rate * np.where(y, 0.5, 1.5)
    icp[missing] = np.nan
    rmse[missing] = np.nan
    rows = []
    for i in range(n):
        s = QualitySignals(
            seg_score=seg[i],
            pose_score=rng.random(),
            icp_fitness=icp[i],
            icp_rmse_mm=rmse[i],
            depth_coverage=rng.uniform(0.5, 1.0),
            visible_fraction=rng.uniform(0.3, 1.0),
            silhouette_iou=iou[i],
            displacement_mm=disp[i],
            displacement_deg=rng.uniform(0, 20),
        )
        rows.append(
            {
                **s.to_row(),
                "diameter": diameter[i],
                "rejection_reason": "fitness" if (missing[i] and rng.random() < 0.8) else None,
                "rejected": 0.0,
                "success": bool(y[i]),
            }
        )
    df = pd.DataFrame(rows)
    df["rejected"] = df["rejection_reason"].notna().astype(float)
    return df


# ----------------------------------------------------------------------------- schema


def test_schema_imputes_nan_with_indicator_and_normalises_by_diameter():
    schema = FeatureSchema.hypothesis()
    df = synthetic_hypotheses(50, seed=1)
    df.loc[0, "icp_fitness"] = np.nan
    df.loc[0, "icp_rmse_mm"] = np.nan
    df.loc[1, "icp_rmse_mm"] = 6.0
    df.loc[1, "diameter"] = 60.0
    df.loc[2, "icp_rmse_mm"] = np.inf  # an ICP without correspondences: "not available" too
    X = schema.featurize(df)
    names = schema.feature_names
    assert X.shape == (50, len(names)) and np.isfinite(X).all()
    assert X[0, names.index("icp_fitness")] == 0.0
    assert X[0, names.index("icp_fitness" + MISSING_SUFFIX)] == 1.0
    assert X[1, names.index("icp_fitness" + MISSING_SUFFIX)] == 0.0
    assert X[1, names.index("icp_rmse_mm/d")] == pytest.approx(0.1)
    assert X[2, names.index("icp_rmse_mm/d")] == 0.0
    assert X[2, names.index("icp_rmse_mm" + MISSING_SUFFIX)] == 1.0
    assert X[1, names.index("log_diameter")] == pytest.approx(math.log(60.0))
    # multi-view fields absent on every single-view row: constant zero + constant indicator
    assert (X[:, names.index("n_views" + MISSING_SUFFIX)] == 1.0).all()
    assert schema.features_of_signal("dispersion_mm") == [
        "dispersion_mm/d",
        "dispersion_mm_missing",
    ]
    with pytest.raises(ValueError, match="lacks columns"):
        schema.featurize(df.drop(columns=["seg_score"]))
    with pytest.raises(ValueError, match="version"):
        FeatureSchema.from_dict({**schema.to_dict(), "version": SCHEMA_VERSION + 1})
    assert FeatureSchema.from_dict(schema.to_dict()) == schema
    assert FeatureSchema.fused().n_features > schema.n_features


# ----------------------------------------------------------------------------- models


def test_logistic_model_separates_synthetic_pass_fail_signals_with_auc_above_0_9():
    fit, held = synthetic_hypotheses(600, seed=2), synthetic_hypotheses(400, seed=3)
    model = ConfidenceModel.fit(FeatureSchema.hypothesis(), fit, fit["success"], C=1.0)
    p = model.predict_proba(held)
    assert p.shape == (400,) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc(held["success"], p) > 0.9
    assert brier(held["success"], p) < 0.15
    assert ece(held["success"], p) < 0.1
    coef = model.coefficients()
    assert coef["icp_fitness"] > 0 and coef["silhouette_iou"] > 0
    assert coef["displacement_mm/d"] < 0 and coef["icp_rmse_mm/d"] < 0
    assert model.provenance["n_rows"] == 600 and 0.4 < model.provenance["base_rate"] < 0.7


def test_model_round_trips_through_json_and_mlp_comparator_fits(tmp_path):
    fit, held = synthetic_hypotheses(500, seed=4), synthetic_hypotheses(200, seed=5)
    for kind in ("logistic", "mlp"):
        model = ConfidenceModel.fit(FeatureSchema.hypothesis(), fit, fit["success"], kind=kind)
        path = tmp_path / f"{kind}.json"
        model.save(path)
        again = ConfidenceModel.load(path)
        assert again.kind == kind and again.schema == model.schema
        np.testing.assert_allclose(again.predict_proba(held), model.predict_proba(held))
        assert roc_auc(held["success"], again.predict_proba(held)) > 0.85
    with pytest.raises(ValueError, match="both successes and failures"):
        ConfidenceModel.fit(FeatureSchema.hypothesis(), fit, np.ones(len(fit), dtype=bool))
    assert (
        ConfidenceModel.fit(
            FeatureSchema.hypothesis(), fit, fit["success"], kind="mlp"
        ).coefficients()
        == {}
    )


def test_platt_recalibration_round_trips_and_recovers_a_known_scale(tmp_path):
    from binposert.confidence import platt_scaling

    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 4000)
    y = rng.random(len(z)) < 1 / (1 + np.exp(-(0.5 * z - 0.3)))  # truth: half the sharpness
    a, b = platt_scaling(z, y)
    assert a == pytest.approx(0.5, abs=0.1) and b == pytest.approx(-0.3, abs=0.15)
    fit = synthetic_hypotheses(500, seed=8)
    model = ConfidenceModel.fit(FeatureSchema.hypothesis(), fit, fit["success"])
    cal = model.recalibrated(a, b)
    p_raw, p_cal = model.predict_proba(fit), cal.predict_proba(fit)
    assert np.all(np.diff(p_cal[np.argsort(p_raw)]) >= -1e-12)  # monotone in the raw score
    assert p_cal.std() < p_raw.std()  # less extreme
    cal.save(tmp_path / "cal.json")
    again = ConfidenceModel.load(tmp_path / "cal.json")
    assert again.calibration == cal.calibration
    np.testing.assert_allclose(again.predict_proba(fit), p_cal)
    assert model.calibration == (1.0, 0.0)


# ----------------------------------------------------------------------------- calibration


def test_calibration_metrics_on_perfect_constant_and_random_confidences():
    rng = np.random.default_rng(0)
    y = rng.random(1000) < 0.6
    perfect = y.astype(float)
    assert roc_auc(y, perfect) == 1.0 and pr_auc(y, perfect) == 1.0
    assert brier(y, perfect) == 0.0 and ece(y, perfect) == 0.0
    rc = risk_coverage(y, perfect)
    assert rc.e_aurc == pytest.approx(0.0) and rc.risk_at(0.5) == 0.0
    assert rc.coverage_at_risk(0.0) == pytest.approx(y.mean(), abs=2e-3)
    # a constant confidence equal to the base rate is calibrated but does not discriminate
    const = np.full(1000, y.mean())
    assert ece(y, const) == pytest.approx(0.0, abs=1e-12)
    assert brier(y, const) == pytest.approx(y.mean() * (1 - y.mean()))
    assert math.isnan(roc_auc(np.ones(5, dtype=bool), np.ones(5)))
    # a constant confidence far from the base rate has ECE equal to the gap
    assert ece(y, np.full(1000, 0.9)) == pytest.approx(0.9 - y.mean(), abs=1e-12)
    s = summary(y, rng.random(1000))
    assert 0.4 < s["roc_auc"] < 0.6 and s["n"] == 1000 and s["e_aurc"] > 0
    assert set(s) >= {"brier", "ece", "ece_quantile", "mce", "aurc", "coverage_at_risk_05"}


def test_reliability_table_bins_cover_the_unit_interval_and_count_every_row():
    y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    p = np.array([1.0, 0.0, 0.95, 0.91, 0.05, 0.5, 0.5, 0.49, 0.85, 0.99])
    bins = reliability_table(y, p, n_bins=10)
    assert len(bins) == 10 and sum(b.n for b in bins) == 10
    assert bins[0].lo == 0.0 and bins[-1].hi == 1.0
    assert bins[-1].n == 4 and bins[-1].accuracy == 1.0  # 0.91, 0.95, 0.99, 1.0
    assert bins[0].n == 2 and bins[0].accuracy == 0.0  # 0.0, 0.05
    q = reliability_table(y, p, n_bins=4, strategy="quantile")
    assert sum(b.n for b in q) == 10 and q[0].lo == 0.0 and q[-1].hi == 1.0
    with pytest.raises(ValueError):
        ece(y, p * 2)


# ----------------------------------------------------------------------------- verdict


def test_thresholds_meet_the_target_precisions_and_partition_the_confidences():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = rng.random(2000) < p  # perfectly calibrated
    thr = choose_thresholds(y, p, accept_precision=0.9, reject_precision=0.8)
    assert thr.accept_satisfiable and thr.reject_satisfiable
    assert 0.0 < thr.tau_reject < thr.tau_accept < 1.0
    rates = verdict_rates(y, p, thr)
    assert rates["accept_precision"] >= 0.9 and rates["reject_precision"] >= 0.8
    assert rates["accept_rate"] + rates["reject_rate"] + rates[
        "request_view_rate"
    ] == pytest.approx(1)
    assert rates["accept_rate"] > 0.1 and rates["reject_rate"] > 0.1
    # a calibrated model accepts above ~0.8 on average for 90 % precision
    assert 0.7 < thr.tau_accept < 0.9
    v = thr.verdicts(
        [0.0, thr.tau_reject, (thr.tau_reject + thr.tau_accept) / 2, thr.tau_accept, 1]
    )
    assert v == [
        Verdict.REJECT,
        Verdict.REJECT,
        Verdict.REQUEST_VIEW,
        Verdict.ACCEPT,
        Verdict.ACCEPT,
    ]
    again = VerdictThresholds.from_dict(thr.to_dict())
    assert again == thr


def test_unsatisfiable_target_leaves_the_band_empty():
    rng = np.random.default_rng(2)
    p = rng.random(300)
    y = rng.random(300) < 0.5  # confidence carries no information
    thr = choose_thresholds(y, p, accept_precision=0.99, reject_precision=0.99, min_band_rows=30)
    assert not thr.accept_satisfiable and thr.tau_accept == math.inf
    assert not thr.reject_satisfiable and thr.tau_reject == -math.inf
    rates = verdict_rates(y, p, thr)
    assert rates["request_view_rate"] == 1.0 and math.isnan(rates["accept_precision"])


def test_overlapping_bands_keep_the_acceptance_band_and_recheck_the_rejection_band():
    # bands overlap when the rows at one confidence (here 0.5, ten failures) are few enough to
    # be swallowed by both the ≥ 90 %-success suffix and the ≥ 90 %-failure prefix
    def case(low_successes: int) -> tuple[np.ndarray, np.ndarray]:
        p = np.concatenate(
            [np.repeat([0.1, 0.2, 0.3, 0.4, 0.5], 10), np.repeat([0.6, 0.7, 0.8, 0.9, 1.0], 20)]
        )
        y = np.concatenate(
            [
                np.array([True] * low_successes + [False] * (10 - low_successes)),
                np.zeros(40, dtype=bool),  # 0.2 … 0.5: failures
                np.ones(100, dtype=bool),  # 0.6 … 1.0: successes
            ]
        )
        return y, p

    y, p = case(2)
    thr = choose_thresholds(y, p, accept_precision=0.9, reject_precision=0.9, min_band_rows=10)
    # accept: {p ≥ 0.5} = 100 / 110 successes; reject: {p ≤ 0.5} = 48 / 50 failures → overlap at
    # 0.5, the acceptance band wins and the rejection band is pushed to 0.4 (38 / 40 failures)
    assert thr.tau_accept == pytest.approx(0.5) and thr.tau_reject == pytest.approx(0.4)
    assert thr.accept_satisfiable and thr.reject_satisfiable
    rates = verdict_rates(y, p, thr)
    assert rates["request_view_rate"] == 0.0 and rates["reject_precision"] == pytest.approx(0.95)

    y2, p2 = case(5)  # the pushed-down band is only 35 / 40 failures: flagged, not claimed
    thr2 = choose_thresholds(y2, p2, accept_precision=0.9, reject_precision=0.9, min_band_rows=10)
    assert thr2.tau_accept == pytest.approx(0.5) and thr2.tau_reject == pytest.approx(0.4)
    assert not thr2.reject_satisfiable
    assert verdict_rates(y2, p2, thr2)["reject_precision"] == pytest.approx(0.875)


def test_mlp_penalty_matches_logistic_regression():
    """A hidden-layer-free MLP with alpha = 1 / C is the logistic regression with C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = rng.random(400) < 1 / (1 + np.exp(-(X @ [1.5, -1.0, 0.5, 0.0])))
    for C in (0.1, 1.0):
        lr = LogisticRegression(C=C, max_iter=5000).fit(X, y)
        mlp = MLPClassifier(
            hidden_layer_sizes=(), alpha=1.0 / C, solver="lbfgs", max_iter=5000, random_state=0
        ).fit(X, y)
        np.testing.assert_allclose(mlp.coefs_[0][:, 0], lr.coef_[0], atol=0.05)


# ----------------------------------------------------------------------------- labels


def _hyp(view, gt, T, hypothesis_id=0, reason=None) -> PoseHypothesis:
    return PoseHypothesis(
        camera_id=view.camera_id,
        object_id=gt.object_id,
        detection_id=gt.gt_index,
        hypothesis_id=hypothesis_id,
        T_camera_object=T,
        stage=Stage.REFINED,
        signals=QualitySignals(seg_score=0.5),
        source="test",
        rejection_reason=reason,
    )


def test_hypothesis_labels_follow_mssd_against_the_nearest_gt(mini_bop: BopDataset, monkeypatch):
    view, gts = mini_bop.load_view(1, 0, load_rgb=False, load_depth=False)
    g0, g1, g5 = gts  # two copies of object 1 (continuous symmetry) and one object 5
    d1 = mini_bop.load_model(1).diameter
    recs = [
        HypothesisRecord(1, 0, _hyp(view, g0, g0.T_camera_object, 0)),  # exact
        HypothesisRecord(1, 0, _hyp(view, g0, g0.T_camera_object @ rotvec_T([0, 0, 1], 137), 1)),
        HypothesisRecord(1, 0, _hyp(view, g0, g0.T_camera_object @ rotvec_T([1, 0, 0], 60), 2)),
        HypothesisRecord(
            1, 0, _hyp(view, g1, g0.T_camera_object, 0)
        ),  # detection of the other copy
        HypothesisRecord(1, 0, _hyp(view, g1, g1.T_camera_object, 1, reason="fitness")),
    ]
    table = pd.DataFrame([hypothesis_to_row(r) for r in recs])
    # a hypothesis for an object that is not in the image (segmentation false positive)
    fp = hypothesis_to_row(HypothesisRecord(1, 0, _hyp(view, g5, g5.T_camera_object, 7)))
    table = pd.concat([table, pd.DataFrame([fp])], ignore_index=True)
    only_object_1 = [g for g in gts if g.object_id == 1]
    monkeypatch.setattr(mini_bop, "ground_truth", lambda sid, iid: only_object_1)
    lab = label_hypotheses(table, mini_bop)
    assert len(lab) == 6 and list(lab["success"]) == [True, True, False, True, True, False]
    assert lab.loc[0, "mssd_mm"] == pytest.approx(0.0, abs=1e-9)
    assert lab.loc[1, "mssd_mm"] < 0.02 * d1  # symmetry-aware: a 137° turn about the axis is free
    assert lab.loc[2, "mssd_mm"] > 0.1 * d1 and lab.loc[2, "r_err_deg"] == pytest.approx(60, abs=1)
    assert lab.loc[3, "gt_index"] == g0.gt_index  # matched to the nearest copy, not its detection
    assert lab.loc[4, "gt_index"] == g1.gt_index
    assert list(lab["diameter"]) == [d1] * 5 + [mini_bop.load_model(5).diameter]
    assert list(lab["rejected"]) == [0, 0, 0, 0, 1, 0]
    assert not lab.loc[5, "gt_present"] and lab.loc[5, "mssd_mm"] == math.inf
    assert lab.loc[5, "gt_index"] == -1 and math.isnan(lab.loc[5, "gt_visible_fraction"])
    assert lab.loc[0, "gt_valid"] and lab.loc[0, "gt_visible_fraction"] == g0.visible_fraction


def test_fused_labels_use_world_frame_ground_truth_of_every_group_view(mini_bop: BopDataset):
    v0, gts0 = mini_bop.load_view(1, 0, load_rgb=False, load_depth=False)
    v1, _ = mini_bop.load_view(1, 1, load_rgb=False, load_depth=False)
    g0, g5 = gts0[0], gts0[2]
    T_w0 = v0.T_world_camera @ g0.T_camera_object
    T_w5 = v1.T_world_camera @ (v1.T_camera_world @ v0.T_world_camera @ g5.T_camera_object)
    rows = []
    for tid, (oid, T, imgs) in enumerate(
        [
            (1, T_w0, "0,1"),
            (5, T_w5, "1"),
            (1, T_w0 @ rotvec_T([1, 0, 0], 90), "0,1"),
            (1, T_w0, ""),
        ]
    ):
        rows.append(
            {
                "scene_id": 1,
                "group_id": 0,
                "track_id": tid,
                "object_id": oid,
                "image_ids": imgs,
                **transform_to_columns(T),
            }
        )
    lab = label_fused(pd.DataFrame(rows), mini_bop)
    assert list(lab["success"]) == [True, True, False, False]
    assert lab.loc[0, "mssd_mm"] == pytest.approx(0.0, abs=1e-6) and lab.loc[0, "gt_image_id"] in (
        0,
        1,
    )
    assert lab.loc[1, "gt_index"] == g5.gt_index and lab.loc[1, "gt_image_id"] == 1
    assert lab.loc[2, "r_err_deg"] == pytest.approx(90, abs=1)
    assert not lab.loc[3, "gt_present"]  # no views, no ground truth


# ----------------------------------------------------------------------------- pipeline stage


def synthetic_fused(n: int, seed: int = 0) -> pd.DataFrame:
    """Fused-table rows: aggregated signals plus the Model F extras."""
    rng = np.random.default_rng(seed)
    df = synthetic_hypotheses(n, seed=seed)
    y = df["success"].to_numpy()
    df["n_views"] = rng.integers(1, 5, n).astype(float)
    df["dispersion_mm"] = np.where(y, rng.normal(1, 0.5, n), rng.normal(6, 3, n)).clip(0)
    df["dispersion_deg"] = np.where(y, rng.normal(1, 0.5, n), rng.normal(8, 4, n)).clip(0)
    df["n_members"] = df["n_views"]
    df["n_aligned"] = rng.integers(0, 2, n).astype(float)
    df["weight_sum"] = rng.uniform(0.1, 3, n)
    p = np.where(y, rng.beta(6, 2, n), rng.beta(2, 5, n))
    df["p_h_mean"] = p
    df["p_h_min"] = (p * rng.uniform(0.5, 1, n)).clip(0, 1)
    df["p_h_max"] = (p * rng.uniform(1, 1.5, n)).clip(0, 1)
    df["rejected"] = rng.random(n) < np.where(y, 0.05, 0.4)
    df["rejected"] = df["rejected"].astype(float)
    return df


def fit_test_models(out: Path, seed: int = 0) -> dict[str, Path]:
    """Model H / F and thresholds fitted on synthetic rows, saved as the stage expects them."""
    from binposert.confidence import ConfidenceModel

    h_rows = synthetic_hypotheses(600, seed=seed)
    f_rows = synthetic_fused(600, seed=seed + 1)
    model_h = ConfidenceModel.fit(FeatureSchema.hypothesis(), h_rows, h_rows["success"])
    model_f = ConfidenceModel.fit(FeatureSchema.fused(), f_rows, f_rows["success"])
    thr = choose_thresholds(f_rows["success"], model_f.predict_proba(f_rows), 0.9, 0.8)
    paths = {
        "model_h": out / "model_h.json",
        "model_f": out / "model_f.json",
        "thresholds": out / "thresholds.json",
    }
    model_h.save(paths["model_h"])
    model_f.save(paths["model_f"])
    thr.save(paths["thresholds"])
    return paths


def test_confidence_stage_scores_every_fused_pose_on_the_fixture(tmp_path):
    from hydra import compose, initialize_config_dir

    from binposert.pipeline.run import planned_refs, run

    REPO = Path(__file__).resolve().parents[1]
    paths = fit_test_models(tmp_path / "models")

    def _compose(*overrides: str):
        with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
            return compose(config_name="config", overrides=list(overrides))

    base = (
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "estimator.params.t_sigma_mm=4",
        "estimator.params.r_sigma_deg=6",
        "evaluate.with_vsd=false",
        "multiview.params.groups.n_views=2",
        f"confidence.params.model_h={paths['model_h']}",
        f"confidence.params.model_f={paths['model_f']}",
        f"confidence.params.thresholds={paths['thresholds']}",
    )
    stages = "stages=[segment,coarse_pose,associate,fuse,confidence]"
    with_eval = stages[:-1] + ",evaluate]"
    res = run(_compose(*base, with_eval, "evaluate.score_signal=confidence"), repo_root=REPO)
    conf_dir = res.refs["confidence"].dir
    fused = pd.read_parquet(conf_dir / "fused.parquet")
    assert len(fused) == 6 and fused["confidence"].between(0, 1).all()
    assert set(fused["verdict"]) <= {v.value for v in Verdict}
    assert fused[["p_h_mean", "p_h_min", "p_h_max", "rejected"]].notna().all().all()
    assert (fused.p_h_min <= fused.p_h_mean).all() and (fused.p_h_mean <= fused.p_h_max).all()
    hyps = pd.read_parquet(conf_dir / "hypotheses.parquet")
    assert len(hyps) == 12 and hyps["confidence"].notna().all()
    # every projection of a track carries the track's Confidence
    per_track = hyps.groupby(["scene_id", "hypothesis_id"])["confidence"].nunique()
    assert (per_track == 1).all()
    summary = json.loads((conf_dir / "confidence_summary.json").read_text())
    assert summary["scored"] == "fused" and summary["n_fused"] == 6
    assert sum(summary["verdict_rates"].values()) == pytest.approx(1.0)
    rep = json.loads((res.refs["evaluate"].dir / "report.json").read_text())
    assert rep["n_gt"] == 12 and rep["n_predictions"] == 12

    # a refitted model file changes the stage hash (the config only names the path)
    before = planned_refs(_compose(*base, stages), repo_root=REPO)["confidence"].hash
    fit_test_models(tmp_path / "models", seed=7)
    after = planned_refs(_compose(*base, stages), repo_root=REPO)["confidence"].hash
    assert before != after
    refit = planned_refs(_compose(*base, stages), repo_root=REPO)
    assert refit["fuse"].hash == res.refs["fuse"].hash  # upstream untouched

    # single-view pass-through rows are scored by Model H
    none = run(_compose(*base, stages, "fusion=none"), repo_root=REPO)
    s_none = json.loads((none.refs["confidence"].dir / "confidence_summary.json").read_text())
    assert s_none["scored"] == "hypotheses" and s_none["n_hypotheses"] == 12
    h_none = pd.read_parquet(none.refs["confidence"].dir / "hypotheses.parquet")
    assert h_none["confidence"].between(0, 1).all() and h_none["verdict"].notna().all()

    # Model H as the fusion weight: a different association, weights in [floor, 1]
    weighted = run(
        _compose(
            *base,
            stages,
            "+multiview.params.weights.source=model_h",
            f"+multiview.params.weights.model_h={paths['model_h']}",
        ),
        repo_root=REPO,
    )
    assert weighted.refs["associate"].hash != res.refs["associate"].hash
    tracks = pd.read_parquet(weighted.refs["associate"].dir / "tracks.parquet")
    assert tracks["weight"].between(1e-3, 1.0).all() and tracks["weight"].nunique() > 1
    plain = pd.read_parquet(res.refs["associate"].dir / "tracks.parquet")
    assert not np.allclose(tracks["weight"].to_numpy(), plain["weight"].to_numpy())
