import numpy as np
import pytest

from binposert import transforms as tf
from binposert.data import BopDataset
from binposert.evaluate import (
    PosePrediction,
    add,
    adi,
    evaluate_localisation,
    mspd,
    mssd,
    read_bop_csv,
    vsd,
    write_bop_csv,
)
from binposert.render import MeshRenderer
from tests.synth import box_model, cylinder_model, simple_K


def _gt_predictions(ds: BopDataset, perturb=None, score=1.0):
    preds = []
    for s in ds.scene_ids:
        for i in ds.image_ids(s):
            _, gts = ds.load_view(s, i, load_rgb=False, load_depth=False)
            for g in gts:
                T = g.T_camera_object if perturb is None else g.T_camera_object @ perturb(g)
                preds.append(PosePrediction(s, i, g.object_id, score, T))
    return preds


def test_add_and_adi_on_translation():
    box = box_model()
    pts = box.sample_points(500)
    T_gt = tf.make_T(np.eye(3), [0, 0, 400])
    T_est = tf.make_T(np.eye(3), [3, 4, 400])
    assert add(T_est, T_gt, pts) == pytest.approx(5.0)
    assert adi(T_est, T_gt, pts) <= 5.0 + 1e-9


def test_mssd_is_symmetry_aware_and_mspd_scales_with_focal():
    cyl = cylinder_model(steps=36)
    pts = cyl.sample_points(500)
    T_gt = tf.rotvec_T([1, 0, 0], 30.0, t=[0, 0, 400])
    T_sym = T_gt @ tf.rotvec_T([0, 0, 1], 90.0)  # exactly on the symmetry grid
    assert mssd(T_sym, T_gt, pts, cyl.symmetry) == pytest.approx(0.0, abs=1e-9)
    assert mssd(T_sym, T_gt, pts, cyl.symmetry.trivial()) > 10.0
    T_shift = tf.make_T(T_gt[:3, :3], T_gt[:3, 3] + [2, 0, 0])
    assert mssd(T_shift, T_gt, pts, cyl.symmetry) == pytest.approx(2.0, abs=1e-6)
    K1 = simple_K(f=400.0)
    K2 = simple_K(f=800.0)
    assert mspd(T_shift, T_gt, pts, cyl.symmetry, K2) == pytest.approx(
        2 * mspd(T_shift, T_gt, pts, cyl.symmetry, K1), rel=1e-6
    )


def test_vsd_zero_for_identical_pose_and_grows_with_error():
    box = box_model(symmetric=False)
    K = simple_K()
    ren = MeshRenderer.from_model(box)
    T_gt = tf.rotvec_T([0, 1, 0], 20.0, t=[0, 0, 400])
    depth = ren.render(T_gt, K, (240, 320)).depth
    e0 = vsd(T_gt, T_gt, ren, depth, K, tau_mm=[5.0, 20.0])
    np.testing.assert_allclose(e0, 0.0)
    T_bad = tf.make_T(T_gt[:3, :3], T_gt[:3, 3] + [40, 0, 0])
    e1 = vsd(T_bad, T_gt, ren, depth, K, tau_mm=[5.0, 20.0])
    assert e1[0] > 0.3
    assert e1[0] >= e1[1]  # looser tau never increases the error


def test_bop_csv_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    preds = [PosePrediction(1, 2, 5, 0.75, tf.random_rigid(rng), 0.123) for _ in range(3)]
    path = tmp_path / "res.csv"
    write_bop_csv(path, preds)
    back = read_bop_csv(path)
    assert path.read_text().splitlines()[0] == "scene_id,im_id,obj_id,score,R,t,time"
    for p, q in zip(preds, back, strict=True):
        assert (p.scene_id, p.image_id, p.object_id) == (q.scene_id, q.image_id, q.object_id)
        assert q.score == pytest.approx(0.75) and q.time_s == pytest.approx(0.123)
        np.testing.assert_allclose(p.T_camera_object, q.T_camera_object, atol=1e-6)


def test_ground_truth_as_prediction_scores_perfect_ar(mini_bop: BopDataset):
    report = evaluate_localisation(_gt_predictions(mini_bop), mini_bop, n_model_points=500)
    assert report.n_gt == 12
    assert report.ar_mssd == pytest.approx(1.0)
    assert report.ar_mspd == pytest.approx(1.0)
    assert report.ar_vsd == pytest.approx(1.0)
    assert report.ar == pytest.approx(1.0)
    assert set(report.per_object) == {1, 5}
    assert all(r["success_0.1d"] == 1.0 for r in report.rows)


def test_symmetric_equivalent_predictions_still_score_perfect(mini_bop: BopDataset):
    def perturb(g):
        return tf.rotvec_T([0, 0, 1], 180.0)  # symmetry of obj 1 (axis) and obj 5 (180 deg flip)

    report = evaluate_localisation(
        _gt_predictions(mini_bop, perturb), mini_bop, n_model_points=500, with_vsd=False
    )
    assert report.ar_mssd == pytest.approx(1.0)
    assert report.ar_mspd == pytest.approx(1.0)
    assert np.isnan(report.ar_vsd)


def test_large_errors_and_missing_predictions_lower_recall(mini_bop: BopDataset):
    def perturb(g):
        return tf.make_T(np.eye(3), [200.0, 0, 0])

    bad = evaluate_localisation(
        _gt_predictions(mini_bop, perturb), mini_bop, n_model_points=300, with_vsd=False
    )
    assert bad.ar_mssd == pytest.approx(0.0)
    half = [p for p in _gt_predictions(mini_bop) if p.object_id == 5]
    partial = evaluate_localisation(half, mini_bop, n_model_points=300, with_vsd=False)
    # BOP19 convention (eval_bop19_pose): recall pooled over all valid GT -> 4 obj-5 of 12
    assert partial.ar_mssd == pytest.approx(4 / 12)
    assert partial.per_object[5]["ar_mssd"] == pytest.approx(1.0)
    assert partial.per_object[1]["ar_mssd"] == pytest.approx(0.0)
