"""Refinement recovers known truth (D16): small perturbations are corrected, large ones and
converged-but-wrong registrations are rejected by the independent gate."""

import numpy as np
import pytest

from binposert import transforms as tf
from binposert.refine import (
    GateParams,
    Refiner,
    RefinerParams,
    boundary_error_px,
    check_gate,
    register,
    scene_cloud,
    silhouette_iou,
    translation_from_depth,
    visible_model_cloud,
    visible_silhouette,
)
from binposert.render import MeshRenderer
from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.types import PoseHypothesis, QualitySignals, Stage
from tests.synth import box_model, cylinder_model, render_view, simple_K

K = simple_K()
SIZE = (240, 320)


def _hyp(T, object_id=1, detection_id=0):
    return PoseHypothesis("000000", object_id, detection_id, 0, T, Stage.COARSE, QualitySignals())


def _perturb(T_gt, axis, deg, t):
    """Perturb about the object's own origin (about the camera it would move by centimetres)."""
    return T_gt @ tf.rotvec_T(axis, deg, t=t)


def _errors(T, T_gt, model):
    return tf.translation_distance(T, T_gt), sym_aware_rotation_distance_deg(
        T, T_gt, model.symmetry
    )


# ----------------------------------------------------------------------------- building blocks


def test_scene_cloud_and_visible_model_cloud_agree_at_the_true_pose():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 0.3, 0], 35.0, t=[10, -5, 400])
    view, masks = render_view([(box, T_gt)], K, SIZE)
    pts_cam, normals_cam, coverage = scene_cloud(view.depth, K, masks[0], erode_px=0)
    assert coverage == pytest.approx(1.0) and len(pts_cam) > 500
    assert np.allclose(np.linalg.norm(normals_cam, axis=1), 1.0)
    assert (np.sum(normals_cam * -pts_cam, axis=1) > 0).all()  # oriented towards the camera
    pts_obj, normals_obj, n_px = visible_model_cloud(MeshRenderer.from_model(box), T_gt, K, SIZE)
    assert n_px == int(masks[0].sum())
    # every visible model point (in camera frame) lies on the observed surface
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts_cam).query(tf.transform_points(T_gt, pts_obj))
    assert np.median(d) < 1.5  # pixel spacing at 400 mm with f = 400 is 1 mm


@pytest.mark.parametrize("variant", ["point_to_plane", "robust", "gicp"])
def test_register_recovers_small_perturbation(variant):
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 1, 0.3], 35.0, t=[0, 0, 400])  # three faces visible
    view, masks = render_view([(box, T_gt)], K, SIZE)
    pts_cam, normals_cam, _ = scene_cloud(view.depth, K, masks[0], erode_px=1)
    T_init = _perturb(T_gt, [1, 1, 0], 4.0, [3, -2, 4])
    pts_obj, normals_obj, _ = visible_model_cloud(MeshRenderer.from_model(box), T_init, K, SIZE)
    reg = register(pts_obj, normals_obj, pts_cam, normals_cam, T_init, variant, max_corr_dist_mm=12)
    assert reg.converged and reg.fitness > 0.8
    dt, dr = _errors(reg.T_camera_object, T_gt, box)
    # GICP's covariance weighting lets planar box faces slide a little; the others are sub-mm
    assert dt < (2.0 if variant == "gicp" else 1.0) and dr < 1.0, (variant, dt, dr)


# ----------------------------------------------------------------------------- the stage


@pytest.mark.parametrize("variant", ["point_to_plane", "robust", "gicp"])
def test_refiner_recovers_small_perturbation_and_fills_signals(variant):
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([0.5, 1, 0.2], 30.0, t=[15, 5, 420])
    view, masks = render_view([(box, T_gt)], K, SIZE)
    T_coarse = _perturb(T_gt, [0, 0, 1], 6.0, [4, 3, -6])
    dt0, dr0 = _errors(T_coarse, T_gt, box)
    assert dt0 > 5 and dr0 > 5
    out = Refiner(box, RefinerParams(variant=variant)).refine(view, masks[0], _hyp(T_coarse))
    h = out.hypothesis
    assert h.stage is Stage.REFINED and h.rejection_reason is None and out.gate.accepted
    dt, dr = _errors(h.T_camera_object, T_gt, box)
    assert dt < (2.0 if variant == "gicp" else 1.0) and dr < 1.0, (variant, dt, dr)
    s = h.signals
    assert s.icp_fitness > 0.8 and s.icp_rmse_mm < 2.0 and s.depth_coverage == pytest.approx(1.0)
    # the displacement is ICP's move from the depth-initialised start pose
    T_start = T_coarse.copy()
    T_start[:3, 3] *= (T_coarse[2, 3] + out.z_shift_mm) / T_coarse[2, 3]
    assert s.silhouette_iou > 0.9 and out.z_init_overlap > 100
    assert s.displacement_mm == pytest.approx(
        tf.translation_distance(T_start, h.T_camera_object), abs=1e-6
    )
    assert s.visible_fraction > 0.9 and h.source.endswith("+" + variant)


@pytest.mark.parametrize("noise_mm", [0.0, 1.0, 2.0])
def test_noise_sweep_degrades_gracefully(noise_mm):
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 0.7, 0.2], 30.0, t=[0, 0, 400])
    view, masks = render_view([(box, T_gt)], K, SIZE, depth_noise_mm=noise_mm, dropout=0.05, seed=3)
    T_coarse = _perturb(T_gt, [0, 1, 0], 5.0, [5, 0, 5])
    out = Refiner(box, RefinerParams(variant="robust")).refine(view, masks[0], _hyp(T_coarse))
    assert out.hypothesis.rejection_reason is None
    dt, dr = _errors(out.hypothesis.T_camera_object, T_gt, box)
    assert dt < 1.0 + noise_mm and dr < 1.5 + noise_mm


def test_degenerate_two_face_view_falls_back_to_point_to_point():
    """Only two box faces visible: point-to-plane has a null space along the hidden axis and
    Open3D returns an absurd translation; register() must detect it and fall back."""
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([0, 1, 0], 25.0, t=[0, 0, 400])
    view, masks = render_view([(box, T_gt)], K, SIZE)
    pts_cam, normals_cam, _ = scene_cloud(view.depth, K, masks[0], erode_px=1)
    T_init = _perturb(T_gt, [1, 1, 0], 4.0, [3, -2, 4])
    pts_obj, normals_obj, _ = visible_model_cloud(MeshRenderer.from_model(box), T_init, K, SIZE)
    reg = register(pts_obj, normals_obj, pts_cam, normals_cam, T_init, "point_to_plane", 12.0)
    assert reg.converged and reg.fitness > 0.9
    dt, dr = _errors(reg.T_camera_object, T_gt, box)
    assert dt < 3.0 and dr < 1.5  # y stays unobservable, but nothing explodes
    # the raw plane solve is what fails here (Open3D returns a ~1e16 translation), so either the
    # fallback ran or the solver happened to stay put — both are acceptable; explosion is not.


def test_depth_initialisation_recovers_a_coarse_pose_that_is_far_off_along_the_ray():
    """An RGB coarse pose that is right in the image but 0.8 d too far: beyond the z window, the
    ICP search radius and the displacement cap. The depth initialisation brings it back."""
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 0.3, 0], 35.0, t=[20, -10, 400])
    view, masks = render_view([(box, T_gt)], K, SIZE)
    T_coarse = T_gt.copy()
    T_coarse[:3, 3] = T_gt[:3, 3] * (400 + 0.8 * box.diameter) / 400  # same ray, farther
    render_c = MeshRenderer.from_model(box).render(T_coarse, K, SIZE)
    T_init, dz, n = translation_from_depth(render_c, view.depth, masks[0], T_coarse)
    assert n > 100 and dz == pytest.approx(-0.8 * box.diameter, abs=0.1 * box.diameter)
    assert tf.translation_distance(T_init, T_gt) < 0.1 * box.diameter
    out = Refiner(box).refine(view, masks[0], _hyp(T_coarse))
    assert out.hypothesis.rejection_reason is None and out.z_shift_mm == pytest.approx(dz)
    t_err, r_err = _errors(out.hypothesis.T_camera_object, T_gt, box)
    assert t_err < 1.0 and r_err < 1.0
    # the same coarse pose without the initialisation cannot be refined locally
    off = Refiner(box, RefinerParams(z_init="none")).refine(view, masks[0], _hyp(T_coarse))
    assert off.hypothesis.rejection_reason in ("depth_window", "icp_failed", "no_overlap")
    assert off.z_shift_mm == 0.0
    np.testing.assert_array_equal(off.hypothesis.T_camera_object, T_coarse)


def test_depth_initialisation_needs_overlap_and_a_positive_depth():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([0, 1, 0], 20.0, t=[0, 0, 400])
    view, masks = render_view([(box, T_gt)], K, SIZE)
    renderer = MeshRenderer.from_model(box)
    T_far = tf.make_T(T_gt[:3, :3], T_gt[:3, 3] + [3 * box.diameter, 0, 0])  # no overlap
    T_out, dz, n = translation_from_depth(
        renderer.render(T_far, K, SIZE), view.depth, masks[0], T_far
    )
    assert n < 50 and dz == 0.0
    np.testing.assert_array_equal(T_out, T_far)
    T_same, dz, n = translation_from_depth(
        renderer.render(T_gt, K, SIZE), view.depth, masks[0], T_gt
    )
    assert abs(dz) < 0.5 and n > 100 and np.allclose(T_same, T_gt, atol=0.5)


def test_large_perturbation_is_rejected_and_coarse_pose_returned():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 1, 0.3], 35.0, t=[0, 0, 400])
    view, masks = render_view([(box, T_gt)], K, SIZE)
    T_coarse = tf.make_T(T_gt[:3, :3], T_gt[:3, 3] + [0.6 * box.diameter, 0, 0])
    out = Refiner(box).refine(view, masks[0], _hyp(T_coarse))
    h = out.hypothesis
    assert h.rejection_reason is not None and h.stage is Stage.REFINED
    np.testing.assert_array_equal(h.T_camera_object, T_coarse)  # unchanged
    assert np.isfinite(h.signals.icp_fitness)  # signals are stored even for rejections


def _two_boxes():
    box = box_model(symmetric=False, extents=(40.0, 30.0, 20.0))
    R = tf.rotvec_T([1, 1, 0.3], 35.0)[:3, :3]
    T_a = tf.make_T(R, [-25, 0, 420])
    T_b = tf.make_T(R, [25, 0, 420])  # 50 mm apart, same orientation
    view, masks = render_view([(box, T_a), (box, T_b)], K, SIZE)
    return box, T_a, T_b, view, masks


def test_masked_scene_cloud_prevents_registration_onto_a_neighbour():
    """Two identical boxes side by side; the Detection is box A, the coarse pose sits on box B.
    Because only the masked scene surface is registered, ICP finds nothing to converge onto and
    the coarse pose comes back untouched — the wrong object is never 'refined'."""
    box, T_a, T_b, view, masks = _two_boxes()
    T_coarse = tf.make_T(T_b[:3, :3], T_b[:3, 3] + [2, -1, 3])
    lenient = RefinerParams(gate=GateParams(alpha=10.0, beta_deg=180.0, min_iou=0.5))
    out = Refiner(box, lenient).refine(view, masks[0], _hyp(T_coarse))
    assert out.hypothesis.rejection_reason in ("icp_failed", "no_overlap")
    np.testing.assert_array_equal(out.hypothesis.T_camera_object, T_coarse)


def test_gate_rejects_a_converged_pose_whose_silhouette_is_wrong():
    """A registration that converged (fitness 1.0) onto the neighbouring box B while the Detection
    is box A: the silhouette check rejects it even with the displacement caps switched off, and
    the displacement cap catches it first with the D8 defaults."""
    box, T_a, T_b, view, masks = _two_boxes()
    ren = MeshRenderer.from_model(box)
    T_coarse = tf.make_T(T_a[:3, :3], T_a[:3, 3] + [1, 1, 2])
    r_c = ren.render(T_coarse, K, SIZE)
    r_r = ren.render(T_b, K, SIZE)
    lenient = GateParams(alpha=10.0, beta_deg=180.0, min_iou=0.5)
    d = check_gate(T_coarse, T_b, box, masks[0], r_c, r_r, 1.0, lenient, view.depth)
    assert not d.accepted and d.reason == "silhouette_iou" and d.iou_refined < 0.05
    assert d.iou_coarse > 0.7 and d.boundary_px > 20
    d2 = check_gate(T_coarse, T_b, box, masks[0], r_c, r_r, 1.0, GateParams(), view.depth)
    assert d2.reason == "displacement_translation" and d2.displacement_mm == pytest.approx(
        50, abs=3
    )


def test_rotation_cap_is_symmetry_aware_on_the_cylinder():
    cyl = cylinder_model(steps=36, flip=False)
    T_coarse = tf.rotvec_T([1, 0, 0], 40.0, t=[0, 0, 400])
    T_refined = T_coarse @ tf.rotvec_T([0, 0, 1], 137.0)  # spin about the symmetry axis
    view, masks = render_view([(cyl, T_coarse)], K, SIZE)
    ren = MeshRenderer.from_model(cyl)
    r_c = ren.render(T_coarse, K, SIZE)
    r_r = ren.render(T_refined, K, SIZE)
    assert silhouette_iou(r_c.mask, r_r.mask) > 0.97 and boundary_error_px(r_c.mask, r_r.mask) < 1.0
    ok = check_gate(T_coarse, T_refined, cyl, masks[0], r_c, r_r, fitness=1.0, params=GateParams())
    assert ok.accepted and ok.displacement_deg < 5.0  # 137° lands 3° off the 10° symmetry grid
    trivial = cylinder_model(steps=36, flip=False)
    trivial = type(trivial)(**{**trivial.__dict__, "symmetry": trivial.symmetry.trivial()})
    bad = check_gate(T_coarse, T_refined, trivial, masks[0], r_c, r_r, 1.0, GateParams())
    assert not bad.accepted and bad.reason == "displacement_rotation"
    assert bad.displacement_deg == pytest.approx(137.0, abs=0.1)


def test_visible_silhouette_ignores_pixels_hidden_behind_an_occluder():
    """A box half hidden behind another: at the *true* pose the occlusion-aware silhouette matches
    the visible mask, whereas the raw full silhouette does not."""
    box = box_model(symmetric=False, extents=(40.0, 30.0, 20.0))
    T_back = tf.make_T(np.eye(3), [0, 0, 440])
    T_front = tf.make_T(np.eye(3), [-22, 0, 400])  # covers the left part of the back box
    view, masks = render_view([(box, T_back), (box, T_front)], K, SIZE)
    assert 0.3 < masks[0].sum() / masks[1].sum() < 0.9
    r = MeshRenderer.from_model(box).render(T_back, K, SIZE)
    raw = silhouette_iou(r.mask, masks[0])
    aware = silhouette_iou(visible_silhouette(r, view.depth, 15.0), masks[0])
    assert raw < 0.75 and aware > 0.98


def test_gate_metrics_on_masks():
    a = np.zeros((20, 20), bool)
    a[5:15, 5:15] = True
    b = np.roll(a, 2, axis=1)
    assert silhouette_iou(a, a) == 1.0 and 0.6 < silhouette_iou(a, b) < 0.7
    assert boundary_error_px(a, a) == 0.0 and 0.5 < boundary_error_px(a, b) < 2.0
    assert boundary_error_px(a, np.zeros_like(a)) == float("inf")
