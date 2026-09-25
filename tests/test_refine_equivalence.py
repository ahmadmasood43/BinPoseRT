"""Phase 3 CI twin: ROI crop equivalence and Class-A bit-identity checks (< 5 s, no GPU/network).

Covers:
- ``Roi.K_shifted``: unproject_depth with shifted K gives identical 3D points to full-frame
- ``Roi.from_mask_and_sphere``: ROI contains all mask pixels
- ``Refiner.refine`` with ``roi="bbox"`` vs ``roi="none"``: transform agrees within 0.01 mm / 0.1 deg
- F5/F5b: ``sym_aware_rotation_distance_deg`` / ``align_to_reference`` vectorised batch == scalar loop
- F7: normals dot product restricted to hit pixels is bit-identical
"""
from __future__ import annotations

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from binposert import transforms as tf
from binposert.refine import (
    GateParams,
    Refiner,
    RefinerParams,
    scene_cloud,
    visible_model_cloud,
)
from binposert.refine.roi import Roi
from binposert.render import MeshRenderer, unproject_depth
from binposert.symmetry.group import align_to_reference, sym_aware_rotation_distance_deg
from binposert.types import PoseHypothesis, QualitySignals, Stage, SymmetryGroup
from tests.synth import box_model, cylinder_model, render_view, simple_K

K = simple_K()
SIZE = (240, 320)


def _hyp(T, object_id=1, detection_id=0):
    return PoseHypothesis("000000", object_id, detection_id, 0, T, Stage.COARSE, QualitySignals())


# ---------------------------------------------------------------------------
# Roi geometry
# ---------------------------------------------------------------------------

def test_K_shifted_unproject_identical():
    """Unprojecting ROI pixels with K_shifted gives the same 3D points as full-frame."""
    model = box_model()
    T_gt = tf.make_T(np.eye(3), [0, 0, 350])
    view, masks = render_view([(model, T_gt)], K, SIZE)
    mask = masks[0]

    roi = Roi.from_mask_and_sphere(mask, T_gt, K, model.diameter, SIZE, margin_px=8)
    K_roi = roi.K_shifted(K)
    depth_roi = roi.crop(view.depth)
    mask_roi = roi.crop(mask)

    pts_full = unproject_depth(view.depth, K, mask)
    pts_roi = unproject_depth(depth_roi, K_roi, mask_roi)

    assert pts_full.shape == pts_roi.shape, "same number of valid pixels"
    np.testing.assert_array_equal(pts_full, pts_roi, err_msg="K_shifted must give identical 3D pts")


def test_roi_contains_all_mask_pixels():
    """ROI must contain every True pixel of the detection mask."""
    model = box_model()
    T_gt = tf.make_T(np.eye(3), [20, -10, 300])
    view, masks = render_view([(model, T_gt)], K, SIZE)
    mask = masks[0]

    roi = Roi.from_mask_and_sphere(
        mask, T_gt, K, model.diameter, SIZE,
        dilate_px=3 + 2 + 1, margin_px=8,
    )
    mask_roi = roi.crop(mask)
    # Every True pixel in the full mask must appear as True in the cropped mask
    assert int(mask_roi.sum()) == int(mask.sum()), "ROI must not clip any mask pixel"


def test_roi_full_frame_when_empty_mask():
    """Empty mask falls back to full frame."""
    mask = np.zeros((240, 320), dtype=bool)
    T = tf.make_T(np.eye(3), [0, 0, 300])
    roi = Roi.from_mask_and_sphere(mask, T, K, 70.0, (240, 320))
    assert roi.size == (240, 320)


# ---------------------------------------------------------------------------
# Refiner ROI vs full-frame equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roi_mode", ["none", "bbox"])
def test_refiner_roi_close_to_full(roi_mode):
    """roi='bbox' produces a refined pose within 0.01 mm / 0.1 deg of roi='none'."""
    model = box_model()
    T_gt = tf.make_T(np.eye(3), [5, -3, 350])
    view, masks = render_view([(model, T_gt)], K, SIZE, depth_noise_mm=0.5)
    mask = masks[0]

    perturb = T_gt @ tf.rotvec_T([0, 1, 0], 10.0)
    hyp = _hyp(perturb)

    p_none = RefinerParams(roi="none")
    p_bbox = RefinerParams(roi="bbox")

    out_none = Refiner(model, p_none).refine(view, mask, hyp)
    out_bbox = Refiner(model, p_bbox).refine(view, mask, hyp)

    # Both should accept (or both reject with same reason) on an easy case
    assert out_none.hypothesis.rejection_reason == out_bbox.hypothesis.rejection_reason, (
        f"acceptance mismatch: none={out_none.hypothesis.rejection_reason!r} "
        f"bbox={out_bbox.hypothesis.rejection_reason!r}"
    )
    dt = tf.translation_distance(out_none.T_candidate, out_bbox.T_candidate)
    dr = sym_aware_rotation_distance_deg(out_none.T_candidate, out_bbox.T_candidate, model.symmetry)
    assert dt < 0.01, f"|Δt| {dt:.4f} mm exceeds 0.01 mm tolerance"
    assert dr < 0.1, f"|ΔR| {dr:.4f} deg exceeds 0.1 deg tolerance"


# ---------------------------------------------------------------------------
# F5 / F5b: vectorised symmetry == scalar loop (scipy batch path identical)
# ---------------------------------------------------------------------------

def test_sym_aware_rotation_distance_matches_scalar():
    """Vectorised sym_aware_rotation_distance_deg matches the scalar loop result."""
    model = cylinder_model()
    group = model.symmetry
    rng = np.random.default_rng(42)
    for _ in range(20):
        T_a = tf.make_T(Rotation.random(random_state=rng).as_matrix(), rng.uniform(-100, 100, 3))
        T_b = tf.make_T(Rotation.random(random_state=rng).as_matrix(), rng.uniform(-100, 100, 3))
        expected = min(tf.rotation_distance_deg(T_a, T_b @ S) for S in group.transforms)
        got = sym_aware_rotation_distance_deg(T_a, T_b, group)
        assert abs(got - expected) < 1e-9, f"expected {expected}, got {got}"


def test_align_to_reference_matches_scalar():
    """Vectorised align_to_reference matches the scalar loop result."""
    from binposert.transforms import se3_distance

    model = cylinder_model()
    group = model.symmetry
    rng = np.random.default_rng(7)
    for _ in range(10):
        T_ref = tf.make_T(Rotation.random(random_state=rng).as_matrix(), rng.uniform(-50, 50, 3))
        T = tf.make_T(Rotation.random(random_state=rng).as_matrix(), rng.uniform(-50, 50, 3))

        best_T_scalar, best_i_scalar = T, 0
        best_d = np.inf
        for i, S in enumerate(group.transforms):
            TS = T @ S
            d = se3_distance(T_ref, TS)
            if d < best_d:
                best_T_scalar, best_i_scalar, best_d = TS, i, d

        best_T_vec, best_i_vec = align_to_reference(T_ref, T, group)
        assert best_i_vec == best_i_scalar, "vectorised index differs from scalar"
        np.testing.assert_allclose(best_T_vec, best_T_scalar, atol=1e-10)


# ---------------------------------------------------------------------------
# F7: normals restricted to hit pixels is bit-identical to full-frame computation
# ---------------------------------------------------------------------------

def test_f7_normals_bit_identical():
    """Normals computed on hit-only pixels give the same result as the full-frame path."""
    model = box_model()
    T = tf.make_T(np.eye(3), [0, 0, 300])
    renderer = MeshRenderer.from_model(model)
    result = renderer.render(T, K, SIZE)

    # Manually replicate the original full-frame path (before F7) to compare
    from binposert.render.raycast import _cast_checked, _pinhole_rays, NO_HIT  # noqa: F401
    import open3d as o3d

    rays = _pinhole_rays(K, T, SIZE[1], SIZE[0])
    t_hit, prim, _, normals_obj = _cast_checked(renderer._scene, rays, SIZE, renderer.n_faces, 1)
    hit = prim != NO_HIT
    R = T[:3, :3]
    dirs_obj = rays.numpy()[..., 3:6].astype(np.float64)
    dirs_cam = dirs_obj @ R.T
    normals_cam_orig = normals_obj @ R.T
    # original full-frame flip (pre-F7)
    flip_full = np.sum(normals_cam_orig * dirs_cam, axis=-1) > 0
    normals_cam_pre_f7 = normals_cam_orig.copy()
    normals_cam_pre_f7[flip_full] *= -1.0
    normals_cam_pre_f7[~hit] = 0.0

    # F7 hit-restricted flip (post-F7, what the production code does)
    normals_cam_f7 = normals_cam_orig.copy()
    flip_f7 = np.zeros(hit.shape, dtype=bool)
    flip_f7[hit] = (normals_cam_f7[hit] * dirs_cam[hit]).sum(-1) > 0
    normals_cam_f7[flip_f7] *= -1.0
    normals_cam_f7[~hit] = 0.0

    np.testing.assert_array_equal(normals_cam_pre_f7, normals_cam_f7,
                                   err_msg="F7 normals must be bit-identical to full-frame path")
    # Also confirm it matches the live result
    np.testing.assert_array_equal(result.normals_camera, normals_cam_f7,
                                   err_msg="renderer.render must use the F7 normals")
