import numpy as np
import pytest

from binposert import transforms as tf
from binposert.symmetry import (
    align_to_reference,
    from_bop_model_info,
    sym_aware_rotation_distance_deg,
)
from binposert.types import SymmetryGroup
from tests.synth import box_model, cylinder_model


def test_group_always_starts_with_identity():
    g = from_bop_model_info({})
    assert g.is_trivial
    with pytest.raises(ValueError):
        SymmetryGroup((tf.rotvec_T([0, 0, 1], 90.0),))


def test_continuous_axis_is_sampled_and_closed_with_discrete():
    cyl = cylinder_model(steps=36, flip=True)
    assert len(cyl.symmetry) == 72  # 36 axial x 2 (flip)
    cyl_noflip = cylinder_model(steps=12, flip=False)
    assert len(cyl_noflip.symmetry) == 12


def test_cylinder_rotated_137deg_about_axis_has_zero_error():
    cyl = cylinder_model(steps=36)
    T_ref = tf.rotvec_T([1, 0.3, 0.2], 40.0, t=[10, 20, 300])
    T_rot = T_ref @ tf.rotvec_T([0, 0, 1], 137.0)
    naive = tf.rotation_distance_deg(T_ref, T_rot)
    assert naive > 100
    # 137 is not a multiple of the 10 deg step; residual must be <= half a step
    assert sym_aware_rotation_distance_deg(T_ref, T_rot, cyl.symmetry) <= 5.0 + 1e-6
    T_aligned, idx = align_to_reference(T_ref, T_rot, cyl.symmetry)
    assert idx != 0
    assert tf.rotation_distance_deg(T_ref, T_aligned) <= 5.0 + 1e-6
    np.testing.assert_allclose(T_aligned[:3, 3], T_ref[:3, 3])


def test_discrete_flip_is_aligned_exactly():
    box = box_model(symmetric=True)
    assert len(box.symmetry) == 4
    T_ref = tf.rotvec_T([0, 1, 0], 25.0, t=[0, 0, 400])
    T_flip = T_ref @ tf.rotvec_T([0, 0, 1], 180.0)
    assert tf.rotation_distance_deg(T_ref, T_flip) == pytest.approx(180.0)
    assert sym_aware_rotation_distance_deg(T_ref, T_flip, box.symmetry) == pytest.approx(
        0.0, abs=1e-9
    )
    T_aligned, _ = align_to_reference(T_ref, T_flip, box.symmetry)
    np.testing.assert_allclose(T_aligned, T_ref, atol=1e-9)


def test_alignment_is_right_multiplication_in_object_frame():
    """Symmetry acts before T: a symmetric pose must keep the same translation."""
    cyl = cylinder_model(steps=8)
    T = tf.rotvec_T([1, 1, 0], 70.0, t=[50, -20, 500])
    for S in cyl.symmetry.transforms:
        np.testing.assert_allclose((T @ S)[:3, 3], T[:3, 3], atol=1e-9)


def test_offset_axis_symmetry_keeps_offset_point_fixed():
    info = {"symmetries_continuous": [{"axis": [0, 0, 1], "offset": [5.0, 0.0, 0.0]}]}
    g = from_bop_model_info(info, continuous_steps=4)
    for S in g.transforms:
        np.testing.assert_allclose(
            tf.transform_points(S, [[5.0, 0.0, 7.0]]), [[5.0, 0.0, 7.0]], atol=1e-9
        )
