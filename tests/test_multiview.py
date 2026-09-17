"""Multi-view association, fusion and joint polish recover known truth (D16)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from binposert import transforms as tf
from binposert.multiview import (
    AssociationParams,
    FusionParams,
    JointIcpParams,
    JointRefiner,
    WeightParams,
    WorldHypothesis,
    associate,
    fuse_track,
    hypothesis_weight,
    lift_to_world,
)
from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.types import ObjectTrack, PoseHypothesis, QualitySignals, Stage
from tests.synth import box_model, cylinder_model, look_at_T_world_camera, render_view, simple_K

K = simple_K()
SIZE = (240, 320)


def _cameras(n: int, radius: float = 450.0, elevation: float = 300.0) -> list:
    """``n`` cameras on a ring above the origin, all looking at it."""
    Ts = []
    for i in range(n):
        a = 2 * np.pi * i / n
        Ts.append(look_at_T_world_camera([radius * np.cos(a), radius * np.sin(a), elevation]))
    return Ts


def _hyp(T_camera_object, camera_id, object_id=1, detection_id=0, **signals):
    return PoseHypothesis(
        camera_id,
        object_id,
        detection_id,
        0,
        T_camera_object,
        Stage.REFINED,
        QualitySignals(**signals),
    )


def _world_hyp(T_world_object, T_world_camera, camera_id, weight=1.0, **kw):
    T_co = tf.invert(T_world_camera) @ T_world_object
    return lift_to_world(_hyp(T_co, camera_id, **kw), T_world_camera, weight)


def _errors(T, T_gt, model):
    return tf.translation_distance(T, T_gt), sym_aware_rotation_distance_deg(
        T, T_gt, model.symmetry
    )


# ----------------------------------------------------------------------------- association


def test_three_copies_two_cameras_one_occluded_give_three_tracks():
    box = box_model()
    rng = np.random.default_rng(0)
    # three copies of one ObjectModel 45 mm apart (0.6 d), on the world plane
    truths = [
        tf.rotvec_T([0, 0, 1], 20.0 * i, t=[45.0 * i - 45.0, 10.0 * i, 10.0]) for i in range(3)
    ]
    cams = _cameras(2)
    hyps_by_view = {}
    for c, T_wc in enumerate(cams):
        cid = f"{c:06d}"
        hyps_by_view[cid] = []
        for j, T_wo in enumerate(truths):
            if c == 1 and j == 2:
                continue  # occluded in the second View: no Detection, no hypothesis
            noise = tf.rotvec_T(rng.normal(size=3), 2.0, t=rng.normal(0, 1.5, 3))
            hyps_by_view[cid].append(
                _world_hyp(T_wo @ noise, T_wc, cid, weight=1.0, detection_id=j)
            )
    res = associate(hyps_by_view, {1: box})
    assert len(res.tracks) == 3
    sizes = sorted(len(t.hypotheses) for t in res.tracks)
    assert sizes == [1, 2, 2]
    # every two-member track holds the same physical copy (its members' world poses agree)
    for t in res.tracks:
        ms = res.members[t.track_id]
        if len(ms) == 2:
            assert tf.translation_distance(ms[0].T_world_object, ms[1].T_world_object) < 6.0
    assert res.n_gated_out > 0  # neighbours were excluded by the gate, not just out-costed


def test_association_is_one_to_one_per_view_and_symmetry_aware():
    cyl = cylinder_model()
    T_wo = tf.rotvec_T([1, 0, 0], 30.0, t=[0, 0, 30])
    cams = _cameras(2)
    # the second View sees the same cylinder turned 137° about its axis: still one track
    turned = T_wo @ tf.rotvec_T([0, 0, 1], 137.0)
    hyps_by_view = {
        "000000": [_world_hyp(T_wo, cams[0], "000000", object_id=2)],
        "000001": [
            _world_hyp(turned, cams[1], "000001", object_id=2, detection_id=0),
            # a second hypothesis in the same View at the same place cannot join the same track
            _world_hyp(
                T_wo @ tf.rotvec_T([0, 0, 1], 5.0, t=[1, 0, 0]),
                cams[1],
                "000001",
                object_id=2,
                detection_id=1,
            ),
        ],
    }
    res = associate(hyps_by_view, {2: cyl}, AssociationParams(gate_rot_deg=20.0))
    assert len(res.tracks) == 2
    assert sorted(len(t.hypotheses) for t in res.tracks) == [1, 2]


# ----------------------------------------------------------------------------- fusion


def test_weights_skip_nan_signals_and_penalise_rejections():
    p = WeightParams()
    s = QualitySignals(seg_score=0.5, icp_fitness=0.8)  # coverage / visible fraction unknown
    assert hypothesis_weight(s, rejected=False, p=p) == pytest.approx(0.4)
    assert hypothesis_weight(s, rejected=True, p=p) == pytest.approx(0.4 * p.rejected_factor)
    assert hypothesis_weight(QualitySignals(seg_score=0.0), False, p) == p.floor


def test_three_noisy_views_fuse_better_than_the_best_single_view():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([0.2, 1, 0], 40.0, t=[20, -10, 15])
    cams = _cameras(3)
    # errors that cancel in the mean: each member is off by 4 mm / 3 deg in a different direction
    xis = [
        np.array([4, 0, 0, 0, 0, np.radians(3)]),
        np.array([-2, 3.4, 0, np.radians(2.6), 0, np.radians(-1.5)]),
    ]
    xis.append(-(xis[0] + xis[1]))
    members = [
        _world_hyp(T_gt @ tf.se3_exp(xi), T_wc, f"{c:06d}", weight=1.0)
        for c, (xi, T_wc) in enumerate(zip(xis, cams, strict=True))
    ]
    track = ObjectTrack(0, 1, [m.hypothesis for m in members])
    out = fuse_track(track, members, box, FusionParams(method="mean"))
    dt, dr = _errors(out.fused.T_world_object, T_gt, box)
    singles = [_errors(m.T_world_object, T_gt, box) for m in members]
    assert dt < min(s[0] for s in singles) and dr < min(s[1] for s in singles)
    assert dt < 0.5 and dr < 0.5
    assert out.fused.signals.n_views == 3
    assert 2.0 < out.fused.signals.dispersion_mm < 6.0
    # "best" keeps the highest-weight member untouched
    best = fuse_track(track, members, box, FusionParams(method="best"))
    assert np.allclose(best.fused.T_world_object, members[0].T_world_object)


def test_cylinder_turned_137_degrees_fuses_to_the_same_pose():
    cyl = cylinder_model(steps=36)
    T_gt = tf.rotvec_T([1, 0.5, 0], 25.0, t=[5, 5, 40])
    cams = _cameras(2)
    members = [
        _world_hyp(T_gt, cams[0], "000000", object_id=2),
        _world_hyp(T_gt @ tf.rotvec_T([0, 0, 1], 137.0), cams[1], "000001", object_id=2),
    ]
    out = fuse_track(ObjectTrack(0, 2, [m.hypothesis for m in members]), members, cyl)
    dt, dr = _errors(out.fused.T_world_object, T_gt, cyl)
    assert dt < 1e-6
    assert dr <= 180.0 / 36 + 1e-6  # within half a discretisation step about the axis
    assert out.n_aligned == 1 and out.branch_index[1] != 0
    # without alignment the naive mean would sit ~68 deg off about the axis
    naive = tf.weighted_mean_se3([m.T_world_object for m in members])
    assert tf.rotation_distance_deg(naive, T_gt) > 60.0


def test_members_in_different_symmetry_branches_do_not_corrupt_the_mean():
    box = box_model(symmetric=True)  # 180 deg flips about each axis
    T_gt = tf.rotvec_T([0, 1, 0.3], 50.0, t=[0, 0, 20])
    cams = _cameras(3)
    flip_z = tf.rotvec_T([0, 0, 1], 180.0)
    flip_x = tf.rotvec_T([1, 0, 0], 180.0)
    small = tf.rotvec_T([1, 1, 0], 1.0, t=[1, 0, 0])
    members = [
        _world_hyp(T_gt, cams[0], "000000"),
        _world_hyp(T_gt @ small @ flip_z, cams[1], "000001"),
        _world_hyp(T_gt @ tf.invert(small) @ flip_x, cams[2], "000002"),
    ]
    out = fuse_track(ObjectTrack(0, 1, [m.hypothesis for m in members]), members, box)
    dt, dr = _errors(out.fused.T_world_object, T_gt, box)
    assert dt < 0.5 and dr < 0.5
    assert out.n_aligned == 2
    assert out.fused.signals.dispersion_deg < 1.0


# ----------------------------------------------------------------------------- joint polish


def _multiview_scene(model, T_gt, cams, noise_mm=0.3, seed=0):
    views = []
    for c, T_wc in enumerate(cams):
        T_co = tf.invert(T_wc) @ T_gt
        view, masks = render_view(
            [(model, T_co)], K, SIZE, depth_noise_mm=noise_mm, seed=seed + c, camera_id=f"{c:06d}"
        )
        view = dataclasses.replace(view, T_world_camera=T_wc)
        views.append((view, masks[0]))
    return views


def test_joint_icp_recovers_truth_from_two_views():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 1, 0.3], 35.0, t=[0, 0, 20])
    views = _multiview_scene(box, T_gt, _cameras(2, radius=350.0, elevation=250.0))
    T_start = T_gt @ tf.rotvec_T([1, 0.5, 0], 4.0, t=[3, -2, 4])
    out = JointRefiner(box, JointIcpParams()).polish(T_start, views)
    assert out.accepted, out.reason
    assert out.registration is not None and out.registration.fitness > 0.8
    assert out.n_views_with_depth == 2
    dt, dr = _errors(out.T_world_object, T_gt, box)
    assert dt < 1.0 and dr < 1.0, (dt, dr)
    assert out.silhouette_iou > 0.9
    assert out.displacement_mm == pytest.approx(
        tf.translation_distance(T_start, out.T_world_object)
    )


def test_joint_icp_rejects_when_no_view_has_depth():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 1, 0.3], 35.0, t=[0, 0, 20])
    views = _multiview_scene(box, T_gt, _cameras(2, radius=350.0, elevation=250.0))
    views = [(dataclasses.replace(v, depth=None), m) for v, m in views]
    out = JointRefiner(box).polish(T_gt, views)
    assert not out.accepted and out.reason == "no_depth"
    assert np.allclose(out.T_world_object, T_gt)


def test_joint_icp_gate_caps_the_displacement():
    box = box_model(symmetric=False)
    T_gt = tf.rotvec_T([1, 1, 0.3], 35.0, t=[0, 0, 20])
    views = _multiview_scene(box, T_gt, _cameras(2, radius=350.0, elevation=250.0))
    T_start = T_gt @ tf.rotvec_T([1, 0.5, 0], 4.0, t=[3, -2, 4])
    from binposert.refine import GateParams

    tight = JointIcpParams(gate=GateParams(alpha=0.01, beta_deg=90, min_iou=0.0, max_iou_drop=1.0))
    out = JointRefiner(box, tight).polish(T_start, views)
    assert not out.accepted and out.reason == "displacement_translation"
    assert np.allclose(out.T_world_object, T_start)
    assert _errors(out.T_candidate, T_gt, box)[0] < 1.0  # the candidate itself was right


def test_world_hypothesis_round_trip():
    T_wc = _cameras(1)[0]
    T_co = tf.rotvec_T([0, 1, 0], 10.0, t=[0, 0, 400])
    w = lift_to_world(_hyp(T_co, "000000"), T_wc, 0.7)
    assert isinstance(w, WorldHypothesis)
    assert np.allclose(tf.invert(T_wc) @ w.T_world_object, T_co)
