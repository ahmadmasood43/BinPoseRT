"""The simulated pick (D14): the grasp chain composes in the right order, the authored grasp
fits the model, and the Open3D drawing runs where an offscreen context exists."""

from __future__ import annotations

import numpy as np
import pytest

from binposert.transforms import invert, is_rigid, make_T, rotvec_T
from binposert.types import FusedPose, QualitySignals, Verdict, View
from binposert.viz.grasp import (
    GraspPose,
    author_bbox_grasp,
    chain_text,
    load_grasps,
    pick_transform,
    plan_picks,
    render_pick,
    save_grasps,
)
from tests.synth import box_model, look_at_T_world_camera, simple_K


def test_bbox_grasp_closes_across_the_smallest_extent_and_round_trips(tmp_path):
    model = box_model(extents=(60.0, 40.0, 20.0))
    g = author_bbox_grasp(model)
    T = g.T_object_gripper
    assert is_rigid(T) and np.allclose(T[:3, 3], 0.0, atol=1e-9)  # the box is centred
    assert g.jaw_opening_mm == pytest.approx(20.0)
    assert np.allclose(np.abs(T[:3, 0]), [0, 0, 1])  # jaws close along z (20 mm)
    assert np.allclose(T[:3, 2], [0, -1, 0])  # approach along the middle extent, from +y
    save_grasps(tmp_path / "g.json", {model.object_id: g})
    back = load_grasps(tmp_path / "g.json")[model.object_id]
    assert np.allclose(back.T_object_gripper, T) and back.name == g.name


def test_pick_chain_composes_robot_world_object_gripper():
    T_world_object = rotvec_T([0, 0, 1], 90.0, t=(100.0, 50.0, 0.0))
    T_object_gripper = make_T(np.eye(3), (0.0, 0.0, 10.0))
    T_world_robot = make_T(np.eye(3), (-600.0, 0.0, 0.0))
    T_robot_world = invert(T_world_robot)
    T = pick_transform(T_robot_world, T_world_object, T_object_gripper)
    # the gripper origin: object frame (0, 0, 10) -> world (100, 50, 10) -> robot (700, 50, 10)
    assert np.allclose(T[:3, 3], [700.0, 50.0, 10.0])
    assert np.allclose(T[:3, :3], T_world_object[:3, :3])
    fused = [
        FusedPose(0, 1, T_world_object, 0.9, Verdict.ACCEPT, QualitySignals()),
        FusedPose(1, 1, np.eye(4), 0.3, Verdict.REJECT, QualitySignals()),
        FusedPose(2, 7, np.eye(4), 0.99, Verdict.ACCEPT, QualitySignals()),  # no grasp authored
    ]
    grasps = {1: GraspPose(1, "test", T_object_gripper, 20.0)}
    picks = plan_picks(fused, grasps, T_robot_world)
    assert [p.track_id for p in picks] == [0, 1]  # most confident first, object 7 skipped
    assert np.allclose(picks[0].T_robot_gripper, T)
    text = chain_text(picks[0], grasps[1], T_robot_world)
    assert "T_robot_gripper = T_robot_world @ T_world_object @ T_object_gripper" in text


def test_render_pick_writes_an_image(tmp_path):
    import open3d as o3d

    try:
        o3d.visualization.rendering.OffscreenRenderer(32, 32)
    except Exception as e:  # noqa: BLE001 — no EGL / OSMesa on this machine
        pytest.skip(f"no offscreen renderer: {e}")
    model = box_model(extents=(60.0, 40.0, 20.0))
    g = author_bbox_grasp(model)
    T_a = make_T(np.eye(3), (0.0, 0.0, 0.0))
    T_b = make_T(np.eye(3), (120.0, 0.0, 0.0))
    fused = [
        FusedPose(0, model.object_id, T_a, 0.95, Verdict.ACCEPT, QualitySignals()),
        FusedPose(1, model.object_id, T_b, 0.5, Verdict.REQUEST_VIEW, QualitySignals()),
    ]
    T_robot_world = invert(make_T(np.eye(3), (-600.0, 0.0, 0.0)))
    picks = plan_picks(fused, {model.object_id: g}, T_robot_world)
    view = View("000000", simple_K(), look_at_T_world_camera((0, 0, 500)), None, None, (240, 320))
    path = tmp_path / "pick.png"
    render_pick(
        {model.object_id: model},
        fused,
        picks,
        [view],
        {model.object_id: g},
        path,
        T_robot_world,
        image_size=(120, 160),
    )
    img = np.asarray(o3d.io.read_image(str(path)))
    assert img.shape == (120, 160, 3) and img.std() > 0
