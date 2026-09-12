import numpy as np
import pytest

from binposert import transforms as tf
from binposert.render import MeshRenderer, render_scene, unproject_depth, visible_surface_points
from tests.synth import box_model, cylinder_model, look_at_T_world_camera, simple_K


def test_box_front_face_depth_matches_analytic_value():
    box = box_model(extents=(60.0, 40.0, 20.0), symmetric=False)
    K = simple_K()
    T = tf.make_T(np.eye(3), [0, 0, 500.0])  # box centred 500 mm in front, faces axis-aligned
    r = MeshRenderer.from_model(box).render(T, K, (240, 320))
    h, w = r.depth.shape
    assert r.mask[h // 2, w // 2]
    # front face is at z = 500 - 10 = 490 mm everywhere (orthogonal face, depth = z not ray length)
    front = r.depth[r.mask]
    assert front.min() == pytest.approx(490.0, abs=1e-3)
    # off-centre pixels on the same planar face must also read 490 (i.e. depth is z, not distance)
    assert r.depth[h // 2, w // 2 + 15] == pytest.approx(490.0, abs=1e-3)
    # projected extent: 60 mm wide at 490 mm with f=400 -> ~49 px
    xs = np.nonzero(r.mask.any(axis=0))[0]
    assert xs.max() - xs.min() + 1 == pytest.approx(60 * 400 / 490, abs=2)
    # normals on the front face point toward the camera (-z)
    n = r.normals_camera[h // 2, w // 2]
    np.testing.assert_allclose(n, [0, 0, -1], atol=1e-6)


def test_unproject_round_trips_rendered_depth():
    cyl = cylinder_model()
    K = simple_K()
    T = tf.rotvec_T([1, 0, 0], 60.0, t=[10, -5, 400])
    ren = MeshRenderer.from_model(cyl)
    r = ren.render(T, K, (240, 320))
    pts_cam = unproject_depth(r.depth, K)
    pts_obj = tf.transform_points(tf.invert(T), pts_cam)
    # all unprojected points lie on the cylinder surface (radius 15 or end caps at |z|=30)
    radial = np.linalg.norm(pts_obj[:, :2], axis=1)
    on_side = np.abs(radial - 15.0) < 0.5
    on_cap = np.abs(np.abs(pts_obj[:, 2]) - 30.0) < 0.5
    assert np.mean(on_side | on_cap) > 0.99
    pts_obj2, pts_cam2, normals = visible_surface_points(ren, T, K, (240, 320))
    np.testing.assert_allclose(pts_obj2, pts_obj)
    assert normals.shape == (len(pts_cam2), 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_silhouette_iou_with_itself_is_one_and_changes_with_pose():
    box = box_model()
    K = simple_K()
    ren = MeshRenderer.from_model(box)
    T = tf.rotvec_T([0, 1, 0], 30.0, t=[0, 0, 400])
    m1 = ren.render(T, K, (240, 320)).mask
    m2 = ren.render(T, K, (240, 320)).mask
    assert (m1 & m2).sum() / (m1 | m2).sum() == 1.0
    m3 = ren.render(T @ tf.make_T(np.eye(3), [30, 0, 0]), K, (240, 320)).mask
    assert (m1 & m3).sum() / (m1 | m3).sum() < 0.8


def test_scene_render_handles_occlusion_between_objects():
    box = box_model(object_id=1)
    cyl = cylinder_model(object_id=2)
    K = simple_K()
    T_box = tf.make_T(np.eye(3), [0, 0, 500.0])
    T_cyl = tf.rotvec_T([1, 0, 0], 90.0, t=[0, 0, 450.0])  # lying cylinder in front of the box
    res = render_scene([(box, T_box), (cyl, T_cyl)], K, (240, 320))
    h, w = res.depth.shape
    assert res.geometry_ids[h // 2, w // 2] == 1  # cylinder wins at the centre
    assert (res.geometry_ids == 0).any() and (res.geometry_ids == 1).any()
    assert (res.geometry_ids == -1).any()
    assert res.depth[res.geometry_ids == 1].min() == pytest.approx(435.0, abs=0.5)


def test_look_at_camera_sees_origin_in_image_centre():
    box = box_model(symmetric=False)
    K = simple_K()
    T_world_camera = look_at_T_world_camera(eye=[300, 200, 400], target=[0, 0, 0])
    T_world_object = np.eye(4)
    T_camera_object = tf.invert(T_world_camera) @ T_world_object
    r = MeshRenderer.from_model(box).render(T_camera_object, K, (240, 320))
    ys, xs = np.nonzero(r.mask)
    assert abs(xs.mean() - 160) < 3 and abs(ys.mean() - 120) < 3
