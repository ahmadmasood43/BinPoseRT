import numpy as np
import pytest

from binposert import transforms as tf


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_frame_convention_composes_left_to_right(rng):
    """T_a_b @ T_b_c == T_a_c, and points move b -> a. This test runs first on purpose."""
    T_world_camera = tf.random_rigid(rng)
    T_camera_object = tf.random_rigid(rng)
    T_world_object = tf.compose(T_world_camera, T_camera_object)
    p_object = rng.normal(size=(5, 3))
    p_camera = tf.transform_points(T_camera_object, p_object)
    p_world = tf.transform_points(T_world_camera, p_camera)
    np.testing.assert_allclose(tf.transform_points(T_world_object, p_object), p_world, atol=1e-9)


def test_invert_round_trip(rng):
    T = tf.random_rigid(rng)
    np.testing.assert_allclose(tf.invert(T) @ T, np.eye(4), atol=1e-9)
    np.testing.assert_allclose(tf.invert(T), np.linalg.inv(T), atol=1e-9)
    assert tf.is_rigid(T)


def test_se3_log_exp_round_trip(rng):
    for _ in range(20):
        T = tf.random_rigid(rng, t_scale_mm=300)
        np.testing.assert_allclose(tf.se3_exp(tf.se3_log(T)), T, atol=1e-8)
    xi = np.zeros(6)
    np.testing.assert_allclose(tf.se3_exp(xi), np.eye(4))
    T_small = tf.rotvec_T([0, 0, 1], 1e-7, t=[1, 2, 3])
    np.testing.assert_allclose(tf.se3_exp(tf.se3_log(T_small)), T_small, atol=1e-9)


def test_rotation_and_translation_distances():
    T_a = np.eye(4)
    T_b = tf.rotvec_T([0, 1, 0], 30.0, t=[3, 4, 0])
    assert tf.rotation_distance_deg(T_a, T_b) == pytest.approx(30.0)
    assert tf.translation_distance(T_a, T_b) == pytest.approx(5.0)
    assert tf.rotation_angle_deg(np.eye(3)) == 0.0


def test_weighted_mean_se3_recovers_center(rng):
    T_true = tf.random_rigid(rng)
    Ts, ws = [], []
    for _ in range(30):
        noise = tf.rotvec_T(rng.normal(size=3), rng.normal(0, 2.0), t=rng.normal(0, 2.0, 3))
        Ts.append(T_true @ noise)
        ws.append(1.0)
    T_mean = tf.weighted_mean_se3(Ts, ws)
    assert tf.rotation_distance_deg(T_mean, T_true) < 1.0
    assert tf.translation_distance(T_mean, T_true) < 1.0


def test_weighted_mean_se3_respects_weights():
    T_a = tf.make_T(np.eye(3), [0, 0, 0])
    T_b = tf.make_T(np.eye(3), [10, 0, 0])
    T = tf.weighted_mean_se3([T_a, T_b], [3.0, 1.0])
    np.testing.assert_allclose(T[:3, 3], [2.5, 0, 0], atol=1e-9)
    with pytest.raises(ValueError):
        tf.weighted_mean_se3([T_a, T_b], [0.0, 0.0])
