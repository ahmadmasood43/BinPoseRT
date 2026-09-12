"""SE(3) helpers. Every transform is a 4x4 float64 ``T_a_b`` mapping frame-b points into frame a."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

Mat4 = npt.NDArray[np.float64]

_EPS = 1e-12


def make_T(R: npt.ArrayLike, t: npt.ArrayLike) -> Mat4:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rotation(T: Mat4) -> npt.NDArray[np.float64]:
    return T[:3, :3]


def translation(T: Mat4) -> npt.NDArray[np.float64]:
    return T[:3, 3]


def invert(T: Mat4) -> Mat4:
    """Closed-form inverse for rigid transforms (no numerical inversion)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(*Ts: Mat4) -> Mat4:
    """compose(T_a_b, T_b_c, T_c_d) -> T_a_d."""
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def transform_points(T: Mat4, pts: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.asarray(pts, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def is_rigid(T: Mat4, atol: float = 1e-6) -> bool:
    R = T[:3, :3]
    return (
        T.shape == (4, 4)
        and np.allclose(R @ R.T, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
        and np.allclose(T[3], [0, 0, 0, 1], atol=atol)
    )


def rotation_angle_deg(R: npt.NDArray[np.float64]) -> float:
    """Geodesic angle of a rotation matrix, in degrees (quaternion-based, precise near 0)."""
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def rotation_distance_deg(T_a: Mat4, T_b: Mat4) -> float:
    return rotation_angle_deg(T_a[:3, :3].T @ T_b[:3, :3])


def translation_distance(T_a: Mat4, T_b: Mat4) -> float:
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def se3_distance(T_a: Mat4, T_b: Mat4, rot_weight_mm_per_deg: float = 1.0) -> float:
    """Scalar SE(3) distance: translation (mm) + weight * rotation (deg), for symmetry selection."""
    return translation_distance(T_a, T_b) + rot_weight_mm_per_deg * rotation_distance_deg(T_a, T_b)


# ----------------------------------------------------------------------------- Lie algebra


def se3_log(T: Mat4) -> npt.NDArray[np.float64]:
    """Log map SE(3) -> se(3): a 6-vector [rho (3), phi (3)] with rotation in radians."""
    R = T[:3, :3]
    t = T[:3, 3]
    phi = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(phi)
    if theta < 1e-9:
        V_inv = np.eye(3) - 0.5 * _skew(phi)
    else:
        a = phi / theta
        A = _skew(a)
        half = theta / 2.0
        cot_half = 1.0 / np.tan(half)
        V_inv = half * cot_half * np.eye(3) + (1 - half * cot_half) * np.outer(a, a) - half * A
    rho = V_inv @ t
    return np.concatenate([rho, phi])


def se3_exp(xi: npt.ArrayLike) -> Mat4:
    """Exp map se(3) -> SE(3) for a 6-vector [rho, phi]."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho, phi = xi[:3], xi[3:]
    theta = np.linalg.norm(phi)
    R = Rotation.from_rotvec(phi).as_matrix()
    if theta < 1e-9:
        V = np.eye(3) + 0.5 * _skew(phi)
    else:
        a = phi / theta
        A = _skew(a)
        V = (
            np.eye(3)
            + ((1 - np.cos(theta)) / theta) * A
            + ((theta - np.sin(theta)) / theta) * (A @ A)
        )
    return make_T(R, V @ rho)


def weighted_mean_se3(
    Ts: list[Mat4] | tuple[Mat4, ...],
    weights: npt.ArrayLike | None = None,
    T_ref: Mat4 | None = None,
    iterations: int = 5,
) -> Mat4:
    """Weighted intrinsic mean on SE(3): T_ref * exp(sum w_i log(T_ref^-1 T_i) / sum w_i), iterated.

    Callers must symmetry-align the inputs to ``T_ref`` first (ADR-0003).
    """
    if len(Ts) == 0:
        raise ValueError("weighted_mean_se3 needs at least one transform")
    w = np.ones(len(Ts)) if weights is None else np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    w = w / w.sum()
    T = Ts[int(np.argmax(w))].copy() if T_ref is None else T_ref.copy()
    for _ in range(iterations):
        xi = sum(wi * se3_log(invert(T) @ Ti) for wi, Ti in zip(w, Ts, strict=True))
        T = T @ se3_exp(xi)
        if np.linalg.norm(xi) < 1e-10:
            break
    return T


def _skew(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def random_rigid(rng: np.random.Generator, t_scale_mm: float = 100.0) -> Mat4:
    """Uniformly random rotation and Gaussian translation — for tests."""
    return make_T(Rotation.random(random_state=rng).as_matrix(), rng.normal(0, t_scale_mm, 3))


def rotvec_T(axis: npt.ArrayLike, angle_deg: float, t: npt.ArrayLike = (0.0, 0.0, 0.0)) -> Mat4:
    axis_v = np.asarray(axis, dtype=np.float64)
    axis_v = axis_v / (np.linalg.norm(axis_v) + _EPS)
    return make_T(Rotation.from_rotvec(axis_v * np.radians(angle_deg)).as_matrix(), t)
