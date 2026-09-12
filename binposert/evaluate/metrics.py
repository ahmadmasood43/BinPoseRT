"""Pose-error functions following the BOP definitions (Hodaň et al., BOP Challenge 2020).

All poses are ``T_camera_object`` (mm). ``pts`` are object-frame model points (N, 3). The symmetry
group is the flat list from ADR-0003; for MSSD/MSPD the minimum over symmetries is taken as in BOP.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from binposert.render import MeshRenderer
from binposert.transforms import Mat4, transform_points
from binposert.types import Mat3, SymmetryGroup

F64 = npt.NDArray[np.float64]


def add(T_est: Mat4, T_gt: Mat4, pts: F64) -> float:
    """Average distance of corresponding model points."""
    return float(
        np.mean(np.linalg.norm(transform_points(T_est, pts) - transform_points(T_gt, pts), axis=1))
    )


def adi(T_est: Mat4, T_gt: Mat4, pts: F64) -> float:
    """Average distance to the closest model point (symmetry-agnostic ADD-S)."""
    est = transform_points(T_est, pts)
    gt = transform_points(T_gt, pts)
    d, _ = cKDTree(gt).query(est, k=1)
    return float(np.mean(d))


def mssd(T_est: Mat4, T_gt: Mat4, pts: F64, symmetry: SymmetryGroup) -> float:
    """Maximum symmetry-aware surface distance: min_S max_x ||T_est x - T_gt S x||."""
    est = transform_points(T_est, pts)
    best = np.inf
    for S in symmetry.transforms:
        gt = transform_points(T_gt @ S, pts)
        best = min(best, float(np.max(np.linalg.norm(est - gt, axis=1))))
    return best


def mspd(T_est: Mat4, T_gt: Mat4, pts: F64, symmetry: SymmetryGroup, K: Mat3) -> float:
    """Maximum symmetry-aware projection distance in pixels."""
    est = project(K, transform_points(T_est, pts))
    best = np.inf
    for S in symmetry.transforms:
        gt = project(K, transform_points(T_gt @ S, pts))
        best = min(best, float(np.max(np.linalg.norm(est - gt, axis=1))))
    return best


def vsd(
    T_est: Mat4,
    T_gt: Mat4,
    renderer: MeshRenderer,
    depth_test: F64,
    K: Mat3,
    delta_mm: float = 15.0,
    tau_mm: float | npt.ArrayLike = 20.0,
) -> F64:
    """Visible surface discrepancy for one or several misalignment tolerances ``tau``.

    Returns an array with one VSD value per tau in [0, 1]. Depth images are in mm.
    """
    taus = np.atleast_1d(np.asarray(tau_mm, dtype=np.float64))
    size = depth_test.shape
    d_est = renderer.render(T_est, K, size).depth
    d_gt = renderer.render(T_gt, K, size).depth
    valid = depth_test > 0
    vis_est = (d_est > 0) & valid & ((d_est - depth_test) <= delta_mm)
    vis_gt = (d_gt > 0) & valid & ((d_gt - depth_test) <= delta_mm)
    union = vis_est | vis_gt
    n_union = int(union.sum())
    if n_union == 0:
        return np.ones_like(taus)
    inter = vis_est & vis_gt
    diff = np.abs(d_est - d_gt)[inter]
    out = np.empty_like(taus)
    for i, tau in enumerate(taus):
        n_ok = int(np.sum(diff <= tau))
        out[i] = 1.0 - n_ok / n_union
    return out


def project(K: Mat3, pts_camera: F64) -> F64:
    z = pts_camera[:, 2:3]
    uv = (K @ pts_camera.T).T
    return uv[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
