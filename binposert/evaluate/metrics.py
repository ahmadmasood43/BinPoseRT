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
    """Visible surface discrepancy for one or several misalignment tolerances ``tau``, following
    bop_toolkit's BOP19 definition exactly: depth images are converted to *distance* (ray length)
    images, a surface is visible where the rendered distance is within ``delta`` in front of the
    test distance **or the test depth is missing**, the estimate's visible region is unioned with
    the GT-visible pixels it covers, and a pixel is wrong when ``|dist_est - dist_gt| >= tau``.

    Returns an array with one VSD value per tau in [0, 1]. Depth images are in mm.
    """
    size = depth_test.shape
    ray = _ray_length_factor(K, size)
    dist_est = renderer.render(T_est, K, size).depth * ray
    dist_gt = renderer.render(T_gt, K, size).depth * ray
    return vsd_from_distances(dist_est, dist_gt, depth_test * ray, delta_mm, tau_mm)


def vsd_from_distances(
    dist_est: F64,
    dist_gt: F64,
    dist_test: F64,
    delta_mm: float = 15.0,
    tau_mm: float | npt.ArrayLike = 20.0,
) -> F64:
    """:func:`vsd` on pre-rendered *distance* images (``depth * ray length factor``), so a caller
    scoring every candidate against every ground-truth pose of an image renders each pose once
    instead of once per pair (the evaluate stage: n_candidates + n_gt renders instead of
    n_candidates × n_gt × 2)."""
    taus = np.atleast_1d(np.asarray(tau_mm, dtype=np.float64))
    missing = dist_test <= 0
    vis_gt = (dist_gt > 0) & (((dist_gt - dist_test) <= delta_mm) | missing)
    vis_est = (dist_est > 0) & (((dist_est - dist_test) <= delta_mm) | missing)
    vis_est |= vis_gt & (dist_est > 0)
    union = vis_est | vis_gt
    n_union = int(union.sum())
    if n_union == 0:
        return np.ones_like(taus)
    inter = vis_est & vis_gt
    n_comp = n_union - int(inter.sum())
    diff = np.abs(dist_est - dist_gt)[inter]
    out = np.empty_like(taus)
    for i, tau in enumerate(taus):
        out[i] = (int(np.sum(diff >= tau)) + n_comp) / n_union
    return out


def _ray_length_factor(K: Mat3, size: tuple[int, int]) -> F64:
    """Per-pixel factor converting z-depth to distance along the ray through the pixel centre."""
    h, w = size
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.sqrt(1.0 + x**2 + y**2)


def project(K: Mat3, pts_camera: F64) -> F64:
    z = pts_camera[:, 2:3]
    uv = (K @ pts_camera.T).T
    return uv[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
