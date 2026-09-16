"""Observed side of registration: the masked scene point cloud with normals."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d

from binposert.render import unproject_depth
from binposert.types import Mat3

F64 = npt.NDArray[np.float64]


def scene_cloud(
    depth: F64,
    K: Mat3,
    mask: npt.NDArray[np.bool_],
    erode_px: int = 2,
    z_center_mm: float | None = None,
    z_window_mm: float | None = None,
    max_points: int = 3000,
    normal_radius_mm: float = 10.0,
    seed: int = 0,
) -> tuple[F64, F64, float]:
    """Camera-frame points and normals inside ``mask`` where depth is valid.

    The mask is eroded by ``erode_px`` so boundary pixels, which mix object and background depth,
    are dropped. With ``z_center_mm``/``z_window_mm`` points whose depth is farther than the window
    from the coarse estimate are discarded (background seen through holes in the mask). Normals are
    oriented towards the camera. Returns ``(points, normals, depth_coverage)`` where depth_coverage
    is the fraction of the *original* mask with valid depth.
    """
    valid = depth > 0
    coverage = float((valid & mask).sum() / max(int(mask.sum()), 1))
    m = mask.astype(np.uint8)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, kernel).astype(np.uint8)
    sel = valid & (m > 0)
    if z_center_mm is not None and z_window_mm is not None:
        sel &= np.abs(depth - z_center_mm) <= z_window_mm
    pts = unproject_depth(depth, K, sel)
    if len(pts) == 0:
        return pts, np.zeros((0, 3)), coverage
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_mm, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pts, np.asarray(pcd.normals, dtype=np.float64), coverage
