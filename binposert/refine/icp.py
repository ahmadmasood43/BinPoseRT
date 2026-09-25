"""Local registration variants behind one signature (RQ-B): point-to-plane, robust, GICP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import open3d as o3d

from binposert.transforms import Mat4, invert, is_rigid, translation_distance

F64 = npt.NDArray[np.float64]

VARIANTS = ("point_to_plane", "robust", "gicp")


@dataclass(frozen=True)
class RegistrationResult:
    T_camera_object: Mat4
    fitness: float  # fraction of observed points with a model correspondence within max_corr_dist
    inlier_rmse_mm: float
    n_correspondences: int
    converged: bool
    fallback: bool = False  # point-to-point was used because the plane solve was degenerate


def register(
    pts_obj: F64,
    normals_obj: F64,
    pts_cam: F64,
    normals_cam: F64,
    T_init: Mat4,
    variant: str = "point_to_plane",
    max_corr_dist_mm: float = 10.0,
    max_iterations: int = 30,
    robust_k_mm: float = 5.0,
    src_pcd: "o3d.geometry.PointCloud | None" = None,
) -> RegistrationResult:
    """Refine ``T_init`` (``T_camera_object``) by registering the observed camera-frame cloud onto
    the object-frame model cloud.

    The *scene* is the moving side and the *model* the fixed target: the target normals of
    point-to-plane ICP are then the renderer's exact normals rather than normals estimated from a
    small noisy depth patch, which is what keeps the solve stable on partially visible objects.
    ``fitness`` is the fraction of observed points explained by the model surface.

    Point-to-plane has a null space when the visible surface shows fewer than three independent
    normal directions (two faces of a box) and Open3D's solver then returns an absurd translation;
    such a result is detected (no correspondences, non-rigid, or a jump beyond
    ``10 x max_corr_dist``) and the stage is redone with point-to-point, flagged ``fallback``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown ICP variant {variant!r}; choose from {VARIANTS}")
    if len(pts_obj) < 10 or len(pts_cam) < 10:
        return RegistrationResult(T_init.copy(), 0.0, float("inf"), 0, False)
    if src_pcd is None:
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
        src.normals = o3d.utility.Vector3dVector(normals_cam)
    else:
        src = src_pcd  # pre-built by caller; pts_cam/normals_cam are identical across ICP levels
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_obj))
    dst.normals = o3d.utility.Vector3dVector(normals_obj)
    init = invert(T_init)  # T_object_camera moves scene points into the model frame
    reg = o3d.pipelines.registration
    criteria = reg.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iterations
    )
    if variant == "point_to_plane":
        result = reg.registration_icp(
            src, dst, max_corr_dist_mm, init, reg.TransformationEstimationPointToPlane(), criteria
        )
    elif variant == "robust":
        result = reg.registration_icp(
            src,
            dst,
            max_corr_dist_mm,
            init,
            reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=robust_k_mm)),
            criteria,
        )
    else:
        src.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        dst.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        result = reg.registration_generalized_icp(
            src,
            dst,
            max_corr_dist_mm,
            init,
            reg.TransformationEstimationForGeneralizedICP(),
            criteria,
        )
    out = _result(result, T_init, max_corr_dist_mm)
    if out.converged or variant == "gicp":
        return out
    result = reg.registration_icp(
        src, dst, max_corr_dist_mm, init, reg.TransformationEstimationPointToPoint(), criteria
    )
    out = _result(result, T_init, max_corr_dist_mm)
    return RegistrationResult(
        out.T_camera_object,
        out.fitness,
        out.inlier_rmse_mm,
        out.n_correspondences,
        out.converged,
        True,
    )


def _result(result: Any, T_init: Mat4, max_corr_dist_mm: float) -> RegistrationResult:
    T_oc = np.asarray(result.transformation, dtype=np.float64)
    n_corr = len(result.correspondence_set)
    ok = bool(np.all(np.isfinite(T_oc))) and is_rigid(T_oc, atol=1e-4) and n_corr > 0
    if ok:
        T = invert(T_oc)
        ok = translation_distance(T, T_init) <= 10.0 * max_corr_dist_mm
    if not ok:
        return RegistrationResult(T_init.copy(), 0.0, float("inf"), n_corr, False)
    return RegistrationResult(
        invert(T_oc), float(result.fitness), float(result.inlier_rmse), n_corr, True
    )
