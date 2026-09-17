"""Joint multi-view polish (D10 step 4): one registration of the visibility-cropped model against
the union of every View's masked point cloud, expressed in the world frame.

The Refinement building blocks are reused unchanged: :func:`scene_cloud` gives each View's
observed points in its camera frame (moved into the world frame with the View's extrinsics),
:func:`visible_model_cloud` gives the model surface each camera can see at the current pose (the
union over Views is the model side), and :func:`register` runs ICP with ``T_world_object`` in the
role ``T_camera_object`` plays for a single View — it only needs a transform from the object frame
into the frame the observed points live in. The gate is D8's: fitness and the displacement caps
relative to the fused mean the polish started from; per-view silhouette IoU is measured and kept
as a signal (the Beta finding that the mask is not independent evidence holds here too).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from binposert.refine import (
    GateParams,
    RegistrationResult,
    gate_reason,
    register,
    scene_cloud,
    silhouette_iou,
    visible_model_cloud,
    visible_silhouette,
)
from binposert.render import MeshRenderer
from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.transforms import Mat4, transform_points, translation_distance
from binposert.types import ObjectModel, View

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class JointIcpParams:
    variant: str = "point_to_plane"
    corr_dist_factors: tuple[float, ...] = (0.15, 0.08, 0.04)  # × diameter, coarse to fine
    max_iterations: int = 30
    robust_k_factor: float = 0.05
    max_model_points_per_view: int = 2000
    max_scene_points_per_view: int = 2000
    erode_px: int = 2
    mask_dilate_px: int = 3
    z_window_factor: float = 1.5
    min_scene_points: int = 50
    gate: GateParams = field(
        default_factory=lambda: GateParams(min_iou=0.0, max_iou_drop=1.0)  # silhouette: signal only
    )


@dataclass(frozen=True)
class JointIcpOutcome:
    T_world_object: Mat4  # accepted result, or the start pose when rejected
    T_candidate: Mat4  # what ICP produced either way
    accepted: bool
    reason: str | None
    registration: RegistrationResult | None
    n_scene_points: int
    n_views_with_depth: int
    displacement_mm: float
    displacement_deg: float
    silhouette_iou: float  # mean over Views at the final pose
    depth_coverage: float  # mean over Views
    seconds: float


class JointRefiner:
    """One ObjectModel, its renderer built once; call :meth:`polish` per ObjectTrack."""

    def __init__(self, model: ObjectModel, params: JointIcpParams | None = None) -> None:
        self.model = model
        self.params = params or JointIcpParams()
        self.renderer = MeshRenderer.from_model(model)

    def polish(
        self, T_world_object: Mat4, views: list[tuple[View, npt.NDArray[np.bool_]]]
    ) -> JointIcpOutcome:
        """``views`` pairs each View the track was seen in with that View's Detection mask."""
        t0 = time.perf_counter()
        p = self.params
        d = self.model.diameter
        T_start = T_world_object.copy()

        pts_w, normals_w, coverages = [], [], []
        usable: list[tuple[View, npt.NDArray[np.bool_]]] = []
        for view, mask in views:
            if view.depth is None:
                continue
            T_co = view.T_camera_world @ T_start
            pts_c, n_c, cov = scene_cloud(
                view.depth,
                view.K,
                mask,
                erode_px=p.erode_px,
                z_center_mm=float(T_co[2, 3]),
                z_window_mm=p.z_window_factor * d,
                max_points=p.max_scene_points_per_view,
            )
            coverages.append(cov)
            if len(pts_c) == 0:
                continue
            R_wc = view.T_world_camera[:3, :3]
            pts_w.append(transform_points(view.T_world_camera, pts_c))
            normals_w.append(n_c @ R_wc.T)
            usable.append((view, mask))
        coverage = float(np.mean(coverages)) if coverages else 0.0
        n_scene = int(sum(len(x) for x in pts_w))
        if n_scene < p.min_scene_points:
            reason_early = "no_depth" if coverage == 0.0 else "depth_window"
            return self._rejected(
                T_start, T_start, None, n_scene, len(usable), coverage, reason_early, t0
            )
        scene_pts = np.concatenate(pts_w)
        scene_normals = np.concatenate(normals_w)

        T = T_start.copy()
        reg: RegistrationResult | None = None
        for factor in p.corr_dist_factors:
            pts_obj, normals_obj = self._model_cloud(T, usable)
            if len(pts_obj) < p.min_scene_points:
                return self._rejected(
                    T_start, T, reg, n_scene, len(usable), coverage, "no_overlap", t0
                )
            reg = register(
                pts_obj,
                normals_obj,
                scene_pts,
                scene_normals,
                T,
                variant=p.variant,
                max_corr_dist_mm=factor * d,
                max_iterations=p.max_iterations,
                robust_k_mm=p.robust_k_factor * d,
            )
            if not reg.converged:
                return self._rejected(
                    T_start, T, reg, n_scene, len(usable), coverage, "icp_failed", t0
                )
            T = reg.T_camera_object
        assert reg is not None

        d_t = translation_distance(T_start, T)
        d_r = sym_aware_rotation_distance_deg(T_start, T, self.model.symmetry)
        iou = self._mean_iou(T, usable)
        reason = gate_reason(reg.fitness, d_t, d_r, 1.0, iou, d, p.gate)
        return JointIcpOutcome(
            T_world_object=T if reason is None else T_start,
            T_candidate=T,
            accepted=reason is None,
            reason=reason,
            registration=reg,
            n_scene_points=n_scene,
            n_views_with_depth=len(usable),
            displacement_mm=d_t,
            displacement_deg=d_r,
            silhouette_iou=iou,
            depth_coverage=coverage,
            seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------ helpers

    def _model_cloud(
        self, T_world_object: Mat4, views: list[tuple[View, npt.NDArray[np.bool_]]]
    ) -> tuple[F64, F64]:
        pts, normals = [], []
        for view, mask in views:
            T_co = view.T_camera_world @ T_world_object
            p_obj, n_obj, _ = visible_model_cloud(
                self.renderer,
                T_co,
                view.K,
                view.image_size,
                det_mask=mask,
                mask_dilate_px=self.params.mask_dilate_px,
                max_points=self.params.max_model_points_per_view,
            )
            if len(p_obj):
                pts.append(p_obj)
                normals.append(n_obj)
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(pts), np.concatenate(normals)

    def _mean_iou(
        self, T_world_object: Mat4, views: list[tuple[View, npt.NDArray[np.bool_]]]
    ) -> float:
        ious = []
        for view, mask in views:
            r = self.renderer.render(view.T_camera_world @ T_world_object, view.K, view.image_size)
            ious.append(
                silhouette_iou(visible_silhouette(r, view.depth, self.params.gate.delta_mm), mask)
            )
        return float(np.mean(ious)) if ious else float("nan")

    def _rejected(
        self,
        T_start: Mat4,
        T_candidate: Mat4,
        reg: RegistrationResult | None,
        n_scene: int,
        n_views: int,
        coverage: float,
        reason: str,
        t0: float,
    ) -> JointIcpOutcome:
        return JointIcpOutcome(
            T_world_object=T_start,
            T_candidate=T_candidate,
            accepted=False,
            reason=reason,
            registration=reg,
            n_scene_points=n_scene,
            n_views_with_depth=n_views,
            displacement_mm=float("nan"),
            displacement_deg=float("nan"),
            silhouette_iou=float("nan"),
            depth_coverage=coverage,
            seconds=time.perf_counter() - t0,
        )
