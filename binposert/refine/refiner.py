"""Refinement for one Detection: coarse PoseHypothesis -> refined (or rejected) PoseHypothesis.

Schedule: the coarse translation is first re-initialised from the observed depth along its ray
(``z_init``, see :mod:`binposert.refine.depth_init`); then for each correspondence distance in
``corr_dist_factors`` (× diameter, coarse to fine) the visible model surface is re-rendered at the
current pose (visibility-aware crop) and ICP runs; the result is then gated (D8). The displacement
caps bound ICP's move from the pose it started at; the silhouette checks compare against the coarse
pose the stage received. A rejection returns the coarse pose unchanged, with the reason and the
gate/registration QualitySignals attached either way.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import open3d as o3d

from binposert.refine.cloud import scene_cloud
from binposert.refine.crop import visible_model_cloud
from binposert.refine.depth_init import translation_from_depth
from binposert.refine.gate import GateDecision, GateParams, check_gate
from binposert.refine.icp import RegistrationResult, register
from binposert.refine.roi import Roi
from binposert.render import MeshRenderer
from binposert.transforms import Mat4
from binposert.types import ObjectModel, PoseHypothesis, QualitySignals, Stage, View


@dataclass(frozen=True)
class RefinerParams:
    variant: str = "point_to_plane"
    corr_dist_factors: tuple[float, ...] = (0.15, 0.08, 0.04)  # × diameter
    max_iterations: int = 30
    robust_k_factor: float = 0.05  # Tukey k × diameter (robust variant)
    max_model_points: int = 3000
    max_scene_points: int = 3000
    erode_px: int = 2
    mask_dilate_px: int = 3  # model crop keeps rendered pixels within the dilated Detection mask
    z_window_factor: float = 1.5  # keep scene depth within ± factor × diameter of the start z
    min_scene_points: int = 50
    z_init: str = "median_depth"  # "none" | "median_depth": translation initialised from depth
    z_init_min_overlap: int = 50  # rendered-silhouette ∩ mask pixels needed for the initialisation
    gate: GateParams = field(default_factory=GateParams)
    roi: str = "none"            # "none" | "bbox": Class-B ROI crop (F1); hash from YAML only
    roi_margin_px: int = 8       # extra padding around the ROI bounding box


@dataclass(frozen=True)
class RefineOutcome:
    hypothesis: PoseHypothesis  # refined, or the coarse pose with ``rejection_reason``
    T_candidate: Mat4  # what ICP produced, whether or not it was accepted
    z_shift_mm: float  # translation initialisation along the ray (0 when not applied)
    z_init_overlap: int  # pixels the initialisation was measured on
    gate: GateDecision | None
    registration: RegistrationResult | None
    n_scene_points: int
    depth_coverage: float
    seconds: float
    roi_fallback: int = 0   # number of renders that fell back to full frame because pose left ROI


class Refiner:
    """One ObjectModel, its renderer built once; call :meth:`refine` per Detection."""

    def __init__(self, model: ObjectModel, params: RefinerParams | None = None) -> None:
        self.model = model
        self.params = params or RefinerParams()
        self.renderer = MeshRenderer.from_model(model)
        self._z: tuple[float, int] = (0.0, 0)  # depth initialisation of the current call

    def refine(
        self, view: View, det_mask: npt.NDArray[np.bool_], hyp: PoseHypothesis
    ) -> RefineOutcome:
        t0 = time.perf_counter()
        p = self.params
        d = self.model.diameter
        T_coarse = hyp.T_camera_object
        size = view.image_size
        self._z = (0.0, 0)
        if view.depth is None:
            return self._rejected(hyp, T_coarse, None, None, 0, 0.0, "no_depth", t0)

        # Build ROI from detection-mask bbox ∪ projected sphere; use full frame when roi="none".
        if p.roi == "bbox":
            roi: Roi | None = Roi.from_mask_and_sphere(
                det_mask, T_coarse, view.K, d, size,
                dilate_px=p.mask_dilate_px + p.erode_px + 1,
                margin_px=p.roi_margin_px,
            )
            K_use = roi.K_shifted(view.K)
            depth_use = roi.crop(view.depth)
            mask_use = roi.crop(det_mask)
            size_use = roi.size
        elif p.roi == "none":
            roi = None
            K_use = view.K
            depth_use = view.depth
            mask_use = det_mask
            size_use = size
        else:
            raise ValueError(f"unknown roi {p.roi!r}; choose 'none' or 'bbox'")

        render_c = self.renderer.render(T_coarse, K_use, size_use)
        T_start, z_shift, n_overlap = T_coarse.copy(), 0.0, 0
        if p.z_init == "median_depth":
            T_start, z_shift, n_overlap = translation_from_depth(
                render_c, depth_use, mask_use, T_coarse, p.erode_px, p.z_init_min_overlap
            )
        elif p.z_init != "none":
            raise ValueError(f"unknown z_init {p.z_init!r}")
        self._z = (z_shift, n_overlap)

        pts_cam, normals_cam, coverage = scene_cloud(
            depth_use,
            K_use,
            mask_use,
            erode_px=p.erode_px,
            z_center_mm=float(T_start[2, 3]),
            z_window_mm=p.z_window_factor * d,
            max_points=p.max_scene_points,
        )
        if len(pts_cam) < p.min_scene_points:
            reason = "no_depth" if coverage == 0.0 else "depth_window"
            return self._rejected(hyp, T_start, None, None, len(pts_cam), coverage, reason, t0)

        T = T_start.copy()
        reg: RegistrationResult | None = None
        n_fallback = 0
        # Build scene PointCloud once; pts_cam/normals_cam are identical across ICP levels (F3).
        src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
        src_pcd.normals = o3d.utility.Vector3dVector(normals_cam)
        for factor in p.corr_dist_factors:
            # Per-render ROI fallback: if the pose has moved outside the ROI, use full frame.
            if roi is not None and not roi.contains_sphere(T, view.K, d):
                n_fallback += 1
                K_vmc, size_vmc, mask_vmc = view.K, size, det_mask
                src_for_reg = None  # must rebuild from full-frame pts_cam if we ever get here
            else:
                K_vmc, size_vmc, mask_vmc = K_use, size_use, mask_use
                src_for_reg = src_pcd
            pts_obj, normals_obj, _ = visible_model_cloud(
                self.renderer,
                T,
                K_vmc,
                size_vmc,
                det_mask=mask_vmc,
                mask_dilate_px=p.mask_dilate_px,
                max_points=p.max_model_points,
            )
            if len(pts_obj) < p.min_scene_points:
                return self._rejected(
                    hyp, T, None, reg, len(pts_cam), coverage, "no_overlap", t0, n_fallback
                )
            reg = register(
                pts_obj,
                normals_obj,
                pts_cam,
                normals_cam,
                T,
                variant=p.variant,
                max_corr_dist_mm=factor * d,
                max_iterations=p.max_iterations,
                robust_k_mm=p.robust_k_factor * d,
                src_pcd=src_for_reg,
            )
            if not reg.converged:
                return self._rejected(
                    hyp, T, None, reg, len(pts_cam), coverage, "icp_failed", t0, n_fallback
                )
            T = reg.T_camera_object
        assert reg is not None

        # Gate render: use ROI when the final pose still fits, otherwise full frame.
        if roi is not None and not roi.contains_sphere(T, view.K, d):
            n_fallback += 1
            # Re-render coarse at full frame so all three arrays share the same shape.
            render_c_full = self.renderer.render(T_coarse, view.K, size)
            render_r = self.renderer.render(T, view.K, size)
            gate = check_gate(
                T_start, T, self.model, det_mask, render_c_full, render_r, reg.fitness, p.gate,
                view.depth,
            )
            signals = self._signals(hyp.signals, reg, gate, coverage, gate.mask_refined, det_mask)
        else:
            render_r = self.renderer.render(T, K_use, size_use)
            gate = check_gate(
                T_start, T, self.model, mask_use, render_c, render_r, reg.fitness, p.gate,
                depth_use,
            )
            signals = self._signals(hyp.signals, reg, gate, coverage, gate.mask_refined, mask_use)
        source = f"{hyp.source}+{p.variant}"
        if gate.accepted:
            out = dataclasses.replace(
                hyp, T_camera_object=T, stage=Stage.REFINED, signals=signals, source=source
            )
        else:
            out = dataclasses.replace(
                hyp,
                stage=Stage.REFINED,
                signals=signals,
                source=source,
                rejection_reason=gate.reason,
            )
        return RefineOutcome(
            out, T, z_shift, n_overlap, gate, reg, len(pts_cam), coverage,
            time.perf_counter() - t0, n_fallback,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _signals(
        base: QualitySignals,
        reg: RegistrationResult,
        gate: GateDecision,
        coverage: float,
        mask_r: npt.NDArray[np.bool_],
        det_mask: npt.NDArray[np.bool_],
    ) -> QualitySignals:
        s = dataclasses.replace(base)
        s.icp_fitness = reg.fitness
        s.icp_rmse_mm = reg.inlier_rmse_mm
        s.depth_coverage = coverage
        s.silhouette_iou = gate.iou_refined
        s.displacement_mm = gate.displacement_mm
        s.displacement_deg = gate.displacement_deg
        n_render = int(mask_r.sum())
        s.visible_fraction = float((mask_r & det_mask).sum() / n_render) if n_render else 0.0
        return s

    def _rejected(
        self,
        hyp: PoseHypothesis,
        T_candidate: Mat4,
        gate: GateDecision | None,
        reg: RegistrationResult | None,
        n_scene: int,
        coverage: float,
        reason: str,
        t0: float,
        roi_fallback: int = 0,
    ) -> RefineOutcome:
        s = dataclasses.replace(hyp.signals)
        s.depth_coverage = coverage
        if reg is not None:
            s.icp_fitness, s.icp_rmse_mm = reg.fitness, reg.inlier_rmse_mm
        out = dataclasses.replace(
            hyp,
            stage=Stage.REFINED,
            signals=s,
            source=f"{hyp.source}+{self.params.variant}",
            rejection_reason=reason,
        )
        z_shift, n_overlap = self._z
        return RefineOutcome(
            out,
            T_candidate,
            z_shift,
            n_overlap,
            gate,
            reg,
            n_scene,
            coverage,
            time.perf_counter() - t0,
            roi_fallback,
        )
