"""Acceptance gate (D8): rendered-silhouette agreement with the Detection mask, plus a displacement
cap relative to the coarse pose. ICP never optimises the silhouette, so the check is independent.

The silhouette is made occlusion-aware with the observed depth (as in BOP's VSD): a rendered pixel
whose rendered depth lies behind the observed surface by more than ``delta_mm`` is hidden by
another object and is not expected to appear in the (visible) Detection mask."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

from binposert.render import RenderResult
from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.transforms import Mat4, translation_distance
from binposert.types import ObjectModel


@dataclass(frozen=True)
class GateParams:
    """Defaults are the D8 design values; the tuned values live in ``configs/refiner/*.yaml``."""

    alpha: float = 0.25  # translation cap as a fraction of the diameter
    beta_deg: float = 30.0  # symmetry-aware rotation cap
    min_iou: float = 0.5  # rendered silhouette vs Detection mask at the refined pose
    max_iou_drop: float = 0.1  # refined IoU may not fall more than this below the coarse IoU
    min_fitness: float = 0.3  # fraction of observed (masked) points explained by the model
    delta_mm: float = 15.0  # occlusion tolerance for the visible silhouette (VSD's delta)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str | None
    iou_coarse: float
    iou_refined: float
    boundary_px: float
    displacement_mm: float
    displacement_deg: float
    mask_refined: "npt.NDArray[np.bool_] | None" = field(
        default=None, compare=False, hash=False
    )


def visible_silhouette(
    render: RenderResult, depth_obs: npt.NDArray[np.float64] | None, delta_mm: float
) -> npt.NDArray[np.bool_]:
    """Rendered pixels that are not hidden behind the observed surface."""
    if depth_obs is None:
        return render.mask
    hidden = (depth_obs > 0) & (render.depth > depth_obs + delta_mm)
    return render.mask & ~hidden


def silhouette_iou(render_mask: npt.NDArray[np.bool_], det_mask: npt.NDArray[np.bool_]) -> float:
    union = int((render_mask | det_mask).sum())
    return float((render_mask & det_mask).sum() / union) if union else 0.0


def boundary_error_px(render_mask: npt.NDArray[np.bool_], det_mask: npt.NDArray[np.bool_]) -> float:
    """Symmetric mean contour distance in pixels (each contour's mean distance to the other)."""
    if not render_mask.any() or not det_mask.any():
        return float("inf")
    out = []
    for a, b in ((render_mask, det_mask), (det_mask, render_mask)):
        contour = a ^ cv2.erode(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        dist_to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
        out.append(float(dist_to_b[contour].mean()) if contour.any() else 0.0)
    return float(np.mean(out))


def check_gate(
    T_start: Mat4,
    T_refined: Mat4,
    model: ObjectModel,
    det_mask: npt.NDArray[np.bool_],
    render_coarse: RenderResult,
    render_refined: RenderResult,
    fitness: float,
    params: GateParams,
    depth_obs: npt.NDArray[np.float64] | None = None,
) -> GateDecision:
    """``T_start`` is the pose registration started from (the coarse pose after the depth
    initialisation) and bounds the displacement; ``render_coarse`` is the silhouette of the coarse
    pose the stage received and is the baseline of the IoU-drop check."""
    mask_c = visible_silhouette(render_coarse, depth_obs, params.delta_mm)
    mask_r = visible_silhouette(render_refined, depth_obs, params.delta_mm)
    iou_c = silhouette_iou(mask_c, det_mask)
    iou_r = silhouette_iou(mask_r, det_mask)
    boundary = boundary_error_px(mask_r, det_mask)
    d_t = translation_distance(T_start, T_refined)
    d_r = sym_aware_rotation_distance_deg(T_start, T_refined, model.symmetry)
    reason = gate_reason(fitness, d_t, d_r, iou_c, iou_r, model.diameter, params)
    return GateDecision(reason is None, reason, iou_c, iou_r, boundary, d_t, d_r, mask_refined=mask_r)


def gate_reason(
    fitness: float,
    displacement_mm: float,
    displacement_deg: float,
    iou_coarse: float,
    iou_refined: float,
    diameter: float,
    params: GateParams,
) -> str | None:
    """The decision alone, from the measurements: ``None`` accepts, otherwise the first failed
    check in this fixed order. Re-applied offline to the stored measurements of a refine stage
    (``refine_details.parquet``) when tuning the gate, so both paths decide identically."""
    if fitness < params.min_fitness:
        return "fitness"
    if displacement_mm > params.alpha * diameter:
        return "displacement_translation"
    if displacement_deg > params.beta_deg:
        return "displacement_rotation"
    if iou_refined < params.min_iou:
        return "silhouette_iou"
    if iou_refined < iou_coarse - params.max_iou_drop:
        return "silhouette_iou_drop"
    return None
