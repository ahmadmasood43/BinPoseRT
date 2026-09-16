"""Depth-based Refinement (D8): visibility-aware crop, ICP variants, independent acceptance gate."""

from binposert.refine.cloud import scene_cloud
from binposert.refine.crop import visible_model_cloud
from binposert.refine.depth_init import translation_from_depth
from binposert.refine.gate import (
    GateDecision,
    GateParams,
    boundary_error_px,
    check_gate,
    gate_reason,
    silhouette_iou,
    visible_silhouette,
)
from binposert.refine.icp import VARIANTS, RegistrationResult, register
from binposert.refine.refiner import RefineOutcome, Refiner, RefinerParams

__all__ = [
    "VARIANTS",
    "GateDecision",
    "GateParams",
    "RefineOutcome",
    "Refiner",
    "RefinerParams",
    "RegistrationResult",
    "boundary_error_px",
    "check_gate",
    "gate_reason",
    "register",
    "scene_cloud",
    "silhouette_iou",
    "translation_from_depth",
    "visible_model_cloud",
    "visible_silhouette",
]
