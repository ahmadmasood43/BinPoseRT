"""Fusion (D10 steps 2–3): an ObjectTrack's world-frame hypotheses -> one FusedPose.

The reference is the highest-weight member; every other member is symmetry-aligned to it by a
right-multiplication in the object frame (D9, ADR-0003) *before* anything is averaged, so
hypotheses that landed in different symmetry branches (a 137° turn about a cylinder axis, a box
flipped end-over-end) contribute the same physical pose and not a rotation half-way between two
equivalent ones. The fused pose is then the weighted intrinsic mean on SE(3).

Weights are a hand-set product of per-view QualitySignals (segmentation score, ICP fitness, depth
coverage, visible fraction); a signal that is NaN simply drops out of the product. A hypothesis the
Refinement rejected still carries its coarse pose and takes part with a reduced weight, because a
track whose every member was rejected must still yield a pose. Delta replaces the product by Model
H's probability (D11) without touching this module's interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from binposert.multiview.association import WorldHypothesis
from binposert.symmetry import align_to_reference, sym_aware_rotation_distance_deg
from binposert.transforms import Mat4, translation_distance, weighted_mean_se3
from binposert.types import FusedPose, ObjectModel, ObjectTrack, QualitySignals, Verdict

METHODS = ("best", "mean")
# per-view signals whose weighted mean is carried onto the FusedPose
AGGREGATED_SIGNALS = (
    "seg_score",
    "pose_score",
    "icp_fitness",
    "icp_rmse_mm",
    "depth_coverage",
    "visible_fraction",
    "silhouette_iou",
)


@dataclass(frozen=True)
class WeightParams:
    """``weight = Π signal^exponent`` over the listed signals (NaN signals skipped), floored."""

    exponents: dict[str, float] = field(
        default_factory=lambda: {
            "seg_score": 1.0,
            "icp_fitness": 1.0,
            "depth_coverage": 1.0,
            "visible_fraction": 1.0,
        }
    )
    rejected_factor: float = 0.2  # multiplier for hypotheses the Refinement rejected
    floor: float = 1e-3


@dataclass(frozen=True)
class FusionParams:
    method: str = "mean"  # "best" (highest-weight member) | "mean" (weighted SE(3) mean)
    weights: WeightParams = field(default_factory=WeightParams)
    mean_iterations: int = 5


def hypothesis_weight(signals: QualitySignals, rejected: bool, p: WeightParams) -> float:
    w = 1.0
    for name, expo in p.exponents.items():
        v = float(getattr(signals, name))
        if math.isnan(v):
            continue
        w *= max(v, 0.0) ** expo
    if rejected:
        w *= p.rejected_factor
    return max(w, p.floor)


@dataclass
class FusionOutcome:
    fused: FusedPose
    T_reference: Mat4  # the highest-weight member, before alignment
    aligned: list[Mat4]  # every member after symmetry alignment to the reference
    weights: np.ndarray
    branch_index: list[int]  # symmetry element each member was aligned by (0 = none)

    @property
    def n_aligned(self) -> int:
        return int(sum(1 for b in self.branch_index if b != 0))


def fuse_track(
    track: ObjectTrack,
    members: list[WorldHypothesis],
    model: ObjectModel,
    params: FusionParams | None = None,
) -> FusionOutcome:
    """One FusedPose for a track; ``members`` are the track's world-frame hypotheses."""
    p = params or FusionParams()
    if p.method not in METHODS:
        raise ValueError(f"unknown fusion method {p.method!r}; choose from {METHODS}")
    if not members:
        raise ValueError("cannot fuse an empty track")
    w = np.asarray([m.weight for m in members], dtype=np.float64)
    ref_i = int(np.argmax(w))
    T_ref = members[ref_i].T_world_object
    aligned: list[Mat4] = []
    branches: list[int] = []
    for m in members:
        T_al, b = align_to_reference(T_ref, m.T_world_object, model.symmetry)
        aligned.append(T_al)
        branches.append(b)
    if p.method == "best" or len(members) == 1:
        T_fused = T_ref.copy()
    else:
        T_fused = weighted_mean_se3(aligned, w, T_ref=T_ref, iterations=p.mean_iterations)
    signals = fused_signals(members, aligned, w, T_fused, model)
    fused = FusedPose(
        track_id=track.track_id,
        object_id=track.object_id,
        T_world_object=T_fused,
        confidence=float("nan"),  # a ConfidenceModel (Delta) sets this
        verdict=Verdict.ACCEPT,
        signals=signals,
    )
    return FusionOutcome(fused, T_ref, aligned, w, branches)


def fused_signals(
    members: list[WorldHypothesis],
    aligned: list[Mat4],
    weights: np.ndarray,
    T_fused: Mat4,
    model: ObjectModel,
) -> QualitySignals:
    """Weighted means of the per-view signals plus the multi-view ones: view count and the
    weighted RMS dispersion of the aligned members about the fused pose."""
    s = QualitySignals()
    wn = weights / weights.sum()
    for name in AGGREGATED_SIGNALS:
        vals = np.asarray([float(getattr(m.hypothesis.signals, name)) for m in members])
        ok = ~np.isnan(vals)
        if ok.any():
            setattr(s, name, float(np.sum(vals[ok] * wn[ok]) / np.sum(wn[ok])))
    d_t = np.asarray([translation_distance(T_fused, T) for T in aligned])
    d_r = np.asarray([sym_aware_rotation_distance_deg(T_fused, T, model.symmetry) for T in aligned])
    s.n_views = float(len({m.hypothesis.camera_id for m in members}))
    s.dispersion_mm = float(np.sqrt(np.sum(wn * d_t**2)))
    s.dispersion_deg = float(np.sqrt(np.sum(wn * d_r**2)))
    return s


def fusion_summary(outcomes: list[FusionOutcome]) -> dict[str, Any]:
    n = len(outcomes)
    multi = [o for o in outcomes if o.fused.signals.n_views > 1]
    return {
        "n_tracks": n,
        "n_multi_view": len(multi),
        "n_members_aligned_by_symmetry": int(sum(o.n_aligned for o in outcomes)),
        "median_dispersion_mm": float(np.median([o.fused.signals.dispersion_mm for o in multi]))
        if multi
        else float("nan"),
        "median_dispersion_deg": float(np.median([o.fused.signals.dispersion_deg for o in multi]))
        if multi
        else float("nan"),
    }
