"""Association (D10 step 1): the PoseHypotheses of several Views of one Scene -> ObjectTracks.

Views are visited one at a time. Each existing ObjectTrack is represented by its highest-weight
member expressed in the world frame; a View's hypotheses of the same ``object_id`` are assigned to
tracks one-to-one by the Hungarian algorithm on a symmetry-aware SE(3) cost, restricted to pairs
that pass the geometric gate (translation within ``gate_t_factor × diameter`` and symmetry-aware
rotation within ``gate_rot_deg``). Hypotheses that no track can take start a new track, so nothing
is ever discarded here; a track with a single member is simply a single-view ObjectTrack.

The gate is what keeps repeated objects apart: two copies of one ObjectModel a few centimetres
apart are never confused because the translation gate is a fraction of the diameter, while one
copy seen from two Views lands on the same track because calibrated extrinsics put both world
poses within sensor noise of each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.transforms import Mat4, translation_distance
from binposert.types import ObjectModel, ObjectTrack, PoseHypothesis

INF = float("inf")


@dataclass(frozen=True)
class AssociationParams:
    gate_t_factor: float = 0.5  # translation gate as a fraction of the diameter
    gate_rot_deg: float = 45.0  # symmetry-aware rotation gate
    rot_weight_mm_per_deg: float = 0.5  # cost = translation (mm) + weight * rotation (deg)
    view_order: str = "given"  # "given" | "most_hypotheses": which View seeds the tracks


@dataclass
class WorldHypothesis:
    """A PoseHypothesis lifted into the world frame with its fusion weight."""

    hypothesis: PoseHypothesis
    T_world_object: Mat4
    weight: float
    T_world_camera: Mat4


@dataclass
class AssociationResult:
    tracks: list[ObjectTrack]
    members: dict[int, list[WorldHypothesis]] = field(default_factory=dict)  # by track_id
    n_gated_out: int = 0  # (track, hypothesis) pairs the geometric gate excluded
    n_new_tracks_after_first_view: int = 0  # hypotheses no existing track could take

    def track_of(self, hyp: PoseHypothesis) -> int:
        for t in self.tracks:
            if any(h is hyp for h in t.hypotheses):
                return t.track_id
        raise KeyError("hypothesis is not in any track")


def lift_to_world(hyp: PoseHypothesis, T_world_camera: Mat4, weight: float) -> WorldHypothesis:
    return WorldHypothesis(hyp, T_world_camera @ hyp.T_camera_object, weight, T_world_camera)


def associate(
    hyps_by_view: dict[str, list[WorldHypothesis]],
    models: dict[int, ObjectModel],
    params: AssociationParams | None = None,
    first_track_id: int = 0,
) -> AssociationResult:
    """Group the world-frame hypotheses of every View into ObjectTracks, per ``object_id``.

    ``hyps_by_view`` maps ``camera_id`` to that View's hypotheses (already lifted to the world
    frame). Views are visited in the dict's order unless ``params.view_order`` says otherwise.
    """
    p = params or AssociationParams()
    order = list(hyps_by_view)
    if p.view_order == "most_hypotheses":
        order.sort(key=lambda c: -len(hyps_by_view[c]))
    elif p.view_order != "given":
        raise ValueError(f"unknown view_order {p.view_order!r}")

    object_ids = sorted({w.hypothesis.object_id for ws in hyps_by_view.values() for w in ws})
    result = AssociationResult(tracks=[])
    next_id = first_track_id
    for object_id in object_ids:
        model = models[object_id]
        tracks: list[ObjectTrack] = []
        reps: list[WorldHypothesis] = []  # highest-weight member of each track
        for i, camera_id in enumerate(order):
            ws = [w for w in hyps_by_view[camera_id] if w.hypothesis.object_id == object_id]
            if not ws:
                continue
            cost, gated = _cost_matrix(reps, ws, model, p)
            result.n_gated_out += gated
            assigned = _assign(cost)
            taken = set()
            for ti, hi in assigned:
                tracks[ti].hypotheses.append(ws[hi].hypothesis)
                result.members[tracks[ti].track_id].append(ws[hi])
                if ws[hi].weight > reps[ti].weight:
                    reps[ti] = ws[hi]
                taken.add(hi)
            for hi, w in enumerate(ws):
                if hi in taken:
                    continue
                t = ObjectTrack(track_id=next_id, object_id=object_id, hypotheses=[w.hypothesis])
                next_id += 1
                tracks.append(t)
                reps.append(w)
                result.members[t.track_id] = [w]
                if i > 0:
                    result.n_new_tracks_after_first_view += 1
        result.tracks.extend(tracks)
    return result


def _cost_matrix(
    reps: list[WorldHypothesis],
    ws: list[WorldHypothesis],
    model: ObjectModel,
    p: AssociationParams,
) -> tuple[np.ndarray, int]:
    cost = np.full((len(reps), len(ws)), INF)
    gated = 0
    max_t = p.gate_t_factor * model.diameter
    for ti, r in enumerate(reps):
        for hi, w in enumerate(ws):
            d_t = translation_distance(r.T_world_object, w.T_world_object)
            if d_t > max_t:
                gated += 1
                continue
            d_r = sym_aware_rotation_distance_deg(
                r.T_world_object, w.T_world_object, model.symmetry
            )
            if d_r > p.gate_rot_deg:
                gated += 1
                continue
            cost[ti, hi] = d_t + p.rot_weight_mm_per_deg * d_r
    return cost, gated


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian assignment on the finite entries of ``cost``; infinite pairs are never matched."""
    if cost.size == 0 or not np.isfinite(cost).any():
        return []
    big = float(np.nanmax(cost[np.isfinite(cost)])) * 10.0 + 1.0
    filled = np.where(np.isfinite(cost), cost, big)
    rows, cols = linear_sum_assignment(filled)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if np.isfinite(cost[r, c])]


def association_summary(result: AssociationResult) -> dict[str, Any]:
    sizes = [len(t.hypotheses) for t in result.tracks]
    return {
        "n_tracks": len(result.tracks),
        "n_hypotheses": int(sum(sizes)),
        "views_per_track": {
            str(k): int(sum(1 for s in sizes if s == k)) for k in sorted(set(sizes))
        },
        "n_gated_out": result.n_gated_out,
        "n_new_tracks_after_first_view": result.n_new_tracks_after_first_view,
    }
