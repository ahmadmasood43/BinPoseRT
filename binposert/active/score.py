"""Next-best-view score over the Scene's real, not-yet-used Views (D14).

For every Candidate Viewpoint ``v`` and every ObjectTrack ``t`` of the current belief:

- ``U_t(v)``: the mean pairwise silhouette disagreement (``1 − IoU``) of the track's *hypothesis
  set* rendered into ``v``. The set holds the track's members (symmetry-aligned to the fused
  pose, the real samples) topped up with posterior samples of the FusedPose: the cached
  estimators leave one refined hypothesis per Detection, so a single-view track has no "top-K"
  of its own, and its uncertainty is what Beta measured — largest along the observing camera's
  ray ("right in the image, wrong in depth"). A sample therefore perturbs the fused pose in a
  member camera's frame with ``σ_ray`` along the optical axis, ``σ_lateral`` across it and
  ``σ_rot`` about the object origin, all in units of the diameter. Silhouettes that a View sees
  as different are what that View can decide; symmetric poses render identically and so, as
  they should, do not count as disagreement.
- ``V_t(v)``: the predicted visible fraction of the fused pose in ``v`` — its own silhouette
  pixels that no other track's fused pose occludes, over its silhouette area (0 when the
  object leaves the frame).
- ``S(v) = Σ_t w_t · U_t(v) · V_t(v)``: a View is worth unlocking in proportion to how much it
  can tell about the tracks the ConfidenceModel does not vouch for, ``w_t = 1 − Confidence_t``
  by default or the binary entropy of the Confidence (:func:`track_weight`).

Silhouettes are rendered at ``render_scale`` of the View's resolution with the CPU raycaster;
motion and collision costs are zero (D14).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from binposert.render import MeshRenderer, render_scene
from binposert.symmetry import align_to_reference
from binposert.transforms import Mat4, invert, make_T
from binposert.types import Mat3, ObjectModel, View

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NbvScoreParams:
    n_samples: int = 6  # hypothesis set size per track: members + posterior samples
    sigma_ray: float = 0.05  # translation noise along the observing camera's ray, x diameter
    sigma_lateral: float = 0.02  # across the ray, x diameter
    sigma_rot_deg: float = 10.0
    render_scale: float = 0.25  # silhouettes rendered at this fraction of the View's resolution
    seed: int = 0
    weight: str = "one_minus"  # track weight from its Confidence p: "one_minus" (1 − p) | "entropy"


def track_weight(confidence: float, kind: str) -> float:
    """``1 − p``: the probability the track is wrong (E4). ``entropy``: the binary entropy of
    ``p`` (in bits), largest where the ConfidenceModel is least sure — a track it already rejects
    (a spurious Detection, mostly) gains as little from a View as one it accepts."""
    p = float(np.clip(confidence, 0.0, 1.0))
    if kind == "one_minus":
        return 1.0 - p
    if kind == "entropy":
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    raise ValueError(f"unknown track weight {kind!r}; choose one_minus or entropy")


@dataclass
class TrackBelief:
    """What the score needs of one ObjectTrack: its fused pose, its uncertainty and the members
    (world poses, already symmetry-aligned to the fused pose) with the cameras that saw them."""

    track_id: int
    object_id: int
    T_world_object: Mat4
    confidence: float
    members: list[Mat4]
    member_cameras: list[Mat4]  # T_world_camera of each member's View


@dataclass(frozen=True)
class CandidateScore:
    image_id: int
    score: float
    mean_disagreement: float  # unweighted mean of U over the tracks
    mean_visible: float  # unweighted mean of V over the tracks
    n_tracks: int
    seconds: float


def track_beliefs(
    fused_rows: list[dict[str, Any]],
    member_rows: dict[int, list[tuple[Mat4, Mat4]]],
    models: dict[int, ObjectModel],
) -> list[TrackBelief]:
    """Beliefs from fused rows (``track_id``, ``object_id``, ``T_world_object``, ``confidence``)
    and ``{track_id: [(T_world_object_member, T_world_camera), …]}``; members are aligned to the
    fused pose here."""
    out = []
    for r in fused_rows:
        tid, oid = int(r["track_id"]), int(r["object_id"])
        T = np.asarray(r["T_world_object"], dtype=np.float64)
        sym = models[oid].symmetry
        members, cams = [], []
        for T_m, T_wc in member_rows.get(tid, []):
            members.append(align_to_reference(T, T_m, sym)[0])
            cams.append(T_wc)
        out.append(TrackBelief(tid, oid, T, float(r["confidence"]), members, cams))
    return out


def hypothesis_set(
    belief: TrackBelief, diameter: float, p: NbvScoreParams, rng: np.random.Generator
) -> list[Mat4]:
    """The track's members plus posterior samples of the fused pose, ``p.n_samples`` in all."""
    members = list(belief.members[: p.n_samples]) if belief.members else [belief.T_world_object]
    cams = list(belief.member_cameras) or [np.eye(4)]
    out = members
    d = float(diameter)
    for j in range(len(out), p.n_samples):
        T_wc = cams[j % len(cams)]
        T_co = invert(T_wc) @ belief.T_world_object
        dt = rng.normal(0.0, 1.0, 3) * np.array(
            [p.sigma_lateral * d, p.sigma_lateral * d, p.sigma_ray * d]
        )
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1e-12
        angle = np.radians(rng.normal(0.0, p.sigma_rot_deg))
        dR = _rodrigues(axis, angle)
        T_co2 = make_T(dR @ T_co[:3, :3], T_co[:3, 3] + dt)
        out.append(T_wc @ T_co2)
    return out


def _rodrigues(axis: F64, angle: float) -> F64:
    k = np.asarray(axis, dtype=np.float64)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def scaled_camera(
    K: Mat3, image_size: tuple[int, int], scale: float
) -> tuple[Mat3, tuple[int, int]]:
    """Intrinsics and size of the View down-sampled by ``scale`` (pixel-centre convention kept:
    ``u' = (u + 0.5) s − 0.5``)."""
    if scale >= 1.0:
        return np.asarray(K, dtype=np.float64), image_size
    h, w = image_size
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] = (K2[0, 2] + 0.5) * scale - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * scale - 0.5
    return K2, (max(1, int(round(h * scale))), max(1, int(round(w * scale))))


def silhouette_disagreement(masks: list[npt.NDArray[np.bool_]]) -> float:
    """Mean pairwise ``1 − IoU``; two empty silhouettes agree (IoU 1)."""
    n = len(masks)
    if n < 2:
        return 0.0
    areas = [int(m.sum()) for m in masks]
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            union = areas[i] + areas[j] - inter
            total += 0.0 if union == 0 else 1.0 - inter / union
            count += 1
    return total / count


class NbvScorer:
    """Scores Candidate Viewpoints for one Scene's belief; renderers are built once per model."""

    def __init__(self, models: dict[int, ObjectModel], params: NbvScoreParams | None = None):
        self.models = models
        self.p = params or NbvScoreParams()
        self._renderers: dict[int, MeshRenderer] = {}

    def renderer(self, object_id: int) -> MeshRenderer:
        if object_id not in self._renderers:
            self._renderers[object_id] = MeshRenderer.from_model(self.models[object_id])
        return self._renderers[object_id]

    def hypothesis_sets(self, beliefs: list[TrackBelief], scene_id: int) -> list[list[Mat4]]:
        """One hypothesis set per track, deterministic in ``(seed, scene_id, track_id)`` so every
        candidate View is scored on the same samples."""
        out = []
        for b in beliefs:
            rng = np.random.default_rng([self.p.seed, int(scene_id), int(b.track_id)])
            out.append(hypothesis_set(b, self.models[b.object_id].diameter, self.p, rng))
        return out

    def score_view(
        self, view: View, beliefs: list[TrackBelief], sets: list[list[Mat4]]
    ) -> tuple[CandidateScore, list[dict[str, float]]]:
        """Score of one candidate View and the per-track ``(U, V, weight)`` breakdown."""
        t0 = time.perf_counter()
        K, size = scaled_camera(view.K, view.image_size, self.p.render_scale)
        T_cw = invert(view.T_world_camera)
        if not beliefs:
            return CandidateScore(view.image_id, 0.0, 0.0, 0.0, 0, time.perf_counter() - t0), []
        # every fused pose as an occluder of every other
        placed = [(self.models[b.object_id], T_cw @ b.T_world_object) for b in beliefs]
        scene = render_scene(placed, K, size)
        rows: list[dict[str, float]] = []
        score = 0.0
        for i, (b, hyps) in enumerate(zip(beliefs, sets, strict=True)):
            r = self.renderer(b.object_id)
            own = r.render(T_cw @ b.T_world_object, K, size).mask
            area = int(own.sum())
            visible = float((own & (scene.geometry_ids == i)).sum() / area) if area else 0.0
            masks = [r.render(T_cw @ T, K, size).mask for T in hyps]
            u = silhouette_disagreement(masks)
            w = track_weight(b.confidence, self.p.weight)
            rows.append({"track_id": float(b.track_id), "U": u, "V": visible, "weight": w})
            score += w * u * visible
        cs = CandidateScore(
            image_id=int(view.image_id),
            score=float(score),
            mean_disagreement=float(np.mean([r["U"] for r in rows])),
            mean_visible=float(np.mean([r["V"] for r in rows])),
            n_tracks=len(beliefs),
            seconds=time.perf_counter() - t0,
        )
        return cs, rows

    def score_candidates(
        self, scene_id: int, candidates: list[View], beliefs: list[TrackBelief]
    ) -> list[CandidateScore]:
        sets = self.hypothesis_sets(beliefs, scene_id)
        return [self.score_view(v, beliefs, sets)[0] for v in candidates]


__all__ = [
    "CandidateScore",
    "NbvScoreParams",
    "NbvScorer",
    "TrackBelief",
    "hypothesis_set",
    "scaled_camera",
    "silhouette_disagreement",
    "track_beliefs",
    "track_weight",
]
