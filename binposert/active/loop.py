"""The active-view loop (D14): 1 View → associate → fuse → Confidence → Verdict; unlock a
Candidate Viewpoint; repeat until ``accept`` / ``reject`` everywhere, exhaustion or the budget.

An *episode* is one Scene, a start View and a pool of Candidate Viewpoints (the Scene's other
real Views). A *policy* picks the next View from the pool:

- ``nbv``: the argmax of the D14 score (:mod:`binposert.active.score`); on an all-zero score
  (no tracks yet, or nothing renders) it falls back to the fixed order and says so.
- ``random``: a permutation of the pool drawn once per episode from ``(seed, scene, start)``,
  so a budget-``k`` row's Views are a prefix of the budget-``k + 1`` row's.
- ``fixed``: the strided order Gamma's view groups use (``start, start + n//4, …``, then the rest),
  so ``fixed`` with budget 4 is exactly the A6 / A8 ``k = 4`` group that starts there.
- ``oracle``: the View whose belief update yields the most distinct ground-truth instances of the
  reference Views matched by a successful FusedPose (``MSSD < 0.1 d``). It reads the annotations
  and is an *upper bound* on what any view choice can gain, never a policy a robot could run.

Stopping: ``budget`` unlocks Views until the budget is spent (the AR-vs-views curve);
``verdict`` (the D14 loop proper) stops as soon as no ObjectTrack's Verdict is in ``uncertain``
(``request_view`` by default), or when the pool is exhausted or the budget spent.

The belief after every step is exactly what the ``associate → fuse → confidence`` stages would
produce for that set of Views (the same functions, on in-memory tables); the FusedPoses of the
final step are projected into the episode's *reference Views* — fixed per episode, the same for
every policy — so ``evaluate`` compares policies on the same images.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from binposert.active.score import CandidateScore, NbvScorer, TrackBelief, track_beliefs
from binposert.confidence import ConfidenceModel, VerdictThresholds
from binposert.data import BopDataset
from binposert.pipeline.artefacts import columns_to_transform
from binposert.pipeline.confidence_stage import P_H_COLUMNS, fused_features
from binposert.pipeline.multiview_stage import (
    EXTRINSIC_COLUMNS,
    AssociateStageParams,
    FuseStageParams,
    associate_scene,
    fuse_tracks,
)
from binposert.types import ObjectModel, Verdict, View

POLICIES = ("nbv", "random", "fixed", "oracle")
STOP_RULES = ("budget", "verdict")


@dataclass(frozen=True)
class LoopParams:
    policy: str = "nbv"
    budget: int = 4  # Views per episode including the start; 0 = the whole pool
    stop: str = "budget"  # "budget" | "verdict" (D14: stop when no track requests a View)
    uncertain: tuple[str, ...] = ("request_view",)  # Verdicts that keep the loop going
    start_group: int = 0  # the strided group whose first image starts the episode
    n_reference_views: int = 4  # reference Views = the strided group of that size at start_group
    seed: int = 0


@dataclass
class Belief:
    """The Scene's belief after a set of Views: the stage tables and the score's view of them."""

    image_ids: list[int]
    tracks: pd.DataFrame  # rows of tracks.parquet (one group)
    fused: pd.DataFrame  # rows of fused.parquet with confidence / verdict
    beliefs: list[TrackBelief]
    seconds: float


@dataclass
class StepRecord:
    step: int
    image_ids: list[int]
    n_tracks: int
    verdicts: dict[str, int]
    chosen: int | None
    reason: str  # "nbv" | "random" | "fixed" | "fallback" | "stop:<why>"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    seconds_belief: float = 0.0
    seconds_score: float = 0.0


@dataclass
class Episode:
    scene_id: int
    start: int
    pool: list[int]
    reference: list[int]
    steps: list[StepRecord]
    final: Belief
    hyp_rows: list[dict[str, Any]]  # the final FusedPoses projected into the reference Views

    @property
    def views_used(self) -> list[int]:
        return list(self.final.image_ids)


class BeliefUpdater:
    """``associate → fuse → confidence`` for one Scene on in-memory tables."""

    def __init__(
        self,
        scene_id: int,
        dataset: BopDataset,
        hyps: pd.DataFrame,
        assoc: AssociateStageParams,
        fuse: FuseStageParams,
        model_h: ConfidenceModel,
        model_f: ConfidenceModel,
        thresholds: VerdictThresholds,
    ) -> None:
        if fuse.joint_icp is not None:
            raise ValueError("the active loop does not run the joint ICP polish (D10 step 4)")
        self.scene_id = scene_id
        self.ds = dataset
        self.hyps = hyps[hyps.scene_id == scene_id]
        self.assoc = assoc
        self.fuse = fuse
        self.model_h, self.model_f, self.thresholds = model_h, model_f, thresholds
        self.models: dict[int, ObjectModel] = {
            int(o): dataset.load_model(int(o)) for o in sorted(self.hyps.object_id.unique())
        }

    def update(self, image_ids: list[int], project_into: list[int]) -> tuple[Belief, list[dict]]:
        t0 = time.perf_counter()
        rows, _ = associate_scene(
            self.scene_id, self.ds, "", [list(image_ids)], self.assoc, hyps=self.hyps
        )
        tracks = pd.DataFrame(rows)
        if len(tracks) == 0:
            empty = pd.DataFrame()
            return Belief(list(image_ids), tracks, empty, [], time.perf_counter() - t0), []
        tracks["rejection_reason"] = tracks["rejection_reason"].astype(object)
        fused_rows, hyp_rows = fuse_tracks(
            self.scene_id,
            self.ds,
            tracks,
            [list(image_ids)],
            self.fuse,
            project_images=[list(project_into)],
        )
        fused = pd.DataFrame(fused_rows)
        feats = fused_features(fused, tracks, self.ds, self.model_h)
        confidence = self.model_f.predict_proba(feats)
        for c in P_H_COLUMNS:
            fused[c] = feats[c].to_numpy()
        fused["confidence"] = confidence
        fused["verdict"] = [v.value for v in self.thresholds.verdicts(confidence)]
        conf_of = dict(zip(fused["track_id"].astype(int), fused["confidence"], strict=True))
        verdict_of = dict(zip(fused["track_id"].astype(int), fused["verdict"], strict=True))
        for h in hyp_rows:
            h["confidence"] = float(conf_of[int(h["hypothesis_id"])])
            h["verdict"] = str(verdict_of[int(h["hypothesis_id"])])
        members: dict[int, list] = {}
        for _, r in tracks.iterrows():
            T_wc = np.asarray([float(r[c]) for c in EXTRINSIC_COLUMNS]).reshape(4, 4)
            members.setdefault(int(r["track_id"]), []).append(
                (T_wc @ columns_to_transform(r), T_wc)
            )
        beliefs = track_beliefs(
            [
                {
                    "track_id": int(r["track_id"]),
                    "object_id": int(r["object_id"]),
                    "T_world_object": columns_to_transform(r),
                    "confidence": float(r["confidence"]),
                }
                for _, r in fused.iterrows()
            ],
            members,
            self.models,
        )
        return Belief(list(image_ids), tracks, fused, beliefs, time.perf_counter() - t0), hyp_rows


def strided_order(pool: list[int], start_group: int, n_views: int) -> list[int]:
    """The strided group of ``n_views`` starting at index ``start_group``, then the rest of the
    pool in id order — the ``fixed`` policy's order and the source of the reference Views."""
    ids = sorted(pool)
    m = max(1, len(ids) // max(1, n_views))
    g = start_group % m
    head = [ids[g + j * m] for j in range(n_views) if g + j * m < len(ids)]
    return head + [i for i in ids if i not in head]


def verdict_counts(fused: pd.DataFrame) -> dict[str, int]:
    out = {v.value: 0 for v in Verdict}
    if len(fused):
        for v, n in fused["verdict"].value_counts().items():
            out[str(v)] = int(n)
    return out


def run_episode(
    scene_id: int,
    dataset: BopDataset,
    pool: list[int],
    updater: BeliefUpdater,
    scorer: NbvScorer,
    p: LoopParams,
) -> Episode:
    if p.policy not in POLICIES:
        raise ValueError(f"unknown policy {p.policy!r}; choose from {POLICIES}")
    if p.stop not in STOP_RULES:
        raise ValueError(f"unknown stop rule {p.stop!r}; choose from {STOP_RULES}")
    order = strided_order(pool, p.start_group, p.n_reference_views)
    reference = order[: p.n_reference_views]
    start = order[0]
    budget = len(pool) if p.budget <= 0 else min(p.budget, len(pool))
    rng = np.random.default_rng([p.seed, int(scene_id), int(start)])
    random_order = [start] + [int(i) for i in rng.permutation([i for i in order if i != start])]
    views_cache: dict[int, View] = {}

    def view(image_id: int) -> View:
        if image_id not in views_cache:
            v, _ = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
            views_cache[image_id] = v
        return views_cache[image_id]

    used = [start]
    steps: list[StepRecord] = []
    belief, hyp_rows = updater.update(used, reference)
    while True:
        rec = StepRecord(
            step=len(steps),
            image_ids=list(used),
            n_tracks=len(belief.fused),
            verdicts=verdict_counts(belief.fused),
            chosen=None,
            reason="",
            seconds_belief=belief.seconds,
        )
        remaining = [i for i in order if i not in used]
        if len(used) >= budget:
            rec.reason = "stop:budget"
        elif not remaining:
            rec.reason = "stop:exhausted"
        elif p.stop == "verdict" and not any(rec.verdicts.get(u, 0) for u in p.uncertain):
            rec.reason = "stop:verdict"
        if rec.reason:
            steps.append(rec)
            break
        if p.policy == "fixed":
            rec.chosen, rec.reason = remaining[0], "fixed"
        elif p.policy == "random":
            rec.chosen = next(i for i in random_order if i not in used)
            rec.reason = "random"
        elif p.policy == "oracle":
            t0 = time.perf_counter()
            gains = [(oracle_gain(updater, used + [i], reference, dataset), i) for i in remaining]
            rec.seconds_score = time.perf_counter() - t0
            rec.candidates = [{"image_id": i, "score": float(g)} for g, i in gains]
            best_gain = max(g for g, _ in gains)
            rec.chosen = next(i for g, i in gains if g == best_gain)  # ties: fixed order
            rec.reason = "oracle"
        else:
            t0 = time.perf_counter()
            scores = scorer.score_candidates(scene_id, [view(i) for i in remaining], belief.beliefs)
            rec.seconds_score = time.perf_counter() - t0
            rec.candidates = [_candidate_row(c) for c in scores]
            best = max(scores, key=lambda c: c.score)
            if best.score > 0.0:
                rec.chosen, rec.reason = best.image_id, "nbv"
            else:
                rec.chosen, rec.reason = remaining[0], "fallback"
        steps.append(rec)
        used.append(int(rec.chosen))
        belief, hyp_rows = updater.update(used, reference)
    return Episode(scene_id, start, sorted(pool), reference, steps, belief, hyp_rows)


def oracle_gain(
    updater: BeliefUpdater, image_ids: list[int], reference: list[int], dataset: BopDataset
) -> int:
    """Distinct annotated instances of the reference Views that a successful FusedPose of the
    belief after ``image_ids`` matches (recall, not precision: a View that adds spurious tracks
    is not punished, one that fixes a real object is rewarded)."""
    from binposert.confidence.labels import label_fused

    belief, _ = updater.update(image_ids, reference)
    if len(belief.fused) == 0:
        return 0
    fused = belief.fused.copy()
    fused["image_ids"] = ",".join(str(i) for i in reference)  # label against the reference GT
    # 500 sampled surface points: the label is a ranking proxy, not the published metric
    labelled = label_fused(fused, dataset, n_model_points=500)
    ok = labelled[labelled.success.astype(bool)]
    return int(len({(int(a), int(b)) for a, b in zip(ok.gt_image_id, ok.gt_index, strict=True)}))


def _candidate_row(c: CandidateScore) -> dict[str, Any]:
    return {
        "image_id": c.image_id,
        "score": c.score,
        "mean_disagreement": c.mean_disagreement,
        "mean_visible": c.mean_visible,
        "n_tracks": c.n_tracks,
        "seconds": c.seconds,
    }


def episode_summary(e: Episode) -> dict[str, Any]:
    return {
        "scene_id": e.scene_id,
        "start": e.start,
        "pool": e.pool,
        "reference": e.reference,
        "views_used": e.views_used,
        "n_views_used": len(e.views_used),
        "stop": e.steps[-1].reason,
        "n_tracks": len(e.final.fused),
        "verdicts": verdict_counts(e.final.fused),
        "seconds_belief": float(sum(s.seconds_belief for s in e.steps)),
        "seconds_score": float(sum(s.seconds_score for s in e.steps)),
        "steps": [
            {
                "step": s.step,
                "image_ids": s.image_ids,
                "n_tracks": s.n_tracks,
                "verdicts": s.verdicts,
                "chosen": s.chosen,
                "reason": s.reason,
                "candidates": s.candidates,
                "seconds_belief": s.seconds_belief,
                "seconds_score": s.seconds_score,
            }
            for s in e.steps
        ],
    }


__all__ = [
    "POLICIES",
    "STOP_RULES",
    "Belief",
    "BeliefUpdater",
    "Episode",
    "LoopParams",
    "StepRecord",
    "episode_summary",
    "oracle_gain",
    "run_episode",
    "strided_order",
    "verdict_counts",
]
