"""The ``nbv`` stage (D14, D12): the active-view loop of :mod:`binposert.active` run over every
scene of the dataset, one episode per scene.

The stage stands in for ``associate → fuse → confidence`` — it runs the same functions after
every unlocked View — so its cache hash carries the multiview, fusion and confidence configs
(with the model fingerprints) next to its own. It writes what ``evaluate`` reads from a
confidence stage: ``fused.parquet`` (the final FusedPoses with ``confidence`` / ``verdict``),
``hypotheses.parquet`` (those poses projected into the episode's reference Views, with
``confidence`` as a column) and ``groups.json`` (``{scene: [reference Views]}``), plus
``tracks.parquet`` (the final association), ``episodes.json`` (every step of every episode:
Views used, Verdict counts, candidate scores, the chosen View and why) and ``nbv_summary.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from binposert.active import (
    BeliefUpdater,
    LoopParams,
    NbvScoreParams,
    NbvScorer,
    episode_summary,
    run_episode,
)
from binposert.confidence import ConfidenceModel, VerdictThresholds
from binposert.data import BopDataset
from binposert.pipeline.artefacts import HYPOTHESIS_COLUMNS, read_hypotheses_table
from binposert.pipeline.confidence_stage import (
    ConfidenceStageParams,
    model_fingerprints,
)
from binposert.pipeline.confidence_stage import (
    params_from_config as confidence_params_from_config,
)
from binposert.pipeline.multiview_stage import (
    EXTRINSIC_COLUMNS,
    FUSED_FILE,
    GROUPS_FILE,
    TRACKS_FILE,
    AssociateStageParams,
    FuseStageParams,
    associate_params_from_config,
    fuse_params_from_config,
)
from binposert.pipeline.pool import map_scenes

log = logging.getLogger("binposert.pipeline")

EPISODES_FILE = "episodes.json"


@dataclass(frozen=True)
class NbvStageParams:
    loop: LoopParams
    score: NbvScoreParams
    associate: AssociateStageParams
    fuse: FuseStageParams
    confidence: ConfidenceStageParams


def params_from_config(cfg: dict[str, Any], repo_root: Path) -> NbvStageParams:
    p = dict(cfg["nbv"].get("params", {}))
    score = NbvScoreParams(**p.pop("score", {}))
    if "uncertain" in p:
        p["uncertain"] = tuple(str(u) for u in p["uncertain"])
    return NbvStageParams(
        loop=LoopParams(**p),
        score=score,
        associate=associate_params_from_config(cfg["multiview"], repo_root),
        fuse=fuse_params_from_config(cfg["fusion"]),
        confidence=confidence_params_from_config(cfg["confidence"], repo_root),
    )


def stage_config(cfg: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """The hashable config of the stage: its own section plus the three it stands in for."""
    from binposert.pipeline.stages import hashable_config

    return {
        "nbv": hashable_config(cfg["nbv"]),
        "multiview": hashable_config(cfg["multiview"]),
        "fusion": hashable_config(cfg["fusion"]),
        "confidence": {
            **hashable_config(cfg["confidence"]),
            "fingerprints": model_fingerprints(cfg["confidence"], repo_root),
        },
    }


def nbv_scene(
    scene_id: int, dataset: BopDataset, pose_dir: str, params: NbvStageParams
) -> dict[str, Any]:
    hyps = read_hypotheses_table(pose_dir)
    updater = BeliefUpdater(
        scene_id,
        dataset,
        hyps,
        params.associate,
        params.fuse,
        ConfidenceModel.load(params.confidence.model_h),
        ConfidenceModel.load(params.confidence.model_f),
        VerdictThresholds.load(params.confidence.thresholds),
    )
    scorer = NbvScorer(updater.models, params.score)
    pool = dataset.image_ids(scene_id)
    e = run_episode(scene_id, dataset, pool, updater, scorer, params.loop)
    fused = e.final.fused
    tracks = e.final.tracks
    return {
        "episode": episode_summary(e),
        "fused": fused.to_dict("records") if len(fused) else [],
        "tracks": tracks.to_dict("records") if len(tracks) else [],
        "hypotheses": e.hyp_rows,
    }


def run_nbv(
    dataset: BopDataset,
    pose_dir: Path,
    out_dir: Path,
    params: NbvStageParams,
    n_workers: int = 1,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    scene_ids = dataset.scene_ids
    jobs = [(sid, dataset, str(pose_dir), params) for sid in scene_ids]
    if n_workers > 1 and len(scene_ids) > 1:
        results = map_scenes(nbv_scene, jobs, n_workers)
    else:
        results = [nbv_scene(*job) for job in jobs]
    episodes = [r["episode"] for r in results]
    fused = pd.DataFrame([row for r in results for row in r["fused"]])
    tracks = pd.DataFrame([row for r in results for row in r["tracks"]])
    hyps = pd.DataFrame(
        [row for r in results for row in r["hypotheses"]],
        columns=[*HYPOTHESIS_COLUMNS, "confidence", "verdict"],
    )
    for df, cols in ((fused, ("verdict", "joint_reason")), (tracks, ("rejection_reason",))):
        for c in cols:
            if c in df:
                df[c] = df[c].astype(object)
    hyps["rejection_reason"] = hyps["rejection_reason"].astype(object)
    hyps["verdict"] = hyps["verdict"].astype(object)
    if len(tracks) == 0:
        tracks = pd.DataFrame(
            columns=[*HYPOTHESIS_COLUMNS, "group_id", "track_id", "weight", *EXTRINSIC_COLUMNS]
        )
    fused.to_parquet(out_dir / FUSED_FILE, index=False)
    tracks.to_parquet(out_dir / TRACKS_FILE, index=False)
    hyps.to_parquet(out_dir / "hypotheses.parquet", index=False)
    with open(out_dir / GROUPS_FILE, "w") as f:
        json.dump({str(e["scene_id"]): [e["reference"]] for e in episodes}, f)
    with open(out_dir / EPISODES_FILE, "w") as f:
        json.dump(episodes, f, indent=1)
    n_used = [e["n_views_used"] for e in episodes]
    stops: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for e in episodes:
        stops[e["stop"]] = stops.get(e["stop"], 0) + 1
        for s in e["steps"]:
            if s["chosen"] is not None:
                reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
    verdicts: dict[str, int] = {}
    for e in episodes:
        for k, v in e["verdicts"].items():
            verdicts[k] = verdicts.get(k, 0) + int(v)
    summary = {
        "policy": params.loop.policy,
        "budget": params.loop.budget,
        "stop": params.loop.stop,
        "start_group": params.loop.start_group,
        "n_episodes": len(episodes),
        "views_used": {
            "mean": float(np.mean(n_used)) if n_used else None,
            "min": int(min(n_used)) if n_used else None,
            "max": int(max(n_used)) if n_used else None,
        },
        "stops": stops,
        "choices": reasons,
        "n_tracks": int(len(fused)),
        "verdicts": verdicts,
        "n_projected_hypotheses": int(len(hyps)),
        "seconds_belief": float(sum(e["seconds_belief"] for e in episodes)),
        "seconds_score": float(sum(e["seconds_score"] for e in episodes)),
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(out_dir / "nbv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = [
    "EPISODES_FILE",
    "NbvStageParams",
    "nbv_scene",
    "params_from_config",
    "run_nbv",
    "stage_config",
]
