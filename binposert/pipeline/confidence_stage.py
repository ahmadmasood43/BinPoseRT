"""The ``confidence`` stage (D11, D12): every FusedPose gets a Confidence and a Verdict.

Model H scores the members of every ObjectTrack (the refined PoseHypotheses in ``tracks.parquet``);
their probabilities are aggregated per track (mean / min / max, share of Refinement-rejected
members) and joined to the track's aggregated QualitySignals, which Model F turns into the
published Confidence. The Verdict follows from the fitted thresholds. The stage rewrites the fuse
stage's two tables with the new columns — ``fused.parquet`` (one row per FusedPose, now with
``confidence`` / ``verdict``) and ``hypotheses.parquet`` (the fused pose projected into every View,
with ``confidence`` as a column so ``evaluate`` can rank predictions by it) — and copies
``groups.json`` so ``evaluate`` scores the same images.

A single-view pass-through row (``fusion=none``) has no FusedPoses: its hypotheses are scored by
Model H directly and the Verdict uses the same thresholds, which is what the single-view rows of
the calibration analysis report.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from binposert.confidence import ConfidenceModel, VerdictThresholds, file_fingerprint
from binposert.confidence.schema import hypothesis_extras
from binposert.data import BopDataset
from binposert.pipeline.artefacts import read_hypotheses_table, read_stage_info
from binposert.pipeline.multiview_stage import FUSED_FILE, GROUPS_FILE, TRACKS_FILE
from binposert.types import Verdict

log = logging.getLogger("binposert.pipeline")

TRACK_KEYS = ["scene_id", "group_id", "track_id"]
P_H_COLUMNS = ["p_h_mean", "p_h_min", "p_h_max", "rejected"]


@dataclass(frozen=True)
class ConfidenceStageParams:
    model_h: Path
    model_f: Path
    thresholds: Path


def params_from_config(section: dict[str, Any], repo_root: Path) -> ConfidenceStageParams:
    p = dict(section.get("params", {}))

    def _path(key: str) -> Path:
        v = Path(str(p[key]))
        return v if v.is_absolute() else repo_root / v

    return ConfidenceStageParams(_path("model_h"), _path("model_f"), _path("thresholds"))


def model_fingerprints(section: dict[str, Any], repo_root: Path) -> dict[str, str]:
    """Content hashes of the model files a config names, for the stage hash (D12: the cache must
    miss when a model is refitted, and the config only carries the path)."""
    params = params_from_config(section, repo_root)
    return {
        "model_h": file_fingerprint(params.model_h),
        "model_f": file_fingerprint(params.model_f),
        "thresholds": file_fingerprint(params.thresholds),
    }


def with_diameter(table: pd.DataFrame, dataset: BopDataset) -> pd.DataFrame:
    out = table.copy()
    diam = {int(o): dataset.load_model(int(o)).diameter for o in out["object_id"].unique()}
    out["diameter"] = out["object_id"].map(diam).astype(float)
    return out


def score_members(tracks: pd.DataFrame, dataset: BopDataset, model_h: ConfidenceModel) -> pd.Series:
    """Model H probability of every row of a tracks / hypotheses table."""
    t = hypothesis_extras(with_diameter(tracks, dataset))
    return pd.Series(model_h.predict_proba(t), index=tracks.index, name="p_h")


def aggregate_members(tracks: pd.DataFrame, p_h: pd.Series) -> pd.DataFrame:
    """Per-track Model H aggregates: mean / min / max probability and the rejected share."""
    t = tracks[TRACK_KEYS].copy()
    t["p_h"] = p_h.to_numpy()
    t["rejected"] = tracks["rejection_reason"].notna().astype(float).to_numpy()
    g = t.groupby(TRACK_KEYS)
    out = pd.DataFrame(
        {
            "p_h_mean": g["p_h"].mean(),
            "p_h_min": g["p_h"].min(),
            "p_h_max": g["p_h"].max(),
            "rejected": g["rejected"].mean(),
        }
    ).reset_index()
    return out


def fused_features(
    fused: pd.DataFrame, tracks: pd.DataFrame, dataset: BopDataset, model_h: ConfidenceModel
) -> pd.DataFrame:
    """The fused table with the Model F extras (diameter, member aggregates) joined in."""
    p_h = score_members(tracks, dataset, model_h)
    agg = aggregate_members(tracks, p_h)
    out = with_diameter(fused, dataset).merge(agg, on=TRACK_KEYS, how="left", validate="one_to_one")
    missing = out[P_H_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} FusedPoses have no members in tracks.parquet")
    return out


def check_weight_source(assoc_dir: Path, model_f: ConfidenceModel) -> str:
    """Model F reads the fusion weights (``weight_sum``, the reference member, the dispersion);
    it is only calibrated for the weighting it was fitted on. The associate stage's provenance
    says which weighting produced the tracks; the model's provenance which one it was fitted on
    (``product`` when unrecorded, as for the v1 models). A mismatch is logged, not refused: the
    A8w ablation row is exactly that case and its de-calibration is the finding."""
    try:
        cfg = read_stage_info(assoc_dir).get("config", {})
        used = str(cfg.get("params", {}).get("weights", {}).get("source", "product"))
    except (OSError, ValueError):
        used = "unknown"
    fitted = str(model_f.provenance.get("weight_source", "product"))
    if used != fitted:
        log.warning(
            "Model F was fitted on %s-weighted tracks but the association used %s weights: "
            "its Confidence is not calibrated for this row (refit on tracks of this weighting)",
            fitted,
            used,
        )
    return used


def run_confidence(
    dataset: BopDataset,
    assoc_dir: Path,
    fuse_dir: Path,
    out_dir: Path,
    params: ConfidenceStageParams,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model_h = ConfidenceModel.load(params.model_h)
    model_f = ConfidenceModel.load(params.model_f)
    thresholds = VerdictThresholds.load(params.thresholds)
    weight_source = check_weight_source(assoc_dir, model_f)
    fused = pd.read_parquet(fuse_dir / FUSED_FILE)
    hyps = read_hypotheses_table(fuse_dir)
    tracks = pd.read_parquet(assoc_dir / TRACKS_FILE)
    summary: dict[str, Any] = {
        "model_h": str(params.model_h),
        "model_f": str(params.model_f),
        "thresholds": thresholds.to_dict(),
        "weight_source": weight_source,
    }
    if len(fused):
        feats = fused_features(fused, tracks, dataset, model_h)
        confidence = model_f.predict_proba(feats)
        fused = fused.copy()
        for c in P_H_COLUMNS:
            fused[c] = feats[c].to_numpy()
        fused["confidence"] = confidence
        fused["verdict"] = [v.value for v in thresholds.verdicts(confidence)]
        # the projected hypotheses of a track carry hypothesis_id == track_id (track ids are unique
        # within a scene across its groups)
        key = fused[["scene_id", "track_id", "confidence", "verdict"]].rename(
            columns={"track_id": "hypothesis_id"}
        )
        hyps = hyps.merge(key, on=["scene_id", "hypothesis_id"], how="left", validate="many_to_one")
        summary["scored"] = "fused"
    else:
        p_h = score_members(hyps, dataset, model_h)
        hyps = hyps.copy()
        hyps["confidence"] = p_h.to_numpy()
        hyps["verdict"] = [v.value for v in thresholds.verdicts(hyps["confidence"].to_numpy())]
        summary["scored"] = "hypotheses"
    if hyps["confidence"].isna().any():
        raise ValueError("some projected hypotheses received no Confidence")
    hyps["rejection_reason"] = hyps["rejection_reason"].astype(object)
    hyps["verdict"] = hyps["verdict"].astype(object)
    hyps.to_parquet(out_dir / "hypotheses.parquet", index=False)
    if "verdict" in fused:
        fused["verdict"] = fused["verdict"].astype(object)
    for c in ("joint_reason",):
        if c in fused:
            fused[c] = fused[c].astype(object)
    fused.to_parquet(out_dir / FUSED_FILE, index=False)
    shutil.copyfile(fuse_dir / GROUPS_FILE, out_dir / GROUPS_FILE)
    scored = fused if len(fused) else hyps
    conf = scored["confidence"].to_numpy(dtype=float)
    verdicts = scored["verdict"].astype(str)
    summary.update(
        {
            "n_fused": int(len(fused)),
            "n_hypotheses": int(len(hyps)),
            "mean_confidence": float(np.mean(conf)) if len(conf) else None,
            "verdict_rates": {
                v.value: float((verdicts == v.value).mean()) if len(verdicts) else None
                for v in Verdict
            },
            "wall_seconds": time.perf_counter() - t0,
        }
    )
    with open(out_dir / "confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = [
    "P_H_COLUMNS",
    "ConfidenceStageParams",
    "aggregate_members",
    "fused_features",
    "model_fingerprints",
    "params_from_config",
    "run_confidence",
    "score_members",
    "with_diameter",
]
