"""Stage implementations and the registry the runner builds the DAG from (D12).

A stage declares what artefact kind it ``produces`` and which kinds it ``consumes``; the runner
wires each consumed kind to the most recent upstream stage producing it, so ``evaluate`` reads
``pose_hypotheses`` from ``coarse_pose`` today and from ``refine`` once Beta adds it.

``impl == "gpu"`` stages are never executed here (D1): their output directory is filled by an
adapter on the GPU machine from the ``adapter_request.json`` the runner leaves behind.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from binposert import artefacts
from binposert.data import BopDataset
from binposert.evaluate import (
    VISIBILITY_BINS,
    PosePrediction,
    evaluate_localisation,
    stratify,
    to_markdown,
    write_bop_csv,
)
from binposert.pipeline.selection import Selection
from binposert.pose import PerturbedGroundTruthEstimator, PoseEstimator
from binposert.segment import CachedSegmenter, GroundTruthSegmenter, Segmenter
from binposert.viz import failure_gallery

log = logging.getLogger("binposert.pipeline")

ADAPTER_REQUEST_FILE = "adapter_request.json"


@dataclass
class StageContext:
    dataset: BopDataset
    selection: Selection
    config: dict[str, Any]  # this stage's own config block
    inputs: dict[str, Path]  # artefact kind -> upstream stage directory
    out_dir: Path
    run_config: dict[str, Any]  # the whole resolved run config
    seed: int = 0
    info: dict[str, Any] = field(default_factory=dict)  # free-form, lands in meta.json


@dataclass(frozen=True)
class StageSpec:
    name: str
    config_key: str  # which top-level config block parametrises the stage
    produces: str
    consumes: tuple[str, ...]
    run: Callable[[StageContext], None]

    @staticmethod
    def impl(stage_cfg: dict[str, Any]) -> str:
        return str(stage_cfg.get("impl", "local"))

    @staticmethod
    def version(stage_cfg: dict[str, Any]) -> str:
        return str(stage_cfg.get("version", "1"))


# ----------------------------------------------------------------------------- segment


LOCAL_SEGMENTERS: dict[str, Callable[[BopDataset, dict[str, Any]], Segmenter]] = {
    "gt": lambda ds, cfg: GroundTruthSegmenter(
        ds, min_visible_fraction=float(cfg.get("min_visible_fraction", 0.0))
    ),
}


def run_segment(ctx: StageContext) -> None:
    name = str(ctx.config["name"])
    if name not in LOCAL_SEGMENTERS:
        raise KeyError(f"no local segmenter named {name!r}; GPU segmenters need impl: gpu")
    segmenter = LOCAL_SEGMENTERS[name](ctx.dataset, ctx.config)
    writer = artefacts.DetectionsWriter(ctx.out_dir, source=name)
    n = 0
    for scene_id, image_id in ctx.selection:
        view, _ = ctx.dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
        t0 = time.perf_counter()
        dets = segmenter.segment(view, ctx.selection.object_ids(scene_id, image_id))
        dt = time.perf_counter() - t0
        for d in dets:
            writer.add(scene_id, image_id, d, time_s=dt)
        n += len(dets)
    writer.finish(stage="segment", config=ctx.config, n_images=len(ctx.selection), **ctx.info)
    log.info("segment[%s]: %d detections over %d images", name, n, len(ctx.selection))


# ----------------------------------------------------------------------------- coarse_pose


LOCAL_ESTIMATORS: dict[str, Callable[[BopDataset, dict[str, Any], int], PoseEstimator]] = {
    "synthetic": lambda ds, cfg, seed: PerturbedGroundTruthEstimator(
        ds,
        sigma_t_mm=float(cfg.get("sigma_t_mm", 5.0)),
        sigma_rot_deg=float(cfg.get("sigma_rot_deg", 5.0)),
        n_hypotheses=int(cfg.get("n_hypotheses", 1)),
        seed=int(cfg.get("seed", seed)),
    ),
}


def run_coarse_pose(ctx: StageContext) -> None:
    name = str(ctx.config["name"])
    if name not in LOCAL_ESTIMATORS:
        raise KeyError(f"no local estimator named {name!r}; GPU estimators need impl: gpu")
    estimator = LOCAL_ESTIMATORS[name](ctx.dataset, ctx.config, ctx.seed)
    segmenter = CachedSegmenter(ctx.inputs["detections"])
    writer = artefacts.PoseHypothesesWriter(ctx.out_dir, source=name)
    n = 0
    for scene_id, image_id in ctx.selection:
        view, _ = ctx.dataset.load_view(scene_id, image_id)
        dets = segmenter.segment(view, ctx.selection.object_ids(scene_id, image_id))
        t0 = time.perf_counter()
        hyps = [
            (d, h)
            for d in dets
            for h in estimator.estimate(view, d, ctx.dataset.load_model(d.object_id))
        ]
        dt = time.perf_counter() - t0
        for _, h in hyps:
            writer.add(scene_id, image_id, h, time_s=dt)
        n += len(hyps)
    writer.finish(stage="coarse_pose", config=ctx.config, n_images=len(ctx.selection), **ctx.info)
    log.info("coarse_pose[%s]: %d hypotheses over %d images", name, n, len(ctx.selection))


# ----------------------------------------------------------------------------- evaluate


def select_predictions(
    table: pd.DataFrame,
    score_field: str = "pose_score",
    top_k_per_detection: int = 1,
) -> list[PosePrediction]:
    """PoseHypotheses rows → BOP predictions.

    Score = ``score_field``, falling back to ``seg_score`` and then 1.0 where NaN. Rejected
    refinements keep their (coarse) pose. ``time`` is the per-image wall time (max over rows, -1
    when unknown) as the BOP format requires.
    """
    if len(table) == 0:
        return []
    df = table.copy()
    score = (
        df[score_field].astype(float) if score_field in df else pd.Series(np.nan, index=df.index)
    )
    score = score.fillna(df["seg_score"].astype(float) if "seg_score" in df else np.nan).fillna(1.0)
    df["_score"] = score
    df = df.sort_values(
        ["scene_id", "image_id", "detection_id", "_score", "hypothesis_id"],
        ascending=[True, True, True, False, True],
    )
    df = df.groupby(["scene_id", "image_id", "detection_id"], sort=False).head(top_k_per_detection)
    img_time = df.groupby(["scene_id", "image_id"])["time_s"].max()
    preds: list[PosePrediction] = []
    for r in df.to_dict("records"):
        t = img_time.get((r["scene_id"], r["image_id"]), math.nan)
        preds.append(
            PosePrediction(
                scene_id=int(r["scene_id"]),
                image_id=int(r["image_id"]),
                object_id=int(r["object_id"]),
                score=float(r["_score"]),
                T_camera_object=artefacts.pose_matrix(r),
                time_s=float(t) if t is not None and not math.isnan(float(t)) else -1.0,
            )
        )
    return preds


def run_evaluate(ctx: StageContext) -> None:
    cfg = ctx.config
    table = artefacts.read_pose_hypotheses_table(ctx.inputs["pose_hypotheses"])
    preds = select_predictions(
        table,
        score_field=str(cfg.get("score_field", "pose_score")),
        top_k_per_detection=int(cfg.get("top_k_per_detection", 1)),
    )
    write_bop_csv(ctx.out_dir / "results.csv", preds)
    report = evaluate_localisation(
        preds,
        ctx.dataset,
        n_model_points=int(cfg.get("n_model_points", 2000)),
        with_vsd=bool(cfg.get("with_vsd", True)),
        targets=ctx.selection.as_targets(),
    )
    rows = pd.DataFrame(report.rows)
    rows.to_parquet(ctx.out_dir / "rows.parquet", index=False)
    bins = [float(b) for b in cfg.get("visibility_bins", VISIBILITY_BINS)]
    by_vis = stratify(rows, by="visible_fraction", edges=bins)
    summary = {
        "ar": report.ar,
        "ar_vsd": report.ar_vsd,
        "ar_mssd": report.ar_mssd,
        "ar_mspd": report.ar_mspd,
        "n_gt": report.n_gt,
        "n_predictions": len(preds),
        "n_images": len(ctx.selection),
        "per_object": {str(k): v for k, v in sorted(report.per_object.items())},
        "by_visibility": by_vis.to_dict("records"),
        "source": str(table["source"].iloc[0]) if len(table) else "",
        "stage": str(table["stage"].iloc[0]) if len(table) else "",
    }
    with open(ctx.out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=_json_default)
    md = [
        f"# {summary['source']} ({summary['stage']}) on {ctx.dataset.name}/{ctx.dataset.split}",
        "",
        f"AR = {report.ar:.4f}  (VSD {report.ar_vsd:.4f} · MSSD {report.ar_mssd:.4f} · "
        f"MSPD {report.ar_mspd:.4f}) over {report.n_gt} GT in {len(ctx.selection)} images, "
        f"{len(preds)} predictions",
        "",
        to_markdown(by_vis, "AR by visibility bin (pooled over GT rows)"),
    ]
    n_worst = int(cfg.get("gallery_n_worst", 0) or 0)
    if n_worst > 0 and len(rows):
        worst = failure_gallery(
            rows, preds, ctx.dataset, ctx.out_dir / "gallery_worst_mssd.png", n_worst=n_worst
        )
        worst.to_csv(ctx.out_dir / "gallery_worst_mssd.csv", index=False)
        md += ["", f"Failure gallery: `gallery_worst_mssd.png` (worst {len(worst)} GT by MSSD)"]
    (ctx.out_dir / "report.md").write_text("\n".join(md))
    artefacts.write_meta(
        ctx.out_dir, artefact="evaluation", stage="evaluate", config=cfg, **ctx.info
    )
    log.info("evaluate: AR=%.4f (n_gt=%d)", report.ar, report.n_gt)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


# ----------------------------------------------------------------------------- registry

STAGES: dict[str, StageSpec] = {
    "segment": StageSpec("segment", "segmenter", "detections", (), run_segment),
    "coarse_pose": StageSpec(
        "coarse_pose", "estimator", "pose_hypotheses", ("detections",), run_coarse_pose
    ),
    "evaluate": StageSpec("evaluate", "evaluate", "evaluation", ("pose_hypotheses",), run_evaluate),
}
