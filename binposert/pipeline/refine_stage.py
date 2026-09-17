"""The ``refine`` stage: every coarse PoseHypothesis of every View -> refined-or-rejected hypothesis
(``hypotheses.parquet``) plus a per-hypothesis audit table (``refine_details.parquet``) holding the
coarse and candidate poses and the gate measurements, so the analysis can score rejections too.
Scenes are processed in parallel (spawned processes; CPU only)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from binposert.data import BopDataset
from binposert.pipeline.artefacts import (
    HYPOTHESIS_COLUMNS,
    HypothesisRecord,
    hypothesis_from_row,
    hypothesis_to_row,
    load_mask,
    read_detections_table,
    read_hypotheses_table,
    transform_to_columns,
)
from binposert.pipeline.pool import map_scenes
from binposert.refine import GateParams, Refiner, RefinerParams

DETAILS_FILE = "refine_details.parquet"
DETAIL_COLUMNS = [
    "scene_id",
    "image_id",
    "object_id",
    "detection_id",
    "hypothesis_id",
    "accepted",
    "reason",
    "iou_coarse",
    "iou_refined",
    "boundary_px",
    "displacement_mm",
    "displacement_deg",
    "fitness",
    "rmse_mm",
    "n_correspondences",
    "n_scene_points",
    "depth_coverage",
    "z_shift_mm",
    "z_init_overlap",
    "seconds",
]


def params_from_config(section: dict[str, Any]) -> RefinerParams:
    p = dict(section.get("params", {}))
    gate = GateParams(**p.pop("gate", {}))
    if "corr_dist_factors" in p:
        p["corr_dist_factors"] = tuple(float(x) for x in p["corr_dist_factors"])
    return RefinerParams(gate=gate, **p)


def refine_scene(
    scene_id: int,
    dataset: BopDataset,
    seg_dir: str,
    pose_dir: str,
    params: RefinerParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dets = read_detections_table(seg_dir)
    hyps = read_hypotheses_table(pose_dir)
    dets = dets[dets.scene_id == scene_id]
    hyps = hyps[hyps.scene_id == scene_id]
    refiners: dict[int, Refiner] = {}
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for image_id in dataset.image_ids(scene_id):
        h_img = hyps[hyps.image_id == image_id]
        if len(h_img) == 0:
            continue
        view, _ = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=True)
        d_img = dets[dets.image_id == image_id].set_index("detection_id")
        for _, h_row in h_img.iterrows():
            hyp = hypothesis_from_row(h_row)
            oid = hyp.object_id
            if oid not in refiners:
                refiners[oid] = Refiner(dataset.load_model(oid), params)
            mask = load_mask(seg_dir, str(d_img.loc[hyp.detection_id, "mask_path"]))
            out = refiners[oid].refine(view, mask, hyp)
            prior = float(h_row["time_s"]) if pd.notna(h_row["time_s"]) else 0.0
            rows.append(
                hypothesis_to_row(
                    HypothesisRecord(scene_id, image_id, out.hypothesis, prior + out.seconds)
                )
            )
            g, r = out.gate, out.registration
            det: dict[str, Any] = {
                "scene_id": scene_id,
                "image_id": image_id,
                "object_id": oid,
                "detection_id": hyp.detection_id,
                "hypothesis_id": hyp.hypothesis_id,
                "accepted": out.hypothesis.rejection_reason is None,
                "reason": out.hypothesis.rejection_reason,
                "iou_coarse": g.iou_coarse if g else float("nan"),
                "iou_refined": g.iou_refined if g else float("nan"),
                "boundary_px": g.boundary_px if g else float("nan"),
                "displacement_mm": g.displacement_mm if g else float("nan"),
                "displacement_deg": g.displacement_deg if g else float("nan"),
                "fitness": r.fitness if r else float("nan"),
                "rmse_mm": r.inlier_rmse_mm if r else float("nan"),
                "n_correspondences": r.n_correspondences if r else 0,
                "n_scene_points": out.n_scene_points,
                "depth_coverage": out.depth_coverage,
                "z_shift_mm": out.z_shift_mm,
                "z_init_overlap": out.z_init_overlap,
                "seconds": out.seconds,
            }
            det.update(
                {f"coarse_{k}": v for k, v in transform_to_columns(hyp.T_camera_object).items()}
            )
            det.update({f"cand_{k}": v for k, v in transform_to_columns(out.T_candidate).items()})
            details.append(det)
    return rows, details


def run_refine(
    dataset: BopDataset,
    seg_dir: Path,
    pose_dir: Path,
    out_dir: Path,
    params: RefinerParams,
    n_workers: int = 1,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    scene_ids = dataset.scene_ids
    jobs = [(sid, dataset, str(seg_dir), str(pose_dir), params) for sid in scene_ids]
    if n_workers > 1 and len(scene_ids) > 1:
        results = map_scenes(refine_scene, jobs, n_workers)
    else:
        results = [refine_scene(*job) for job in jobs]
    rows = [r for rows_, _ in results for r in rows_]
    details = [d for _, details_ in results for d in details_]
    table = pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS)
    table["rejection_reason"] = table["rejection_reason"].astype(object)
    table.to_parquet(out_dir / "hypotheses.parquet", index=False)
    det_cols = DETAIL_COLUMNS + [
        c for c in (details[0] if details else {}) if c not in DETAIL_COLUMNS
    ]
    det_table = pd.DataFrame(details, columns=det_cols)
    det_table["reason"] = det_table["reason"].astype(object)
    det_table.to_parquet(out_dir / DETAILS_FILE, index=False)
    n = len(det_table)
    accepted = int(det_table["accepted"].sum()) if n else 0
    reasons = det_table["reason"].value_counts(dropna=True).to_dict() if n else {}
    return {
        "n_hypotheses": n,
        "n_accepted": accepted,
        "rejection_rate": (n - accepted) / n if n else float("nan"),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
        "median_seconds": float(det_table["seconds"].median()) if n else float("nan"),
        "wall_seconds": time.perf_counter() - t0,
    }
