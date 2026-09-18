"""Labelled tables for fitting and evaluating ConfidenceModels (D11).

The project-wide success criterion is ``MSSD(T_pred, T_gt) < 0.1 · diameter`` against the nearest
annotated pose of the same object — nearest by MSSD, so a symmetric object matched to a rotated
copy of itself counts as correct exactly as BOP counts it. Every ground-truth pose of the object
is a candidate, not only the BOP-valid ones (``gt_valid`` records that): a pose that is right on a
mostly-occluded copy is still a right pose, and the model must learn from it. A hypothesis whose
object has no annotation in its View (a segmentation false positive) is a failure with an infinite
error.

PoseHypotheses are labelled in their camera frame. A FusedPose is a world-frame pose; MSSD is
invariant under a rigid motion of both poses, so it is scored against the ground truth of every
View of its group lifted through ``T_world_camera`` and the nearest wins — the same as scoring its
projection in the View where it fits best.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from binposert.data import BopDataset
from binposert.evaluate.localisation import valid_ground_truth
from binposert.evaluate.metrics import mssd
from binposert.pipeline.artefacts import columns_to_transform
from binposert.symmetry import sym_aware_rotation_distance_deg
from binposert.transforms import Mat4, translation_distance
from binposert.types import GroundTruthPose

SUCCESS_FACTOR = 0.1  # x diameter
LABEL_COLUMNS = [
    "diameter",
    "mssd_mm",
    "success",
    "gt_present",
    "gt_index",
    "gt_image_id",
    "gt_valid",
    "gt_visible_fraction",
    "t_err_mm",
    "r_err_deg",
]
HYPOTHESIS_KEYS = ["scene_id", "image_id", "object_id", "detection_id", "hypothesis_id"]
FUSED_KEYS = ["scene_id", "group_id", "track_id", "object_id"]
# ground truth farther than this (translation, x diameter) is not worth an MSSD evaluation unless
# it is the nearest there is
PREFILTER_FACTOR = 2.0


class _Labeller:
    def __init__(self, dataset: BopDataset, n_model_points: int) -> None:
        self.ds = dataset
        self.n_model_points = n_model_points
        self._pts: dict[int, npt.NDArray[np.float64]] = {}

    def points(self, object_id: int) -> npt.NDArray[np.float64]:
        if object_id not in self._pts:
            m = self.ds.load_model(object_id)
            self._pts[object_id] = (
                m.sample_points(self.n_model_points, seed=object_id)
                if self.n_model_points > 0
                else m.vertices
            )
        return self._pts[object_id]

    def nearest(
        self,
        T_pred: Mat4,
        object_id: int,
        candidates: list[tuple[Mat4, GroundTruthPose, int, bool]],
    ) -> dict[str, Any]:
        """Label columns for one pose against ``(T_gt, gt, image_id, valid)`` candidates expressed
        in the frame of ``T_pred``."""
        model = self.ds.load_model(object_id)
        d = float(model.diameter)
        if not candidates:
            return {
                "diameter": d,
                "mssd_mm": math.inf,
                "success": False,
                "gt_present": False,
                "gt_index": -1,
                "gt_image_id": -1,
                "gt_valid": False,
                "gt_visible_fraction": math.nan,
                "t_err_mm": math.inf,
                "r_err_deg": math.nan,
            }
        pts = self.points(object_id)
        t_dist = np.asarray([translation_distance(T_pred, c[0]) for c in candidates])
        near = [int(i) for i in np.nonzero(t_dist <= PREFILTER_FACTOR * d)[0]]
        if not near:
            near = [int(np.argmin(t_dist))]
        errors = {i: mssd(T_pred, candidates[i][0], pts, model.symmetry) for i in near}
        best = min(errors, key=lambda i: errors[i])
        T_gt, gt, image_id, valid = candidates[best]
        e = errors[best]
        return {
            "diameter": d,
            "mssd_mm": e,
            "success": bool(e < SUCCESS_FACTOR * d),
            "gt_present": True,
            "gt_index": int(gt.gt_index),
            "gt_image_id": int(image_id),
            "gt_valid": bool(valid),
            "gt_visible_fraction": float(gt.visible_fraction),
            "t_err_mm": float(t_dist[best]),
            "r_err_deg": sym_aware_rotation_distance_deg(T_pred, T_gt, model.symmetry),
        }

    def view_candidates(
        self, scene_id: int, image_id: int, object_id: int, T_world_camera: Mat4 | None
    ) -> list[tuple[Mat4, GroundTruthPose, int, bool]]:
        gts = [g for g in self.ds.ground_truth(scene_id, image_id) if g.object_id == object_id]
        valid = {
            g.gt_index for g in valid_ground_truth(self.ds, scene_id, image_id, object_id, gts)
        }
        out = []
        for g in gts:
            T = g.T_camera_object if T_world_camera is None else T_world_camera @ g.T_camera_object
            out.append((T, g, image_id, g.gt_index in valid))
        return out


# ----------------------------------------------------------------------------- hypotheses


def _label_scene_hypotheses(
    scene_id: int, table: pd.DataFrame, dataset: BopDataset, n_model_points: int
) -> list[dict[str, Any]]:
    lab = _Labeller(dataset, n_model_points)
    rows: list[dict[str, Any]] = []
    cand_cache: dict[tuple[int, int], list[tuple[Mat4, GroundTruthPose, int, bool]]] = {}
    for _, row in table.iterrows():
        image_id, object_id = int(row["image_id"]), int(row["object_id"])
        key = (image_id, object_id)
        if key not in cand_cache:
            cand_cache[key] = lab.view_candidates(scene_id, image_id, object_id, None)
        rec = {k: row[k] for k in HYPOTHESIS_KEYS}
        rec.update(lab.nearest(columns_to_transform(row), object_id, cand_cache[key]))
        rows.append(rec)
    return rows


def label_hypotheses(
    table: pd.DataFrame, dataset: BopDataset, n_model_points: int = 0, n_workers: int = 1
) -> pd.DataFrame:
    """The hypotheses table (any stage's ``hypotheses.parquet`` rows) with :data:`LABEL_COLUMNS`
    and the Model H extra ``rejected`` appended."""
    labels = _by_scene(_label_scene_hypotheses, table, dataset, n_model_points, n_workers)
    lab = pd.DataFrame(labels, columns=HYPOTHESIS_KEYS + LABEL_COLUMNS)
    merged = table.merge(lab, on=HYPOTHESIS_KEYS, how="left", validate="one_to_one")
    from binposert.confidence.schema import hypothesis_extras

    return hypothesis_extras(merged)


# ----------------------------------------------------------------------------- fused poses


def _label_scene_fused(
    scene_id: int, table: pd.DataFrame, dataset: BopDataset, n_model_points: int
) -> list[dict[str, Any]]:
    lab = _Labeller(dataset, n_model_points)
    rows: list[dict[str, Any]] = []
    extrinsics: dict[int, Mat4] = {}
    cand_cache: dict[tuple[int, int], list[tuple[Mat4, GroundTruthPose, int, bool]]] = {}
    for _, row in table.iterrows():
        object_id = int(row["object_id"])
        image_ids = [int(i) for i in str(row["image_ids"]).split(",") if i != ""]
        candidates: list[tuple[Mat4, GroundTruthPose, int, bool]] = []
        for image_id in image_ids:
            if image_id not in extrinsics:
                view, _ = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
                extrinsics[image_id] = view.T_world_camera
            key = (image_id, object_id)
            if key not in cand_cache:
                cand_cache[key] = lab.view_candidates(
                    scene_id, image_id, object_id, extrinsics[image_id]
                )
            candidates.extend(cand_cache[key])
        rec = {k: row[k] for k in FUSED_KEYS}
        rec.update(lab.nearest(columns_to_transform(row), object_id, candidates))
        rows.append(rec)
    return rows


def label_fused(
    fused: pd.DataFrame, dataset: BopDataset, n_model_points: int = 0, n_workers: int = 1
) -> pd.DataFrame:
    """``fused.parquet`` rows with :data:`LABEL_COLUMNS` appended (world-frame MSSD against the
    ground truth of the group's Views). The fused rows must carry the dataset's extrinsics: the
    calibration-sweep rows (perturbed ``T_world_camera``) are not labelled this way."""
    labels = _by_scene(_label_scene_fused, fused, dataset, n_model_points, n_workers)
    lab = pd.DataFrame(labels, columns=FUSED_KEYS + LABEL_COLUMNS)
    return fused.merge(lab, on=FUSED_KEYS, how="left", validate="one_to_one")


def _by_scene(
    fn: Any, table: pd.DataFrame, dataset: BopDataset, n_model_points: int, n_workers: int
) -> list[dict[str, Any]]:
    scene_ids = sorted(int(s) for s in table["scene_id"].unique())
    jobs = [
        (sid, table[table.scene_id == sid].reset_index(drop=True), dataset, n_model_points)
        for sid in scene_ids
    ]
    if n_workers > 1 and len(jobs) > 1:
        from binposert.pipeline.pool import map_scenes

        results = map_scenes(fn, jobs, n_workers)
    else:
        results = [fn(*job) for job in jobs]
    return [r for rows in cast(list[list[dict[str, Any]]], results) for r in rows]


__all__ = [
    "FUSED_KEYS",
    "HYPOTHESIS_KEYS",
    "LABEL_COLUMNS",
    "SUCCESS_FACTOR",
    "label_fused",
    "label_hypotheses",
]
