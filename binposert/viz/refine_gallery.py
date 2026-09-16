"""Refinement failure galleries (Beta): GT (green) vs. coarse (yellow) vs. ICP candidate (red)
contours for the hypotheses where the gate got it wrong — accepted a candidate that is farther
from GT than the coarse pose, or rejected one that is closer and would have succeeded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.evaluate.refinement import GATE_REASONS
from binposert.pipeline.artefacts import TRANSFORM_COLUMNS
from binposert.render import MeshRenderer
from binposert.viz.gallery import crop_around, tile_grid
from binposert.viz.overlay import draw_pose_contour, put_label

GT_COLOR = (0, 220, 0)
COARSE_COLOR = (255, 210, 0)
CAND_COLOR = (255, 60, 60)
LEGEND = {"green": "ground truth", "yellow": "coarse pose", "red": "ICP candidate"}


def select_failures(scored: pd.DataFrame, n: int) -> dict[str, pd.DataFrame]:
    """The two failure kinds of a gate, worst first, among hypotheses matched to a valid GT.

    ``accepted_worse``: accepted, candidate MSSD above the coarse MSSD (largest increase first).
    ``rejected_good``: rejected by the gate, candidate closer to GT *and* within 0.1 d (largest
    decrease first)."""
    s = scored[scored["gt_valid"].fillna(False).astype(bool)].copy()
    s["delta_mssd_mm"] = s["cand_mssd_mm"] - s["coarse_mssd_mm"]
    worse = s[s["accepted"] & (s["delta_mssd_mm"] > 0)]
    good = s[~s["accepted"] & s["reason"].isin(GATE_REASONS) & s["improved"] & s["cand_ok"]]
    return {
        "accepted_worse": worse.sort_values("delta_mssd_mm", ascending=False).head(n),
        "rejected_good": good.sort_values("delta_mssd_mm", ascending=True).head(n),
    }


def make_refine_galleries(
    scored: pd.DataFrame,
    dataset: BopDataset,
    out_dir: str | Path,
    n: int = 20,
    crop_pad: float = 0.6,
    tile: int = 256,
) -> dict[str, Path]:
    """One ``<kind>.png`` + ``<kind>.json`` per failure kind of :func:`select_failures`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderers: dict[int, MeshRenderer] = {}
    written: dict[str, Path] = {}
    for kind, rows in select_failures(scored, n).items():
        tiles: list[np.ndarray] = []
        index: list[dict[str, Any]] = []
        for _, r in rows.iterrows():
            s, i, o = int(r.scene_id), int(r.image_id), int(r.object_id)
            view, gts = dataset.load_view(s, i, load_rgb=True, load_depth=False)
            gt = next(g for g in gts if g.gt_index == int(r.gt_index))
            if o not in renderers:
                renderers[o] = MeshRenderer.from_model(dataset.load_model(o))
            T_c = _transform(r, "coarse_")
            T_r = _transform(r, "cand_")
            assert view.rgb is not None
            img = draw_pose_contour(view.rgb, renderers[o], gt.T_camera_object, view.K, GT_COLOR)
            img = draw_pose_contour(img, renderers[o], T_c, view.K, COARSE_COLOR)
            img = draw_pose_contour(img, renderers[o], T_r, view.K, CAND_COLOR)
            crop = crop_around(
                img, renderers[o], [gt.T_camera_object, T_c, T_r], view.K, crop_pad, tile
            )
            reason = "accepted" if bool(r.accepted) else str(r.reason)
            label = f"s{s} i{i} obj{o} {reason} {r.coarse_mssd_mm:.0f}->{r.cand_mssd_mm:.0f}mm"
            put_label(crop, label)
            tiles.append(crop)
            index.append(
                {
                    "scene_id": s,
                    "image_id": i,
                    "object_id": o,
                    "gt_index": int(r.gt_index),
                    "detection_id": int(r.detection_id),
                    "visible_fraction": float(r.visible_fraction),
                    "accepted": bool(r.accepted),
                    "reason": None if bool(r.accepted) else str(r.reason),
                    "coarse_mssd_mm": float(r.coarse_mssd_mm),
                    "cand_mssd_mm": float(r.cand_mssd_mm),
                    "diameter": float(r.diameter),
                    "iou_coarse": float(r.iou_coarse),
                    "iou_refined": float(r.iou_refined),
                    "displacement_mm": float(r.displacement_mm),
                    "displacement_deg": float(r.displacement_deg),
                    "fitness": float(r.fitness),
                }
            )
        if tiles:
            cv2.imwrite(str(out / f"{kind}.png"), tile_grid(tiles, tile)[:, :, ::-1])
        with open(out / f"{kind}.json", "w") as f:
            json.dump({"legend": LEGEND, "kind": kind, "items": index}, f, indent=2)
        written[kind] = out / f"{kind}.png"
    return written


def _transform(row: Any, prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}{c}"]) for c in TRANSFORM_COLUMNS]).reshape(4, 4)
