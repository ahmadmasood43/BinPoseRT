"""Failure gallery: the worst-N ground-truth objects by MSSD with GT vs. predicted contours."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.evaluate.metrics import mssd
from binposert.pipeline.artefacts import columns_to_transform, read_hypotheses_table
from binposert.render import MeshRenderer
from binposert.viz.overlay import draw_pose_contour, put_label

GT_COLOR = (0, 220, 0)
PRED_COLOR = (255, 60, 60)


def make_failure_gallery(
    evaluate_dir: str | Path,
    dataset: BopDataset,
    n: int = 20,
    out_dir: str | Path | None = None,
    crop_pad: float = 0.6,
    tile: int = 256,
) -> Path:
    """Reads ``gt_rows.parquet`` + ``run_manifest.json`` of an evaluate stage, renders the worst
    ``n`` GT objects (largest MSSD among those that received a prediction) and writes
    ``gallery.png`` + ``gallery.json`` to ``out_dir`` (default: ``<evaluate_dir>/gallery``)."""
    ev = Path(evaluate_dir)
    out = Path(out_dir) if out_dir is not None else ev / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ev / "run_manifest.json").read_text())
    pose_stage = manifest["stages"].get("refine") or manifest["stages"]["coarse_pose"]
    hyps = read_hypotheses_table(pose_stage["dir"])
    rows = pd.read_parquet(ev / "gt_rows.parquet")
    rows = rows[np.isfinite(rows["mssd_mm"])].sort_values("mssd_mm", ascending=False).head(n)

    renderers: dict[int, MeshRenderer] = {}
    pts_cache: dict[int, Any] = {}
    tiles: list[np.ndarray] = []
    index: list[dict[str, Any]] = []
    for _, r in rows.iterrows():
        s, i, o = int(r.scene_id), int(r.image_id), int(r.object_id)
        view, gts = dataset.load_view(s, i, load_rgb=True, load_depth=False)
        gt = next(g for g in gts if g.gt_index == int(r.gt_index))
        model = dataset.load_model(o)
        if o not in renderers:
            renderers[o] = MeshRenderer.from_model(model)
            pts_cache[o] = model.sample_points(500, seed=o)
        cand = hyps[(hyps.scene_id == s) & (hyps.image_id == i) & (hyps.object_id == o)]
        if len(cand) == 0:
            continue
        errs = [
            mssd(columns_to_transform(c), gt.T_camera_object, pts_cache[o], model.symmetry)
            for _, c in cand.iterrows()
        ]
        best = cand.iloc[int(np.argmin(errs))]
        T_pred = columns_to_transform(best)
        assert view.rgb is not None
        img = draw_pose_contour(view.rgb, renderers[o], gt.T_camera_object, view.K, GT_COLOR)
        img = draw_pose_contour(img, renderers[o], T_pred, view.K, PRED_COLOR)
        crop = crop_around(img, renderers[o], [gt.T_camera_object, T_pred], view.K, crop_pad, tile)
        label = f"s{s} i{i} obj{o} vis{r.visible_fraction:.2f} MSSD {r.mssd_mm:.0f}mm"
        put_label(crop, label)
        tiles.append(crop)
        index.append(
            {
                "scene_id": s,
                "image_id": i,
                "object_id": o,
                "gt_index": int(r.gt_index),
                "visible_fraction": float(r.visible_fraction),
                "mssd_mm": float(r.mssd_mm),
                "mssd_over_diameter": float(r.mssd_mm / model.diameter),
                "detection_id": int(best["detection_id"]),
                "pose_score": float(best["pose_score"]),
                "seg_score": float(best["seg_score"]),
            }
        )
    if tiles:
        cv2.imwrite(str(out / "gallery.png"), tile_grid(tiles, tile)[:, :, ::-1])
    with open(out / "gallery.json", "w") as f:
        json.dump(
            {"legend": {"green": "ground truth", "red": "closest prediction"}, "items": index},
            f,
            indent=2,
        )
    return out


def tile_grid(tiles: list[np.ndarray], tile: int, cols: int = 5) -> np.ndarray:
    rows_n = int(np.ceil(len(tiles) / cols))
    grid = np.zeros((rows_n * tile, cols * tile, 3), dtype=np.uint8)
    for k, t in enumerate(tiles):
        y, x = divmod(k, cols)
        grid[y * tile : (y + 1) * tile, x * tile : (x + 1) * tile] = t
    return grid


def crop_around(
    img: np.ndarray,
    renderer: MeshRenderer,
    transforms: list[np.ndarray],
    K: np.ndarray,
    pad: float,
    tile: int,
) -> np.ndarray:
    """Square crop around the union of the model's silhouettes at ``transforms``, resized to
    ``tile`` × ``tile``."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    for T in transforms:
        m |= renderer.render(T, K, (h, w)).mask
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        cx, cy, side = w // 2, h // 2, min(h, w)
    else:
        cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + pad)) + 8
    side = max(32, min(side, max(h, w)))
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
