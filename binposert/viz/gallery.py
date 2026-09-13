"""Failure galleries: the worst GT rows of an evaluation as one tiled PNG (D2 — honest failures)."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.evaluate import PosePrediction
from binposert.render import MeshRenderer
from binposert.viz.overlay import crop_around, overlay_pose


def failure_gallery(
    rows: pd.DataFrame,
    preds: list[PosePrediction],
    dataset: BopDataset,
    out_path: str | Path,
    n_worst: int = 20,
    sort_by: str = "mssd_mm",
    columns: int = 5,
    tile: int = 256,
) -> pd.DataFrame:
    """Tile the ``n_worst`` GT rows by ``sort_by`` (descending). GT contour green, estimate red.

    Returns the selected rows (with the tile index) so the caller can write a caption table.
    """
    if len(rows) == 0:
        return rows
    worst = rows.sort_values(sort_by, ascending=False).head(n_worst).reset_index(drop=True)
    by_key: dict[tuple[int, int, int], list[PosePrediction]] = {}
    for p in preds:
        by_key.setdefault((p.scene_id, p.image_id, p.object_id), []).append(p)
    renderers: dict[int, MeshRenderer] = {}
    tiles = []
    for r in worst.to_dict("records"):
        scene_id, image_id, object_id = int(r["scene_id"]), int(r["image_id"]), int(r["object_id"])
        view, gts = dataset.load_view(scene_id, image_id, load_depth=False)
        gt = next(g for g in gts if g.gt_index == int(r["gt_index"]))
        if object_id not in renderers:
            renderers[object_id] = MeshRenderer.from_model(dataset.load_model(object_id))
        ren = renderers[object_id]
        cand = by_key.get((scene_id, image_id, object_id), [])
        # the candidate the row's error came from: the one closest to this GT (same rule as rows)
        T_est = min(
            (p.T_camera_object for p in cand),
            key=lambda T: float(np.linalg.norm(T[:3, 3] - gt.T_camera_object[:3, 3])),
            default=None,
        )
        rgb = view.rgb if view.rgb is not None else np.zeros((*view.image_size, 3), np.uint8)
        err = r[sort_by]
        vis = r["visible_fraction"]
        label = f"s{scene_id} i{image_id} o{object_id} {sort_by}={err:.1f} vis={vis:.2f}"
        full = overlay_pose(rgb, ren, view.K, T_est, gt.T_camera_object)
        gt_mask = ren.render(gt.T_camera_object, view.K, view.image_size).mask
        est_mask = ren.render(T_est, view.K, view.image_size).mask if T_est is not None else gt_mask
        crop = crop_around(full, gt_mask | est_mask, size=tile)
        cv2.putText(crop, label, (4, tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 2)
        cv2.putText(crop, label, (4, tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
        tiles.append(crop)
    while len(tiles) % columns:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, grid)
    worst.insert(0, "tile", range(len(worst)))
    return worst
