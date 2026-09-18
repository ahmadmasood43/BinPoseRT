"""Confidence galleries (Delta): the poses the ConfidenceModel gets most wrong.

*Confident failures* — the failed FusedPoses with the highest Confidence — are the figure that
matters most for a reliability claim: every one of them is a wrong pose the Verdict would accept.
*Unconfident successes* are the mirror (correct poses that would be rejected or deferred). Every
tile is one member View of a track: the member's own pose in red, the fused pose projected into
that View in yellow, the nearest ground-truth pose in green; the label carries the Confidence, the
error in diameters and the view count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.pipeline.artefacts import columns_to_transform
from binposert.render import MeshRenderer
from binposert.transforms import invert
from binposert.viz.gallery import crop_around, tile_grid
from binposert.viz.multiview_gallery import GREEN, RED, YELLOW
from binposert.viz.overlay import draw_pose_contour, put_label


def make_confidence_galleries(
    dataset: BopDataset,
    scored: pd.DataFrame,
    tracks: pd.DataFrame,
    out_dir: Path,
    n: int = 12,
    tile: int = 200,
    max_views: int = 3,
) -> dict[str, str]:
    """``scored``: fused rows with ``confidence``, ``success``, ``mssd_mm``, ``diameter``,
    ``gt_index``; ``tracks``: the associate stage's ``tracks.parquet`` of the same row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    renderers: dict[int, MeshRenderer] = {}

    def renderer(oid: int) -> MeshRenderer:
        if oid not in renderers:
            renderers[oid] = MeshRenderer.from_model(dataset.load_model(oid))
        return renderers[oid]

    def track_tiles(frow: Any) -> list[np.ndarray]:
        sid, gid, tid, oid = (
            int(frow.scene_id),
            int(frow.group_id),
            int(frow.track_id),
            int(frow.object_id),
        )
        members = tracks[
            (tracks.scene_id == sid) & (tracks.group_id == gid) & (tracks.track_id == tid)
        ]
        T_wo = columns_to_transform(frow)
        rend = renderer(oid)
        tiles = []
        note = f"c{frow.confidence:.2f} e{frow.mssd_mm / frow.diameter:.2f}d k{int(frow.n_views)}"
        for _, m in members.head(max_views).iterrows():
            iid = int(m.image_id)
            rgb = dataset.load_rgb(sid, iid)
            K = np.asarray(dataset.camera(sid, iid)["cam_K"], dtype=np.float64).reshape(3, 3)
            T_wc = np.asarray([m[f"Twc_{i}{j}"] for i in range(4) for j in range(4)]).reshape(4, 4)
            T_member = columns_to_transform(m)
            T_fused_c = invert(T_wc) @ T_wo
            shown = [T_member, T_fused_c]
            img = draw_pose_contour(rgb, rend, T_member, K, RED, 2)
            img = draw_pose_contour(img, rend, T_fused_c, K, YELLOW, 1)
            for g in dataset.ground_truth(sid, iid):
                if g.gt_index == int(frow.gt_index) and g.object_id == oid:
                    img = draw_pose_contour(img, rend, g.T_camera_object, K, GREEN, 1)
                    shown.append(g.T_camera_object)
            t = crop_around(img, rend, shown, K, 0.6, tile)
            put_label(t, f"s{sid} im{iid} o{oid} {note}")
            tiles.append(t)
        return tiles

    written: dict[str, str] = {}
    selections = {
        "confident_failures": scored[~scored.success.astype(bool)].sort_values(
            "confidence", ascending=False
        ),
        "unconfident_successes": scored[scored.success.astype(bool)].sort_values(
            "confidence", ascending=True
        ),
    }
    for name, sel in selections.items():
        tiles: list[np.ndarray] = []
        listed = []
        for _, frow in sel.head(n).iterrows():
            tiles.extend(track_tiles(frow))
            listed.append(
                {
                    "scene_id": int(frow.scene_id),
                    "group_id": int(frow.group_id),
                    "track_id": int(frow.track_id),
                    "object_id": int(frow.object_id),
                    "confidence": float(frow.confidence),
                    "mssd_over_d": float(frow.mssd_mm / frow.diameter),
                    "n_views": int(frow.n_views),
                    "t_err_mm": float(frow.get("t_err_mm", np.nan)),
                    "r_err_deg": float(frow.get("r_err_deg", np.nan)),
                }
            )
        if tiles:
            path = out_dir / f"{name}.png"
            cv2.imwrite(str(path), tile_grid(tiles, tile, cols=6)[:, :, ::-1])
            written[name] = str(path)
            (out_dir / f"{name}.json").write_text(json.dumps(listed, indent=2))
    (out_dir / "legend.json").write_text(
        json.dumps(
            {
                "red": "member's own (refined single-view) pose",
                "yellow": "fused pose projected into the View",
                "green": "nearest ground-truth pose",
                "label": "c = Confidence, e = MSSD in diameters, k = view count",
                "files": written,
            },
            indent=2,
        )
    )
    return written


__all__ = ["make_confidence_galleries"]
