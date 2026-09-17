"""Multi-view galleries: mis-associated tracks, symmetry-branch flips and the joint-ICP moves.

Every tile is one View of a track: the member's own pose in red, the fused pose projected into
that View in yellow, the ground-truth pose of the track's majority instance in green (mixed
gallery) or the joint-ICP candidate in cyan (joint gallery). Tiles of one track sit side by side.
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
from binposert.viz.overlay import draw_pose_contour, put_label

RED, YELLOW, GREEN, CYAN = (255, 60, 60), (255, 220, 0), (60, 220, 60), (0, 220, 255)


def make_multiview_galleries(
    dataset: BopDataset,
    associate_dir: Path,
    fuse_dir: Path,
    out_dir: Path,
    mixed: list[dict[str, Any]],
    n: int = 12,
    tile: int = 200,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_parquet(associate_dir / "tracks.parquet")
    fused = pd.read_parquet(fuse_dir / "fused.parquet")
    if len(fused) == 0:
        return {}
    renderers: dict[int, MeshRenderer] = {}

    def renderer(oid: int) -> MeshRenderer:
        if oid not in renderers:
            renderers[oid] = MeshRenderer.from_model(dataset.load_model(oid))
        return renderers[oid]

    def T_world_camera(row: Any) -> np.ndarray:
        return np.asarray([row[f"Twc_{i}{j}"] for i in range(4) for j in range(4)]).reshape(4, 4)

    def track_tiles(frow: Any, extra: str, third: tuple[str, Any] | None) -> list[np.ndarray]:
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
        for _, m in members.iterrows():
            iid = int(m.image_id)
            rgb = dataset.load_rgb(sid, iid)
            K = np.asarray(dataset.camera(sid, iid)["cam_K"], dtype=np.float64).reshape(3, 3)
            T_cw = invert(T_world_camera(m))
            T_member = columns_to_transform(m)
            T_fused_c = T_cw @ T_wo
            shown = [T_member, T_fused_c]
            img = draw_pose_contour(rgb, rend, T_member, K, RED, 2)
            img = draw_pose_contour(img, rend, T_fused_c, K, YELLOW, 1)
            if third is not None:
                kind, arg = third
                if kind == "gt":
                    for g in dataset.ground_truth(sid, iid):
                        if g.gt_index == arg and g.object_id == oid:
                            img = draw_pose_contour(img, rend, g.T_camera_object, K, GREEN, 1)
                            shown.append(g.T_camera_object)
                elif kind == "cand":
                    T_c = T_cw @ arg
                    img = draw_pose_contour(img, rend, T_c, K, CYAN, 1)
                    shown.append(T_c)
            t = crop_around(img, rend, shown, K, 0.6, tile)
            put_label(t, f"s{sid} g{gid} t{tid} im{iid} o{oid} {extra}")
            tiles.append(t)
        return tiles

    written: dict[str, str] = {}

    # 1. mis-associations: tracks whose labelled members refer to two GT instances
    tiles: list[np.ndarray] = []
    for mx in mixed[:n]:
        sel = fused[
            (fused.scene_id == mx["scene_id"])
            & (fused.group_id == mx["group_id"])
            & (fused.track_id == mx["track_id"])
        ]
        if len(sel) == 0:
            continue
        majority = max(mx["instances"], key=lambda k: mx["instances"][k])
        inst = "+".join(str(k) for k in mx["instances"])
        tiles.extend(track_tiles(sel.iloc[0], f"gt {inst}", ("gt", int(majority))))
    if tiles:
        path = out_dir / "mixed.png"
        cv2.imwrite(str(path), tile_grid(tiles, tile, cols=6)[:, :, ::-1])
        written["mixed"] = str(path)

    # 2. symmetry-branch flips: members that were aligned by a non-identity symmetry element
    flips = fused[fused.n_aligned > 0].sort_values("dispersion_deg", ascending=False)
    tiles = []
    for _, frow in flips.head(n).iterrows():
        tiles.extend(
            track_tiles(
                frow, f"aligned {int(frow.n_aligned)} disp {frow.dispersion_deg:.0f}deg", None
            )
        )
    if tiles:
        path = out_dir / "symmetry_flips.png"
        cv2.imwrite(str(path), tile_grid(tiles, tile, cols=6)[:, :, ::-1])
        written["symmetry_flips"] = str(path)

    # 3. joint ICP: the largest accepted moves and the rejections
    if "joint_accepted" in fused and fused.joint_accepted.notna().any():
        for name, sel in (
            (
                "joint_moves",
                fused[fused.joint_accepted.astype(bool)].sort_values(
                    "joint_displacement_mm", ascending=False
                ),
            ),
            ("joint_rejected", fused[~fused.joint_accepted.astype(bool)]),
        ):
            tiles = []
            for _, frow in sel.head(n).iterrows():
                cand = np.asarray(
                    [frow[f"cand_T_{i}{j}"] for i in range(4) for j in range(4)]
                ).reshape(4, 4)
                note = (
                    f"{frow.joint_reason or 'ok'} {frow.joint_displacement_mm:.1f}mm "
                    f"f{frow.joint_fitness:.2f}"
                )
                tiles.extend(track_tiles(frow, note, ("cand", cand)))
            if tiles:
                path = out_dir / f"{name}.png"
                cv2.imwrite(str(path), tile_grid(tiles, tile, cols=6)[:, :, ::-1])
                written[name] = str(path)

    (out_dir / "legend.json").write_text(
        json.dumps(
            {
                "red": "member's own (refined single-view) pose",
                "yellow": "fused pose projected into the View",
                "green": "GT of the majority instance (mixed gallery)",
                "cyan": "joint-ICP candidate (joint galleries)",
                "files": written,
            },
            indent=2,
        )
    )
    return written
