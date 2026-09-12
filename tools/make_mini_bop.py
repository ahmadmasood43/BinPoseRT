#!/usr/bin/env python
"""Build the committed mini BOP fixture (tests/fixtures/mini_bop) from two T-LESS models.

2 scenes x 2 calibrated Views x 3 objects (two copies of obj 1, one obj 5), rendered on CPU with the
same renderer the pipeline uses. Deterministic: re-running produces identical files.

    uv run python tools/download_bop.py tless --parts models
    uv run python tools/make_mini_bop.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from binposert import transforms as tf
from binposert.data.writer import BopSceneWriter, write_models
from binposert.render import MeshRenderer, render_scene
from binposert.symmetry import from_bop_model_info
from binposert.types import ObjectModel

OBJECT_IDS = (1, 5)
TARGET_FACES = 3000
IMAGE_SIZE = (240, 320)
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])
SPLIT = "test"


def load_decimated(models_dir: Path, object_id: int, info: dict) -> ObjectModel:
    m = trimesh.load(models_dir / f"obj_{object_id:06d}.ply", force="mesh", process=False)
    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, dtype=np.int32)),
    )
    if len(m.faces) > TARGET_FACES:
        o3 = o3.simplify_quadric_decimation(TARGET_FACES)
    o3.remove_unreferenced_vertices()
    return ObjectModel(
        object_id=object_id,
        vertices=np.asarray(o3.vertices, dtype=np.float64),
        faces=np.asarray(o3.triangles, dtype=np.int64),
        diameter=float(info["diameter"]),
        symmetry=from_bop_model_info(info),
    )


def look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return tf.make_T(np.stack([x, y, z], axis=1), eye)


def shade(depth: np.ndarray, geometry_ids: np.ndarray) -> np.ndarray:
    """Cheap fake RGB: per-object tint modulated by depth gradient. Deliberately not realistic."""
    h, w = depth.shape
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)
    tints = np.array([[200, 180, 160], [160, 200, 180], [180, 160, 200]], dtype=np.float64)
    gy, gx = np.gradient(np.where(depth > 0, depth, np.nan))
    slope = np.nan_to_num(np.hypot(gx, gy), nan=0.0)
    shade_f = np.clip(1.0 - slope / 8.0, 0.3, 1.0)
    for gid in range(3):
        sel = geometry_ids == gid
        rgb[sel] = (tints[gid][None, :] * shade_f[sel][:, None]).astype(np.uint8)
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tless-models", type=Path, default=Path("data/bop/tless/models_cad"))
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures/mini_bop"))
    args = ap.parse_args()

    with open(args.tless_models / "models_info.json") as f:
        full_info = json.load(f)
    models = [load_decimated(args.tless_models, oid, full_info[str(oid)]) for oid in OBJECT_IDS]
    info = {str(oid): full_info[str(oid)] for oid in OBJECT_IDS}

    if args.out.exists():
        shutil.rmtree(args.out)
    write_models(args.out, models, info)

    rng = np.random.default_rng(1234)
    by_id = {m.object_id: m for m in models}
    renderers = {m.object_id: MeshRenderer.from_model(m) for m in models}

    for scene_id in (1, 2):
        # three objects resting near the origin on the world plane z = 0
        placed: list[tuple[int, np.ndarray]] = []
        for oid, (x, y) in zip((1, 1, 5), ((-60, 45), (65, -40), (0, 0)), strict=True):
            yaw = rng.uniform(0, 360)
            tilt = rng.uniform(-25, 25)
            R = tf.rotvec_T([0, 0, 1], yaw)[:3, :3] @ tf.rotvec_T([1, 0, 0], tilt)[:3, :3]
            z = 0.5 * (by_id[oid].vertices[:, 2].max() - by_id[oid].vertices[:, 2].min())
            T_world_object = tf.make_T(R, [x + rng.normal(0, 4), y + rng.normal(0, 4), z])
            placed.append((oid, T_world_object))

        writer = BopSceneWriter(args.out, SPLIT, scene_id)
        eyes = [(230.0, -170.0, 150.0), (-190.0, 230.0, 200.0)]
        if scene_id == 2:
            eyes = [(40.0, -300.0, 130.0), (260.0, 110.0, 210.0)]
        for image_id, eye in enumerate(eyes):
            T_world_camera = look_at(eye)
            T_camera_world = tf.invert(T_world_camera)
            objs = [(oid, T_camera_world @ T_wo) for oid, T_wo in placed]
            scene_res = render_scene([(by_id[oid], T_co) for oid, T_co in objs], K, IMAGE_SIZE)
            masks_full = [renderers[oid].render(T_co, K, IMAGE_SIZE).mask for oid, T_co in objs]
            masks_vis = [scene_res.geometry_ids == i for i in range(len(objs))]
            rgb = shade(scene_res.depth, scene_res.geometry_ids)
            writer.add_view(
                image_id, K, T_world_camera, rgb, scene_res.depth, objs, masks_full, masks_vis
            )
        writer.finish()

    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"wrote {args.out} ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
