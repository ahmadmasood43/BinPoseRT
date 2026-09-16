"""Synthetic Scenes with known poses, rendered by the same CPU renderer the pipeline uses."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from binposert.render import render_scene
from binposert.transforms import Mat4
from binposert.types import Mat3, ObjectModel, View


def render_view(
    placed: list[tuple[ObjectModel, Mat4]],
    K: Mat3,
    image_size: tuple[int, int] = (240, 320),
    depth_noise_mm: float = 0.0,
    dropout: float = 0.0,
    seed: int = 0,
    camera_id: str = "000000",
) -> tuple[View, list[npt.NDArray[np.bool_]]]:
    """Depth View of ``placed = [(model, T_camera_object), ...]`` with mutual occlusion, optional
    Gaussian depth noise and random dropout; returns the View and each object's visible mask."""
    res = render_scene(placed, K, image_size)
    depth = res.depth.copy()
    rng = np.random.default_rng(seed)
    hit = depth > 0
    if depth_noise_mm > 0:
        depth[hit] += rng.normal(0.0, depth_noise_mm, size=int(hit.sum()))
    if dropout > 0:
        drop = hit & (rng.random(depth.shape) < dropout)
        depth[drop] = 0.0
    masks = [res.geometry_ids == i for i in range(len(placed))]
    view = View(
        camera_id=camera_id,
        K=K,
        T_world_camera=np.eye(4),
        rgb=None,
        depth=depth,
        image_size=image_size,
        scene_id=0,
        image_id=int(camera_id),
    )
    return view, masks
