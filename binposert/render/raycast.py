"""Headless depth rendering with Open3D's Embree-backed RaycastingScene.

Runs on CPU with no EGL/OSMesa. Signatures are the contract a GPU rasteriser must satisfy later.
Depth images are float64 in millimetres (z along the optical axis), 0 where nothing is hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import open3d as o3d

from binposert.transforms import Mat4, invert, transform_points
from binposert.types import Mat3, ObjectModel

NO_HIT = o3d.t.geometry.RaycastingScene.INVALID_ID


@dataclass(frozen=True)
class RenderResult:
    depth: npt.NDArray[np.float64]  # (H, W) mm, 0 = background
    mask: npt.NDArray[np.bool_]  # (H, W)
    normals_camera: npt.NDArray[
        np.float64
    ]  # (H, W, 3) unit normals in camera frame, 0 on background
    primitive_ids: npt.NDArray[np.int64]  # (H, W) triangle index, -1 on background


@dataclass(frozen=True)
class SceneRenderResult:
    depth: npt.NDArray[np.float64]  # (H, W) mm, closest surface over all objects
    geometry_ids: npt.NDArray[np.int64]  # (H, W) index into the input list, -1 on background


class MeshRenderer:
    """One mesh, BVH built once, rendered at arbitrary camera poses."""

    def __init__(self, vertices: npt.ArrayLike, faces: npt.ArrayLike) -> None:
        v = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
        f = np.ascontiguousarray(np.asarray(faces, dtype=np.uint32))
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
        self.n_faces = int(len(f))

    @classmethod
    def from_model(cls, model: ObjectModel) -> MeshRenderer:
        return cls(model.vertices, model.faces)

    def render(self, T_camera_object: Mat4, K: Mat3, image_size: tuple[int, int]) -> RenderResult:
        h, w = image_size
        rays = _pinhole_rays(K, T_camera_object, w, h)
        ans = self._scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy().astype(np.float64)
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        hit = prim != NO_HIT
        # Rays live in the object frame (the frame the triangles were added in). t_hit is the ray
        # parameter; depth along the camera optical axis is t * (R_camera_object @ dir)_z.
        R = T_camera_object[:3, :3]
        dirs_obj = rays.numpy()[..., 3:6].astype(np.float64)
        dirs_cam = dirs_obj @ R.T
        depth = np.where(hit, t_hit * dirs_cam[..., 2], 0.0)
        normals_obj = ans["primitive_normals"].numpy().astype(np.float64)
        normals_cam = normals_obj @ R.T
        # orient normals toward the camera
        flip = np.sum(normals_cam * dirs_cam, axis=-1) > 0
        normals_cam[flip] *= -1.0
        normals_cam[~hit] = 0.0
        prim[~hit] = -1
        return RenderResult(depth=depth, mask=hit, normals_camera=normals_cam, primitive_ids=prim)


def render_depth(
    model: ObjectModel, T_camera_object: Mat4, K: Mat3, image_size: tuple[int, int]
) -> RenderResult:
    """Convenience one-shot render (rebuilds the BVH; use MeshRenderer for repeated renders)."""
    return MeshRenderer.from_model(model).render(T_camera_object, K, image_size)


def render_scene(
    placed: list[tuple[ObjectModel, Mat4]], K: Mat3, image_size: tuple[int, int]
) -> SceneRenderResult:
    """Render several objects with mutual occlusion; ``placed[i] = (model, T_camera_object)``."""
    h, w = image_size
    scene = o3d.t.geometry.RaycastingScene()
    for model, T in placed:
        v_cam = transform_points(T, model.vertices).astype(np.float32)
        scene.add_triangles(
            o3d.core.Tensor(np.ascontiguousarray(v_cam)),
            o3d.core.Tensor(np.ascontiguousarray(model.faces.astype(np.uint32))),
        )
    rays = _pinhole_rays(K, np.eye(4), w, h)
    ans = scene.cast_rays(rays)
    geom = ans["geometry_ids"].numpy().astype(np.int64)
    hit = geom != NO_HIT
    t_hit = ans["t_hit"].numpy().astype(np.float64)
    depth = np.where(hit, t_hit * rays.numpy()[..., 5].astype(np.float64), 0.0)
    geom[~hit] = -1
    return SceneRenderResult(depth=depth, geometry_ids=geom)


def unproject_depth(
    depth: npt.NDArray[np.float64], K: Mat3, mask: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.float64]:
    """Depth image (mm) -> (N, 3) camera-frame points for valid (depth > 0 and masked) pixels."""
    valid = depth > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def visible_surface_points(
    renderer: MeshRenderer, T_camera_object: Mat4, K: Mat3, image_size: tuple[int, int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predicted visible surface at a pose: (points_object, points_camera, normals_camera).

    This is the model side of visibility-aware registration (D8): only what the camera could see at
    the current pose is matched against observed depth.
    """
    r = renderer.render(T_camera_object, K, image_size)
    pts_cam = unproject_depth(r.depth, K)
    pts_obj = transform_points(invert(T_camera_object), pts_cam)
    normals = r.normals_camera[r.mask]
    return pts_obj, pts_cam, normals


def _pinhole_rays(K: Mat3, T_camera_world: Mat4, w: int, h: int) -> o3d.core.Tensor:
    # Open3D shoots the ray for pixel (u, v) through (u + 0.5, v + 0.5). OpenCV/BOP intrinsics treat
    # integer pixel coordinates as pixel centres, so shift the principal point by half a pixel to
    # make rendered depth unproject exactly with ``unproject_depth``.
    K_o3d = np.asarray(K, dtype=np.float64).copy()
    K_o3d[0, 2] += 0.5
    K_o3d[1, 2] += 0.5
    return o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K_o3d),
        o3d.core.Tensor(np.asarray(T_camera_world, dtype=np.float64)),
        int(w),
        int(h),
    )
