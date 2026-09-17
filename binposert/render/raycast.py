"""Headless depth rendering with Open3D's Embree-backed RaycastingScene.

Runs on CPU with no EGL/OSMesa. Signatures are the contract a GPU rasteriser must satisfy later.
Depth images are float64 in millimetres (z along the optical axis), 0 where nothing is hit.

Rays are cast single-threaded (``nthreads=1``). Open3D 0.19's parallel ``cast_rays`` corrupts its
output under multi-process load: with 15 worker processes on 16 cores, 126 of 1751 renders of
720×540 raised inside numpy on garbage indices and 5 more silently differed from a re-render;
``nthreads=1`` gave 0 of both (13 ms vs 5 ms per render). Stages parallelise over scenes with one
process per scene instead (``binposert.pipeline.pool``). Gamma saw one more corrupted cast under a
load average of ~20 (a stage plus the test suite): every cast is therefore validated (shapes,
primitive ids below the face count, finite positive hit distances) and re-cast up to
``CAST_RETRIES`` times before failing loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import open3d as o3d

from binposert.transforms import Mat4, invert, transform_points
from binposert.types import Mat3, ObjectModel

NO_HIT = o3d.t.geometry.RaycastingScene.INVALID_ID
RAY_THREADS = 1  # see the module docstring
CAST_RETRIES = 3


class RenderCorrupted(RuntimeError):
    """``cast_rays`` returned inconsistent output ``CAST_RETRIES`` times in a row."""


def _cast_checked(
    scene: o3d.t.geometry.RaycastingScene,
    rays: o3d.core.Tensor,
    shape: tuple[int, int],
    n_primitives: int,
    n_geometries: int,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Cast and validate; returns ``(t_hit, primitive_ids, geometry_ids, primitive_normals)``."""
    last = ""
    for _ in range(CAST_RETRIES):
        try:
            ans = scene.cast_rays(rays, nthreads=RAY_THREADS)
            t_hit = np.asarray(ans["t_hit"].numpy(), dtype=np.float64)
            prim = np.asarray(ans["primitive_ids"].numpy(), dtype=np.int64)
            geom = np.asarray(ans["geometry_ids"].numpy(), dtype=np.int64)
            normals = np.asarray(ans["primitive_normals"].numpy(), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 — garbage output raises inside numpy
            last = f"{type(e).__name__}: {e}"
            continue
        if t_hit.shape != shape or prim.shape != shape or geom.shape != shape:
            last = f"shapes {t_hit.shape} {prim.shape} {geom.shape} != {shape}"
            continue
        if normals.shape != (*shape, 3):
            last = f"normals shape {normals.shape}"
            continue
        hit = prim != NO_HIT
        if hit.any():
            p_hit, g_hit, d_hit = prim[hit], geom[hit], t_hit[hit]
            if p_hit.min() < 0 or p_hit.max() >= n_primitives:
                last = f"primitive id out of range [{p_hit.min()}, {p_hit.max()}] / {n_primitives}"
                continue
            if g_hit.min() < 0 or g_hit.max() >= n_geometries:
                last = f"geometry id out of range [{g_hit.min()}, {g_hit.max()}] / {n_geometries}"
                continue
            if not (np.isfinite(d_hit).all() and (d_hit > 0).all()):
                last = "non-finite or non-positive hit distance"
                continue
        return t_hit, prim, geom, normals
    raise RenderCorrupted(f"cast_rays output invalid {CAST_RETRIES} times: {last}")


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
        # The whole render is retried, not only the cast: under load the corruption has also hit
        # numpy's own index buffers *after* the cast validated (an IndexError with a high bit set
        # in the index, e.g. 2097319 for a 1080-row image), so the post-processing is guarded too.
        last: Exception | None = None
        for _ in range(CAST_RETRIES):
            try:
                return self._render_once(T_camera_object, K, image_size)
            except (IndexError, ValueError) as e:
                last = e
        raise RenderCorrupted(f"render failed {CAST_RETRIES} times: {last}")

    def _render_once(
        self, T_camera_object: Mat4, K: Mat3, image_size: tuple[int, int]
    ) -> RenderResult:
        h, w = image_size
        rays = _pinhole_rays(K, T_camera_object, w, h)
        t_hit, prim, _, normals_obj = _cast_checked(self._scene, rays, (h, w), self.n_faces, 1)
        hit = prim != NO_HIT
        # Rays live in the object frame (the frame the triangles were added in). t_hit is the ray
        # parameter; depth along the camera optical axis is t * (R_camera_object @ dir)_z.
        R = T_camera_object[:3, :3]
        dirs_obj = rays.numpy()[..., 3:6].astype(np.float64)
        dirs_cam = dirs_obj @ R.T
        depth = np.where(hit, t_hit * dirs_cam[..., 2], 0.0)
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
    n_faces = max(int(len(m.faces)) for m, _ in placed)  # primitive ids are per geometry
    t_hit, _, geom, _ = _cast_checked(scene, rays, (h, w), n_faces, len(placed))
    hit = geom != NO_HIT
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
