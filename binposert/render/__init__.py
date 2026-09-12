"""CPU depth / silhouette rendering via Open3D RaycastingScene (D8)."""

from binposert.render.raycast import (
    MeshRenderer,
    RenderResult,
    SceneRenderResult,
    render_depth,
    render_scene,
    unproject_depth,
    visible_surface_points,
)

__all__ = [
    "MeshRenderer",
    "RenderResult",
    "SceneRenderResult",
    "render_depth",
    "render_scene",
    "unproject_depth",
    "visible_surface_points",
]
