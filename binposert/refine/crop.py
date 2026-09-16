"""Model side of registration: only the surface the camera could see at the current pose (D8)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from binposert.render import MeshRenderer, unproject_depth
from binposert.transforms import Mat4, invert, transform_points
from binposert.types import Mat3

F64 = npt.NDArray[np.float64]


def visible_model_cloud(
    renderer: MeshRenderer,
    T_camera_object: Mat4,
    K: Mat3,
    image_size: tuple[int, int],
    det_mask: npt.NDArray[np.bool_] | None = None,
    mask_dilate_px: int = 3,
    max_points: int = 3000,
    seed: int = 0,
) -> tuple[F64, F64, int]:
    """Object-frame points and normals of the surface visible from the camera at
    ``T_camera_object``, plus the number of rendered pixels (the predicted silhouette area).

    With ``det_mask`` the crop is further restricted to rendered pixels inside the (dilated)
    Detection mask: self-occlusion comes from the render, occlusion by *other* objects from the
    mask. Without it, model surface hidden behind neighbours has no observed counterpart and drags
    the registration towards the visible patch.
    """
    r = renderer.render(T_camera_object, K, image_size)
    keep = r.mask
    n_pixels = int(keep.sum())
    if det_mask is not None:
        m = det_mask.astype(np.uint8)
        if mask_dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * mask_dilate_px + 1, 2 * mask_dilate_px + 1)
            )
            m = cv2.dilate(m, k).astype(np.uint8)
        keep = keep & (m > 0)
    pts_cam = unproject_depth(r.depth, K, keep)
    pts_obj = transform_points(invert(T_camera_object), pts_cam)
    normals_cam = r.normals_camera[keep]
    R = T_camera_object[:3, :3]
    normals_obj = normals_cam @ R  # R^T applied to row vectors
    if len(pts_obj) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts_obj), size=max_points, replace=False)
        pts_obj, normals_obj = pts_obj[idx], normals_obj[idx]
    return pts_obj, normals_obj, n_pixels
