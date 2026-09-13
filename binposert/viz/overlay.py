"""Draw a pose as the silhouette contour of the rendered ObjectModel over an RGB image."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from binposert.render import MeshRenderer
from binposert.transforms import Mat4
from binposert.types import Mat3

RGB = npt.NDArray[np.uint8]
GREEN = (0, 200, 0)
RED = (230, 30, 30)
YELLOW = (240, 200, 0)


def draw_pose_contour(
    image: RGB,
    renderer: MeshRenderer,
    T_camera_object: Mat4,
    K: Mat3,
    color: tuple[int, int, int] = GREEN,
    thickness: int = 2,
) -> RGB:
    """Return a copy of ``image`` with the silhouette contour of the model at the given pose."""
    out = np.ascontiguousarray(image.copy())
    mask = renderer.render(T_camera_object, K, (image.shape[0], image.shape[1])).mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def overlay_pose(
    image: RGB,
    renderer: MeshRenderer,
    K: Mat3,
    T_estimate: Mat4 | None,
    T_ground_truth: Mat4 | None = None,
    label: str | None = None,
) -> RGB:
    """GT in green, estimate in red, optional caption. Either pose may be missing."""
    out = image
    if T_ground_truth is not None:
        out = draw_pose_contour(out, renderer, T_ground_truth, K, GREEN)
    if T_estimate is not None:
        out = draw_pose_contour(out, renderer, T_estimate, K, RED)
    if label:
        out = np.ascontiguousarray(out)
        cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def crop_around(image: RGB, mask: npt.NDArray[np.bool_], pad: float = 0.4, size: int = 256) -> RGB:
    """Square crop around a mask (padded by ``pad`` x its extent), resized to ``size``."""
    ys, xs = np.nonzero(mask)
    h, w = image.shape[:2]
    if len(xs) == 0:
        cx, cy, half = w // 2, h // 2, min(h, w) // 2
    else:
        x0m, x1m, y0m, y1m = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cx, cy = (x0m + x1m) // 2, (y0m + y1m) // 2
        ext = max(x1m - x0m, y1m - y0m, 16)
        half = int(ext * (0.5 + pad))
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    crop = image[y0:y1, x0:x1]
    return np.asarray(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), dtype=np.uint8)
