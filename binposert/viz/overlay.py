"""Pose overlays on RGB images, drawn from the CPU renderer's silhouette (D8)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from binposert.render import MeshRenderer
from binposert.transforms import Mat4
from binposert.types import Mat3

Color = tuple[int, int, int]


def overlay_mask(
    rgb: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_], color: Color, alpha: float = 0.45
) -> npt.NDArray[np.uint8]:
    out = rgb.astype(np.float64)
    tint = np.asarray(color, dtype=np.float64)
    out[mask] = (1 - alpha) * out[mask] + alpha * tint
    return out.astype(np.uint8)


def draw_pose_contour(
    rgb: npt.NDArray[np.uint8],
    renderer: MeshRenderer,
    T_camera_object: Mat4,
    K: Mat3,
    color: Color,
    thickness: int = 2,
    fill_alpha: float = 0.0,
) -> npt.NDArray[np.uint8]:
    """Render the model at ``T_camera_object`` and draw its silhouette contour onto ``rgb``."""
    h, w = rgb.shape[:2]
    mask = renderer.render(T_camera_object, K, (h, w)).mask
    out = overlay_mask(rgb, mask, color, fill_alpha) if fill_alpha > 0 else rgb.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def put_label(img: npt.NDArray[np.uint8], text: str, color: Color = (255, 255, 255)) -> None:
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 8 + 7 * len(text)), 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
