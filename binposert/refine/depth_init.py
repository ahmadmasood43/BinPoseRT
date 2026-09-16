"""Translation initialisation from observed depth (D8, added in Beta).

RGB coarse estimators place the object correctly in the image plane but poorly along the viewing
ray (Alpha, FoundPose on T-LESS: MSPD 82 vs VSD/MSSD ≈ 50; median coarse depth error 25 mm, 52 %
beyond 0.25 d). Local ICP cannot bridge that — no correspondences within its search radius, or a
move the displacement cap must reject. So before ICP the coarse translation is scaled along its
ray so that the rendered depth matches the observed depth on the pixels both cover: the object's
image position is unchanged, only its distance. Measured offline on 1371 sampled hypotheses this
alone raised the share within 0.1 d from 11.5 % to 46.5 % (89 % improved, 7 % worsened by > 1 mm).
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from binposert.render import RenderResult
from binposert.transforms import Mat4


def translation_from_depth(
    render: RenderResult,
    depth: npt.NDArray[np.float64],
    det_mask: npt.NDArray[np.bool_],
    T_camera_object: Mat4,
    erode_px: int = 2,
    min_overlap: int = 50,
) -> tuple[Mat4, float, int]:
    """Shift ``T_camera_object`` along the ray through the object origin by the median difference
    between observed and rendered depth on the overlap of the rendered silhouette and the (eroded)
    Detection mask. Returns ``(T, z_shift_mm, n_overlap)``; with fewer than ``min_overlap``
    overlapping pixels or a non-positive result the pose is returned unchanged (shift 0)."""
    m = det_mask
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(det_mask.astype(np.uint8), kernel).astype(bool)
    sel = render.mask & m & (depth > 0)
    n = int(sel.sum())
    tz = float(T_camera_object[2, 3])
    if n < min_overlap or tz <= 0:
        return T_camera_object.copy(), 0.0, n
    dz = float(np.median(depth[sel] - render.depth[sel]))
    if tz + dz <= 0:
        return T_camera_object.copy(), 0.0, n
    T = T_camera_object.copy()
    T[:3, 3] = T_camera_object[:3, 3] * (tz + dz) / tz
    return T, dz, n
