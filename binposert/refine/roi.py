"""ROI (region-of-interest) crop for render-bound refinement (F1 / Class B).

A Roi is a (x0, y0, w, h) integer rectangle in full-frame pixel coordinates.
Rendering into a Roi uses ``roi.K_shifted(K)``, which shifts the principal point by the ROI offset
— an exact integer subtraction in float64, no interpolation.  Unprojecting ROI pixel coordinates
with the shifted K produces camera-frame 3D points identical to full-frame unprojection, so ICP
sees the same point sets regardless of crop size.

Cache-safety: ``roi`` and ``roi_margin_px`` are new ``RefinerParams`` fields with current-behaviour
defaults ("none" / 8).  The refine hash comes from the YAML section via ``hashable_config``; adding
defaulted Python fields does not move any existing hash as long as the three ``configs/refiner/*.yaml``
files are not edited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from binposert.transforms import Mat4
from binposert.types import Mat3


@dataclass(frozen=True)
class Roi:
    x0: int
    y0: int
    w: int
    h: int

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) matching numpy/cv2 convention."""
        return (self.h, self.w)

    def K_shifted(self, K: Mat3) -> Mat3:
        """Return K with the principal point shifted to the ROI origin.

        Integer subtraction in float64 — exact, no interpolation.  Pixels in the ROI at
        coordinates (u_roi, v_roi) unproject identically to (u_roi + x0, v_roi + y0) in full frame.
        """
        K2 = K.copy()
        K2[0, 2] = K[0, 2] - self.x0
        K2[1, 2] = K[1, 2] - self.y0
        return K2

    def crop(self, arr: npt.NDArray) -> npt.NDArray:
        """Crop a 2-D (H×W) or 3-D (H×W×C) array to this ROI."""
        return arr[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]

    def contains_sphere(self, T: Mat4, K: Mat3, diameter: float) -> bool:
        """True when the projected bounding sphere at pose T fits inside this ROI."""
        cz = T[2, 3]
        if cz <= 0:
            return False
        fx, fy = K[0, 0], K[1, 1]
        px = T[0, 3] * fx / cz + K[0, 2]
        py = T[1, 3] * fy / cz + K[1, 2]
        r = (diameter / 2.0) * (fx + fy) * 0.5 / cz
        return (
            px - r >= self.x0
            and px + r < self.x0 + self.w
            and py - r >= self.y0
            and py + r < self.y0 + self.h
        )

    @classmethod
    def from_mask_and_sphere(
        cls,
        det_mask: npt.NDArray[np.bool_],
        T_start: Mat4,
        K: Mat3,
        diameter: float,
        image_size: tuple[int, int],
        dilate_px: int = 0,
        margin_px: int = 8,
    ) -> "Roi":
        """Build the ROI as the union of:
        - the detection-mask bounding box dilated by ``dilate_px + margin_px``
        - the projected model bounding sphere at ``T_start`` expanded by ``margin_px``

        Clipped to ``image_size``.  Falls back to the full frame when the mask is empty.
        """
        h_full, w_full = image_size
        ys, xs = np.nonzero(det_mask)
        if len(xs) == 0:
            return cls(0, 0, w_full, h_full)

        d = dilate_px + margin_px
        bx0 = max(0, int(xs.min()) - d)
        by0 = max(0, int(ys.min()) - d)
        bx1 = min(w_full, int(xs.max()) + d + 1)
        by1 = min(h_full, int(ys.max()) + d + 1)

        # Union with projected bounding sphere at T_start
        cz = T_start[2, 3]
        if cz > 0:
            fx, fy = K[0, 0], K[1, 1]
            px = T_start[0, 3] * fx / cz + K[0, 2]
            py = T_start[1, 3] * fy / cz + K[1, 2]
            r = (diameter / 2.0) * (fx + fy) * 0.5 / cz + margin_px
            bx0 = min(bx0, max(0, int(px - r)))
            by0 = min(by0, max(0, int(py - r)))
            bx1 = max(bx1, min(w_full, int(px + r) + 1))
            by1 = max(by1, min(h_full, int(py + r) + 1))

        return cls(bx0, by0, bx1 - bx0, by1 - by0)
