"""GT-free estimate of the depth↔RGB registration offset of a sensor (Beta decision B20).

Beta found ICP converging ~2.5 mm below the T-LESS ground truth along camera y with perfect
fitness: the depth map is registered a few pixels off the RGB image the poses were annotated on.
The offset is measured here without any pose annotation: depth discontinuities (object
boundaries in the depth map) are matched against RGB edges by a search over integer pixel shifts,
refined to sub-pixel precision with a parabola through the best cell, and the per-image shifts of
several images are pooled by their median. ``shift = (du, dv)`` is the translation that moves the
depth map *onto* the RGB image; :meth:`BopDataset.load_depth` applies it when the dataset config
carries ``depth_shift_px``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OffsetEstimate:
    du: float  # pixels, + moves the depth map right
    dv: float  # pixels, + moves the depth map down
    score: float  # robust agreement of the shifted depth edges with the RGB edges (0..1)
    score_unshifted: float
    n_edges: int


def depth_edges(
    depth: F64, jump_mm: float = 40.0, mask: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.bool_]:
    """Pixels where the median-filtered depth jumps by more than ``jump_mm``: object boundaries."""
    valid = depth > 0
    d = cv2.medianBlur(depth.astype(np.float32), 5)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    e = (np.hypot(gx, gy) > jump_mm) & valid
    if mask is not None:
        e &= mask
    return e


def rgb_edges(rgb: npt.NDArray[np.uint8], low: int = 60, high: int = 140) -> npt.NDArray[np.bool_]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    return cv2.Canny(gray, low, high) > 0


def estimate_offset(
    rgb: npt.NDArray[np.uint8],
    depth: F64,
    radius_px: int = 8,
    mask: npt.NDArray[np.bool_] | None = None,
    jump_mm: float = 40.0,
    sigma_px: float = 1.5,
) -> OffsetEstimate:
    """The pixel shift of the depth map that best aligns its discontinuities with the RGB edges.

    The objective is the mean of ``exp(-(d / sigma)^2)`` over depth-edge pixels, ``d`` being the
    distance to the nearest RGB edge: a robust inlier score that ignores the many depth "edges"
    noisy low-elevation views produce far from any image edge, where a mean distance would be
    dominated by them."""
    e_rgb = rgb_edges(rgb)
    e_d = depth_edges(depth, jump_mm, mask)
    ys, xs = np.nonzero(e_d)
    if len(xs) < 50 or not e_rgb.any():
        return OffsetEstimate(float("nan"), float("nan"), float("nan"), float("nan"), int(len(xs)))
    dist = cv2.distanceTransform((~e_rgb).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = dist.shape
    r = radius_px
    grid = np.empty((2 * r + 1, 2 * r + 1))
    for i in range(2 * r + 1):
        y = np.clip(ys + (i - r), 0, h - 1)
        for j in range(2 * r + 1):
            x = np.clip(xs + (j - r), 0, w - 1)
            grid[i, j] = np.mean(np.exp(-((dist[y, x] / sigma_px) ** 2)))
    i0, j0 = (int(v) for v in np.unravel_index(int(np.argmax(grid)), grid.shape))
    du, dv = float(j0 - r), float(i0 - r)
    # sub-pixel refinement: parabola through the three cells around the maximum, per axis
    if 0 < j0 < 2 * r:
        du += _parabola_vertex(-grid[i0, j0 - 1], -grid[i0, j0], -grid[i0, j0 + 1])
    if 0 < i0 < 2 * r:
        dv += _parabola_vertex(-grid[i0 - 1, j0], -grid[i0, j0], -grid[i0 + 1, j0])
    return OffsetEstimate(du, dv, float(grid[i0, j0]), float(grid[r, r]), int(len(xs)))


def _parabola_vertex(a: float, b: float, c: float) -> float:
    denom = a - 2 * b + c
    if denom <= 1e-9:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def pooled_offset(
    estimates: list[OffsetEstimate], min_score: float = 0.4, min_gain: float = 0.02
) -> tuple[float, float]:
    """Median over the images with a clear peak: agreement at the peak at least ``min_score`` and
    at least ``min_gain`` above the unshifted value (views whose depth is too noisy to show a
    peak — low-elevation T-LESS views — would otherwise pull the median towards zero)."""
    ok = [
        e
        for e in estimates
        if np.isfinite(e.du)
        and np.isfinite(e.dv)
        and e.score >= min_score
        and e.score - e.score_unshifted >= min_gain
    ]
    if not ok:
        return float("nan"), float("nan")
    return float(np.median([e.du for e in ok])), float(np.median([e.dv for e in ok]))


def shift_depth(depth: F64, du: float, dv: float) -> F64:
    """Translate the depth map by ``(du, dv)`` pixels (nearest-neighbour, so no invalid/valid
    blending); uncovered pixels become 0 (missing)."""
    if du == 0.0 and dv == 0.0:
        return depth
    M = np.array([[1.0, 0.0, du], [0.0, 1.0, dv]], dtype=np.float64)
    out = cv2.warpAffine(
        depth,
        M,
        (depth.shape[1], depth.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.asarray(out, dtype=np.float64)
