"""The GT-free depth↔RGB offset estimate recovers a known shift (D16)."""

import numpy as np
import pytest

from binposert import transforms as tf
from binposert.data.depth_offset import (
    OffsetEstimate,
    estimate_offset,
    pooled_offset,
    shift_depth,
)
from tests.synth import box_model, cylinder_model, render_view, simple_K

K = simple_K()
SIZE = (240, 320)


def _rgb_from_depth(depth):
    """A flat 'RGB' image whose only edges are the object silhouettes of ``depth``."""
    img = np.full((*depth.shape, 3), 90, dtype=np.uint8)
    img[depth > 0] = 200
    return img


@pytest.mark.parametrize("shift", [(0.0, 0.0), (2.0, -3.0), (-1.0, 4.0)])
def test_estimate_recovers_a_known_depth_shift(shift):
    placed = [
        (box_model(), tf.rotvec_T([1, 0.3, 0], 30.0, t=[-40, 10, 420])),
        (cylinder_model(), tf.rotvec_T([0.2, 1, 0], 60.0, t=[50, -20, 450])),
    ]
    view, _ = render_view(placed, K, SIZE, depth_noise_mm=0.5)
    rgb = _rgb_from_depth(view.depth)
    # the sensor delivered the depth map displaced by -shift; the estimate is the shift back
    depth = shift_depth(view.depth, -shift[0], -shift[1])
    e = estimate_offset(rgb, depth, radius_px=6)
    assert abs(e.du - shift[0]) < 0.6 and abs(e.dv - shift[1]) < 0.6, (e, shift)
    assert e.score >= e.score_unshifted


def test_pooled_offset_ignores_images_without_a_peak():
    good = [OffsetEstimate(0.1, -3.2, 0.6, 0.5, 4000), OffsetEstimate(-0.2, -3.4, 0.55, 0.45, 4200)]
    flat = [OffsetEstimate(5.0, 8.0, 0.25, 0.24, 40000)]  # noisy view, no real peak
    du, dv = pooled_offset(good + flat)
    assert abs(du - (-0.05)) < 1e-9 and abs(dv - (-3.3)) < 1e-9
    assert np.isnan(pooled_offset(flat)[0])


def test_shift_depth_moves_pixels_and_zero_fills():
    d = np.zeros((10, 10))
    d[4, 4] = 500.0
    s = shift_depth(d, 2, -1)
    assert s[3, 6] == 500.0 and s[4, 4] == 0.0 and s.sum() == 500.0
    assert shift_depth(d, 0.0, 0.0) is d
