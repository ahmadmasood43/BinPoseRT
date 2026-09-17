#!/usr/bin/env python
"""Estimate a dataset's depth↔RGB registration offset without ground truth (B20 → Gamma).

    uv run python tools/depth_rgb_offset.py --dataset tless --scenes 1 6 11 16 --every 10

Depth discontinuities are aligned to RGB edges per image (integer search ± radius, parabolic
sub-pixel refinement); the pooled median is the shift to put in the dataset config as
``depth_shift_px: [du, dv]``. The mm equivalent at the median object depth is printed for
comparison with the ICP bias Beta measured against GT (a check, not an input).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.data.depth_offset import estimate_offset, pooled_offset  # noqa: E402
from binposert.pipeline.artefacts import load_mask, read_detections_table  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--scenes", nargs="+", type=int, default=[1, 6, 11, 16])
    ap.add_argument("--every", type=int, default=10, help="use every n-th target image")
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument(
        "--masks",
        default="",
        help="segment stage directory whose Detection masks focus the edges on objects "
        "(GT-free; without it every edge in the image is used, including the bin and table)",
    )
    ap.add_argument("--dilate", type=int, default=7, help="mask dilation in px")
    ap.add_argument("--out", type=Path, default=None, help="json with per-image and pooled results")
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={args.dataset}"])
    from omegaconf import OmegaConf

    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    plain["dataset"]["depth_shift_px"] = None  # measure the raw sensor data
    ds = make_dataset(plain, REPO)
    dets = read_detections_table(args.masks) if args.masks else None

    rows = []
    estimates = []
    for sid in args.scenes:
        for iid in ds.image_ids(sid)[:: args.every]:
            view, _ = ds.load_view(sid, iid)
            assert view.rgb is not None and view.depth is not None
            mask = None
            if dets is not None:
                sel = dets[(dets.scene_id == sid) & (dets.image_id == iid)]
                mask = np.zeros(view.image_size, dtype=bool)
                for mp in sel.mask_path:
                    mask |= load_mask(args.masks, str(mp))
                k = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)
                mask = cv2.dilate(mask.astype(np.uint8), k) > 0
            e = estimate_offset(view.rgb, view.depth, radius_px=args.radius, mask=mask)
            z = float(np.median(view.depth[view.depth > 0]))
            estimates.append(e)
            rows.append({"scene_id": sid, "image_id": iid, "z_median_mm": z, **e.__dict__})
            print(
                f"scene {sid:2d} image {iid:3d}: du {e.du:+.2f} dv {e.dv:+.2f} px "
                f"(edge agreement {e.score:.2f} vs {e.score_unshifted:.2f} unshifted, "
                f"{e.n_edges} edges)"
            )
    du, dv = pooled_offset(estimates)
    n_used = sum(1 for e in estimates if e.score >= 0.4 and e.score - e.score_unshifted >= 0.02)
    z = float(np.median([r["z_median_mm"] for r in rows]))
    K = np.asarray(ds.camera(args.scenes[0], ds.image_ids(args.scenes[0])[0])["cam_K"]).reshape(
        3, 3
    )
    mm = (du * z / K[0, 0], dv * z / K[1, 1])
    print(
        f"\npooled (median of {n_used} clear-peak images of {len(rows)}): "
        f"du {du:+.2f} dv {dv:+.2f} px"
    )
    print(
        f"at the median depth {z:.0f} mm that is ({mm[0]:+.2f}, {mm[1]:+.2f}) mm along camera x, y"
    )
    print(f"dataset config: depth_shift_px: [{du:.2f}, {dv:.2f}]")
    if args.out is not None:
        args.out.write_text(
            json.dumps({"rows": rows, "du": du, "dv": dv, "mm_at_median_z": mm, "z": z}, indent=2)
        )


if __name__ == "__main__":
    main()
