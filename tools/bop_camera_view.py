#!/usr/bin/env python
"""Expose one camera of a BOP-25 multi-camera dataset (XYZ-IBD, IPD, …) as a standard BOP dataset
made of symlinks, so the loader, the estimator adapters and bop_toolkit run unchanged (D5: a new
dataset is a config entry, not code).

    uv run python tools/bop_camera_view.py xyzibd xyz            # -> data/bop/xyzibd_xyz
    uv run python tools/bop_camera_view.py xyzibd realsense --splits val test

Per scene, ``<sub>_<camera>`` directories and ``scene_*_<camera>.json`` files are linked to their
suffix-less names; a camera without ``rgb_<camera>`` gets ``rgb -> gray_<camera>`` (single-channel
PNGs read as three identical channels everywhere the core and the adapters load images). Models,
camera intrinsics and target files are linked as they are. Re-running refreshes the links.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SUBDIRS = ("rgb", "gray", "depth", "mask", "mask_visib")
SCENE_FILES = ("scene_camera", "scene_gt", "scene_gt_info")


def link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst.parent)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == rel:
            return
        dst.unlink()
    dst.symlink_to(rel)


def build(root: Path, dataset: str, camera: str, splits: list[str]) -> Path:
    src = root / dataset
    out = root / f"{dataset}_{camera}"
    out.mkdir(exist_ok=True)
    for p in src.iterdir():
        if p.name.startswith("models") or p.name.endswith(".json") or p.name.endswith(".md"):
            link(out / p.name, p)
    cam_file = src / f"camera_{camera}.json"
    if cam_file.exists():
        link(out / "camera.json", cam_file)
    if not (src / "models_cad").is_dir() and (src / "models").is_dir():
        link(out / "models_cad", src / "models")  # the estimator adapters render models_cad
    n_scenes = 0
    for split in splits:
        split_dir = src / split
        if not split_dir.is_dir():
            print(f"skip missing split {split_dir}")
            continue
        for scene in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()):
            dst = out / split / scene.name
            for sub in SUBDIRS:
                d = scene / f"{sub}_{camera}"
                if d.is_dir():
                    link(dst / sub, d)
            if not (scene / f"rgb_{camera}").is_dir() and (scene / f"gray_{camera}").is_dir():
                link(dst / "rgb", scene / f"gray_{camera}")
            for name in SCENE_FILES:
                f = scene / f"{name}_{camera}.json"
                if f.exists():
                    link(dst / f"{name}.json", f)
            n_scenes += 1
    print(f"{out}: {n_scenes} scenes linked for camera {camera!r}")
    return out


def targets_from_multiview(
    src: Path, camera: str, split_dir: Path, name: str, min_visib: float = 0.1
) -> Path:
    """A BOP19-style targets file (``obj_id`` / ``inst_count`` per image) for the images a BOP-25
    multi-view targets file lists for ``camera``, counting the instances annotated with at least
    ``min_visib`` visible fraction in ``scene_gt_info_<camera>``. Needs ground truth."""
    with open(src) as f:
        items = json.load(f)
    out: list[dict[str, int]] = []
    for it in items:
        sid = int(it["scene_id"])
        scene = split_dir / f"{sid:06d}"
        gt_p = scene / f"scene_gt_{camera}.json"
        info_p = scene / f"scene_gt_info_{camera}.json"
        if not gt_p.exists():
            raise FileNotFoundError(f"{gt_p}: no ground truth to count instances from")
        with open(gt_p) as f:
            gt = json.load(f)
        info = json.load(open(info_p)) if info_p.exists() else {}
        for pair in it["im_id"]:
            cam, iid = (pair[0], int(pair[1])) if isinstance(pair, list) else (camera, int(pair))
            if cam != camera:
                continue
            counts: dict[int, int] = {}
            for k, g in enumerate(gt[str(iid)]):
                vis = float(info[str(iid)][k]["visib_fract"]) if str(iid) in info else 1.0
                if vis >= min_visib:
                    counts[int(g["obj_id"])] = counts.get(int(g["obj_id"]), 0) + 1
            for oid, n in sorted(counts.items()):
                out.append({"scene_id": sid, "im_id": iid, "obj_id": oid, "inst_count": n})
    dst = split_dir.parent / name
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"{dst}: {len(out)} targets from {src.name}")
    return dst


def targets_strided(
    split_dir: Path, camera: str, every: int, name: str, min_visib: float = 0.1
) -> Path:
    """A BOP19-style targets file over every ``every``-th image of each scene of a split with
    ground truth (XYZ-IBD ``val``: 50 calibrated views per scene, more than the GPU budget)."""
    out: list[dict[str, int]] = []
    for scene in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        gt = json.load(open(scene / f"scene_gt_{camera}.json"))
        info_p = scene / f"scene_gt_info_{camera}.json"
        info = json.load(open(info_p)) if info_p.exists() else {}
        ids = sorted(int(k) for k in gt)[::every]
        for iid in ids:
            counts: dict[int, int] = {}
            for k, g in enumerate(gt[str(iid)]):
                vis = float(info[str(iid)][k]["visib_fract"]) if str(iid) in info else 1.0
                if vis >= min_visib:
                    counts[int(g["obj_id"])] = counts.get(int(g["obj_id"]), 0) + 1
            for oid, n in sorted(counts.items()):
                out.append(
                    {"scene_id": int(scene.name), "im_id": iid, "obj_id": oid, "inst_count": n}
                )
    dst = split_dir.parent / name
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"{dst}: {len(out)} targets, every {every}th image of {split_dir.name}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset")
    ap.add_argument("camera")
    ap.add_argument("--root", type=Path, default=Path("data/bop"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument(
        "--targets-from-multiview",
        default="",
        help="BOP-25 multi-view targets file name; writes a BOP19-style <name> for the camera",
    )
    ap.add_argument("--targets-split", default="test")
    ap.add_argument("--targets-name", default="test_targets_multiview_bop19style.json")
    ap.add_argument(
        "--targets-strided",
        type=int,
        default=0,
        help="write a targets file over every n-th image of --targets-split (needs GT)",
    )
    args = ap.parse_args()
    out = build(args.root, args.dataset, args.camera, args.splits)
    if args.targets_strided:
        targets_strided(
            args.root / args.dataset / args.targets_split,
            args.camera,
            args.targets_strided,
            args.targets_name,
        )
        link(out / args.targets_name, args.root / args.dataset / args.targets_name)
    if args.targets_from_multiview:
        targets_from_multiview(
            args.root / args.dataset / args.targets_from_multiview,
            args.camera,
            args.root / args.dataset / args.targets_split,
            args.targets_name,
        )
        # the symlink view sees the new file through its dataset-level links
        link(out / args.targets_name, args.root / args.dataset / args.targets_name)


if __name__ == "__main__":
    main()
