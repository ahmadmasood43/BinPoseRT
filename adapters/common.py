"""Shared plumbing for the estimator adapters (D15).

Adapters run inside an estimator's own environment (docker/<name> or envs/<name>) with the
repository root on ``PYTHONPATH``; they import only ``binposert.types`` and
``binposert.pipeline.artefacts`` from the core (numpy / pandas / imageio only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass
class AdapterArgs:
    dataset_root: Path
    split: str
    out: Path
    params: dict[str, Any]
    targets: Path | None
    detections: Path | None
    upstream: Path
    checkpoints: Path
    work: Path
    scenes: list[int] | None
    max_images: int | None
    device: str
    extra: dict[str, Any] = field(default_factory=dict)


def parse_args(name: str, description: str, needs_detections: bool = False) -> AdapterArgs:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--dataset-root", required=True, type=Path, help="BOP dataset directory")
    ap.add_argument("--split", required=True, help="split directory name, e.g. test_primesense")
    ap.add_argument("--targets", default="", help="BOP test_targets json (optional)")
    ap.add_argument("--out", required=True, type=Path, help="stage output directory (D12)")
    ap.add_argument("--params", default="{}", help="JSON params from configs/")
    if needs_detections:
        ap.add_argument("--detections", required=True, type=Path, help="segment stage directory")
    ap.add_argument("--upstream", type=Path, default=REPO / "third_party" / name)
    ap.add_argument("--checkpoints", type=Path, default=REPO / "data" / "checkpoints")
    ap.add_argument("--work", type=Path, default=REPO / "data" / "work" / name)
    ap.add_argument("--scenes", default="", help="comma-separated scene ids (smoke runs)")
    ap.add_argument("--max-images", type=int, default=None, help="cap on images (smoke runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--onboard-only",
        action="store_true",
        help="only build the per-object onboarding artefacts (templates), then exit",
    )
    ns = ap.parse_args()
    return AdapterArgs(
        dataset_root=ns.dataset_root.resolve(),
        split=ns.split,
        out=ns.out.resolve(),
        params=json.loads(ns.params) if ns.params else {},
        targets=Path(ns.targets).resolve() if ns.targets else None,
        detections=getattr(ns, "detections", None) and Path(ns.detections).resolve(),
        upstream=ns.upstream.resolve(),
        checkpoints=ns.checkpoints.resolve(),
        work=ns.work.resolve(),
        scenes=[int(s) for s in ns.scenes.split(",") if s] or None,
        max_images=ns.max_images,
        device=ns.device,
        extra={"onboard_only": bool(ns.onboard_only)},
    )


def target_images(args: AdapterArgs) -> list[tuple[int, int]]:
    """(scene_id, image_id) pairs to process, honouring targets, --scenes and --max-images."""
    split_dir = args.dataset_root / args.split
    pairs: list[tuple[int, int]] = []
    if args.targets is not None and args.targets.exists():
        with open(args.targets) as f:
            seen: set[tuple[int, int]] = set()
            for it in json.load(f):
                key = (int(it["scene_id"]), int(it["im_id"]))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        pairs.sort()
    else:
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()):
            with open(scene_dir / "scene_camera.json") as f:
                ims = sorted(int(k) for k in json.load(f))
            pairs.extend((int(scene_dir.name), i) for i in ims)
    if args.scenes is not None:
        pairs = [p for p in pairs if p[0] in args.scenes]
    if args.max_images is not None:
        pairs = pairs[: args.max_images]
    return pairs


def target_objects(args: AdapterArgs) -> dict[tuple[int, int], dict[int, int]] | None:
    if args.targets is None or not args.targets.exists():
        return None
    out: dict[tuple[int, int], dict[int, int]] = {}
    with open(args.targets) as f:
        for it in json.load(f):
            key = (int(it["scene_id"]), int(it["im_id"]))
            out.setdefault(key, {})[int(it["obj_id"])] = int(it.get("inst_count", 1))
    return out


def load_scene_camera(args: AdapterArgs, scene_id: int) -> dict[str, Any]:
    with open(args.dataset_root / args.split / f"{scene_id:06d}" / "scene_camera.json") as f:
        data: dict[str, Any] = json.load(f)
    return data


def rgb_path(args: AdapterArgs, scene_id: int, image_id: int) -> Path:
    d = args.dataset_root / args.split / f"{scene_id:06d}" / "rgb"
    for ext in ("png", "jpg"):
        p = d / f"{image_id:06d}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(d / f"{image_id:06d}.png")


def depth_path(args: AdapterArgs, scene_id: int, image_id: int) -> Path | None:
    p = args.dataset_root / args.split / f"{scene_id:06d}" / "depth" / f"{image_id:06d}.png"
    return p if p.exists() else None


def upstream_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — informational only
        return "unknown"


def write_adapter_info(args: AdapterArgs, name: str, extra: dict[str, Any]) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    info = {
        "adapter": name,
        "upstream": str(args.upstream),
        "upstream_commit": upstream_commit(args.upstream),
        "params": args.params,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "targets": str(args.targets) if args.targets else None,
        "scenes": args.scenes,
        "max_images": args.max_images,
        "gpu": gpu_name(),
        "finished_at": time.time(),
        **extra,
    }
    with open(args.out / "adapter.json", "w") as f:
        json.dump(info, f, indent=2, default=str)


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == str(target):
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            return  # a real directory: leave it alone
    link.symlink_to(target)


def bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.t0
        self.t0 = now
        return dt
