"""Shared plumbing for the estimator adapters (run inside docker/<name>/ on the GPU machine).

An adapter is started with ``run --request <adapter_request.json>``; the request is written by
``tools/run.py`` next to the directory the adapter must fill (D12). Adapters import only
``binposert.artefacts`` / ``binposert.types`` (numpy, pandas, pyarrow, imageio) so the core can be
installed with ``pip install --no-deps`` into the estimator's own environment.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("adapter")


@dataclass
class Request:
    stage: str
    hash: str
    config: dict[str, Any]
    dataset: dict[str, Any]
    inputs: dict[str, Path]
    image_keys: list[tuple[int, int]]
    objects: dict[tuple[int, int], dict[int, int]] | None
    output_dir: Path
    path: Path

    @property
    def dataset_root(self) -> Path:
        root = Path(os.environ.get("BINPOSERT_DATASET_ROOT", self.dataset["root"]))
        if not root.is_dir():
            raise FileNotFoundError(
                f"dataset root {root} not found; set BINPOSERT_DATASET_ROOT if it lives elsewhere"
            )
        return root

    @property
    def dataset_name(self) -> str:
        return str(self.dataset["name"])

    @property
    def split(self) -> str:
        return str(self.dataset["split"])

    def object_ids(self, scene_id: int, image_id: int) -> list[int] | None:
        if self.objects is None:
            return None
        return sorted(self.objects.get((scene_id, image_id), {}))

    @classmethod
    def load(cls, path: str | Path) -> Request:
        path = Path(path).resolve()
        with open(path) as f:
            raw = json.load(f)
        objects = None
        if raw.get("objects") is not None:
            objects = {}
            for key, counts in raw["objects"].items():
                s, i = key.split("/")
                objects[(int(s), int(i))] = {int(o): int(c) for o, c in counts.items()}
        # The request may have been written on another machine: the output directory is the
        # directory the request file lives in, whatever the recorded absolute path says.
        out_dir = path.parent
        inputs = {k: _relocate(Path(v), out_dir) for k, v in raw.get("inputs", {}).items()}
        return cls(
            stage=raw["stage"],
            hash=raw["hash"],
            config=raw["config"],
            dataset=raw["dataset"],
            inputs=inputs,
            image_keys=[(int(s), int(i)) for s, i in raw["image_keys"]],
            objects=objects,
            output_dir=out_dir,
            path=path,
        )


def _relocate(recorded: Path, out_dir: Path) -> Path:
    """Map ``outputs/<ds>/<split>/<stage>/<hash>`` recorded elsewhere onto this machine's tree."""
    if recorded.is_dir():
        return recorded
    parts = recorded.parts
    # out_dir = .../outputs/<ds>/<split>/<stage>/<hash>; the sibling stage dir shares the split root
    split_root = out_dir.parents[1]
    candidate = split_root / parts[-2] / parts[-1]
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"input artefact {recorded} not found (tried {candidate})")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def clean_output_dir(req: Request) -> None:
    """Remove stale outputs of a previous attempt but keep the request file itself."""
    for p in req.output_dir.iterdir():
        if p == req.path:
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_command(
    cmd: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None
) -> float:
    """Run a subprocess with inherited stdout, returning the wall time in seconds."""
    log.info("$ %s", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True)
    return time.perf_counter() - t0


def scene_camera(dataset_root: Path, split: str, scene_id: int) -> dict[str, Any]:
    return read_json(dataset_root / split / f"{scene_id:06d}" / "scene_camera.json")


def rgb_path(dataset_root: Path, split: str, scene_id: int, image_id: int) -> Path:
    d = dataset_root / split / f"{scene_id:06d}" / "rgb"
    for ext in (".png", ".jpg"):
        p = d / f"{image_id:06d}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no rgb image for scene {scene_id} image {image_id} under {d}")


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None
