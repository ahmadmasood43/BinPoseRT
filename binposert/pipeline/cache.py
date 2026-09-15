"""Content-addressed stage cache (D12, ADR-0002).

A stage output lives in ``<root>/<dataset>/<split>/<stage>/<hash>/`` with
``hash = sha256(canonical_json(stage, version, config, upstream_hashes))[:16]``. The directory is
valid only when it holds a ``_SUCCESS`` marker; partial outputs from a crashed run are discarded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from binposert.pipeline.artefacts import (
    SUCCESS_MARKER,
    is_complete,
    mark_success,
    write_stage_info,
)

HASH_LENGTH = 16


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, floats as repr, tuples as lists."""
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), allow_nan=True)


def _normalise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return obj.item()
    return obj


def stage_hash(stage: str, version: str, config: Any, upstream_hashes: list[str]) -> str:
    payload = {
        "stage": stage,
        "version": version,
        "config": config,
        "upstream": list(upstream_hashes),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:HASH_LENGTH]


@dataclass(frozen=True)
class StageRef:
    """Resolved location of one stage's output."""

    stage: str
    hash: str
    dir: Path

    @property
    def complete(self) -> bool:
        return is_complete(self.dir)


class StageCache:
    """Addresses stage outputs under ``<root>/<dataset>/<split>/``."""

    def __init__(self, root: str | Path, dataset: str, split: str) -> None:
        self.base = Path(root) / dataset / split

    def ref(self, stage: str, version: str, config: Any, upstream: list[StageRef]) -> StageRef:
        h = stage_hash(stage, version, config, [u.hash for u in upstream])
        return StageRef(stage=stage, hash=h, dir=self.base / stage / h)

    def begin(self, ref: StageRef, version: str, config: Any, upstream: list[StageRef]) -> Path:
        """Prepare a fresh output directory (wiping incomplete leftovers); record provenance."""
        if ref.dir.exists() and not ref.complete:
            shutil.rmtree(ref.dir)
        ref.dir.mkdir(parents=True, exist_ok=True)
        write_stage_info(
            ref.dir,
            {
                "stage": ref.stage,
                "hash": ref.hash,
                "version": version,
                "config": _normalise(config),
                "upstream": {u.stage: u.hash for u in upstream},
            },
        )
        return ref.dir

    @staticmethod
    def finish(ref: StageRef) -> None:
        mark_success(ref.dir)

    @staticmethod
    def marker(ref: StageRef) -> Path:
        return ref.dir / SUCCESS_MARKER
