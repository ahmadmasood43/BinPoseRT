"""Content addressing for stage outputs (D12).

``hash = sha256(canonical_json({stage, version, config, dataset, inputs}))[:16]`` where ``inputs``
maps upstream stage names to their hashes. Anything path-like must not enter the hash: the same
cache is produced on a GPU machine and rsync'd to the laptop, where the dataset lives elsewhere.
Keys named ``root`` or starting with ``_`` are therefore dropped before hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HASH_LENGTH = 16
_EXCLUDED_KEYS = {"root"}


def hashable(config: Any) -> Any:
    """Recursively drop path-like keys and normalise scalars so the JSON form is canonical."""
    if isinstance(config, dict):
        return {
            str(k): hashable(v)
            for k, v in sorted(config.items())
            if not str(k).startswith("_") and str(k) not in _EXCLUDED_KEYS
        }
    if isinstance(config, (list, tuple)):
        return [hashable(v) for v in config]
    if isinstance(config, Path):
        raise TypeError(f"paths must not enter a stage hash: {config}")
    if isinstance(config, float) and config.is_integer():
        return int(config)
    return config


def stage_hash(
    stage: str,
    version: str,
    config: dict[str, Any],
    dataset: dict[str, Any],
    inputs: dict[str, str],
) -> str:
    payload = {
        "stage": stage,
        "version": str(version),
        "config": hashable(config),
        "dataset": hashable(dataset),
        "inputs": dict(sorted(inputs.items())),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:HASH_LENGTH]


@dataclass
class StagePlan:
    name: str
    impl: str  # "local" | "gpu"
    version: str
    hash: str
    directory: Path
    config: dict[str, Any]
    inputs: dict[str, Path] = field(default_factory=dict)  # artefact kind -> upstream directory
    input_hashes: dict[str, str] = field(default_factory=dict)  # upstream stage -> hash
    produces: str = ""

    @property
    def is_complete(self) -> bool:
        from binposert.artefacts import is_complete

        return is_complete(self.directory)
