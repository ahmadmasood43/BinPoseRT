"""``run_manifest.json`` (D12): everything needed to reproduce and attribute one run."""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import binposert


def git_commit(repo: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return {"nvidia_smi": [line.strip() for line in out.stdout.splitlines() if line.strip()]}
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {"nvidia_smi": []}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class RunManifest:
    experiment: str
    dataset: str
    split: str
    config: dict[str, Any]
    config_hash: str
    stages: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # name -> {hash, dir, cached, s}
    started_at: float = field(default_factory=time.time)
    seed: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def record_stage(self, name: str, hash_: str, dir_: Path, cached: bool, seconds: float) -> None:
        self.stages[name] = {
            "hash": hash_,
            "dir": str(dir_),
            "cached": cached,
            "seconds": round(seconds, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "dataset": self.dataset,
            "split": self.split,
            "git_commit": git_commit(),
            "binposert_version": binposert.__version__,
            "config_hash": self.config_hash,
            "config": self.config,
            "seed": self.seed,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            **gpu_info(),
            "started_at": self.started_at,
            "wall_seconds": round(time.time() - self.started_at, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "stages": self.stages,
            "notes": self.notes,
        }

    def write(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
