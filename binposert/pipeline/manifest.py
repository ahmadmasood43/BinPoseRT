"""``run_manifest.json`` (D12): everything needed to reproduce or audit one run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from binposert import __version__
from binposert.data import BopDataset


def git_info(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "-C", str(repo), "status", "--porcelain"]))
        return {"commit": commit, "dirty": dirty}
    except Exception:  # not a checkout, git missing …
        return {"commit": None, "dirty": None}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "gpu": None,
        "cuda_driver": None,
    }
    if shutil.which("nvidia-smi"):
        try:
            out = _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            )
            info["gpu"] = [line.strip() for line in out.splitlines() if line.strip()]
            smi = _run(["nvidia-smi"])
            if "CUDA Version:" in smi:
                info["cuda_driver"] = smi.split("CUDA Version:")[1].split()[0]
        except Exception:
            pass
    return info


def dataset_fingerprint(dataset: BopDataset, image_keys: list[tuple[int, int]]) -> dict[str, Any]:
    """Cheap identity of the data actually used: models_info hash + covered scenes/images."""
    h = hashlib.sha256()
    h.update((dataset.models_dir / "models_info.json").read_bytes())
    for s, i in image_keys:
        h.update(f"{s}:{i};".encode())
    return {
        "name": dataset.name,
        "split": dataset.split,
        "models_dir": dataset.models_dir.name,
        "n_images": len(image_keys),
        "scene_ids": sorted({s for s, _ in image_keys}),
        "object_ids": dataset.object_ids,
        "fingerprint": h.hexdigest()[:16],
    }


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform != "darwin" else ru / (1024.0 * 1024.0)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"written": datetime.now(timezone.utc).isoformat(timespec="seconds"), **manifest}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def package_versions() -> dict[str, str]:
    out = {"binposert": __version__}
    for name in ("numpy", "scipy", "open3d", "cv2", "pandas", "pyarrow", "hydra"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            continue
    return out


def _run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()
