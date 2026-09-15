#!/usr/bin/env python
"""Run one experiment row of the ablation matrix (D12).

    uv run python tools/run.py experiment=A0 dataset=tless
    uv run python tools/run.py experiment=smoke                  # mini fixture, no GPU
    uv run python tools/run.py experiment=A1 dataset=tless run_external=true   # on the GPU machine

Each stage is cached under outputs/<dataset>/<split>/<stage>/<hash>/ and skipped when complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.pipeline.run import run  # noqa: E402


@hydra.main(version_base=None, config_path=str(REPO / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    result = run(cfg, repo_root=REPO)
    last = list(result.refs.values())[-1]
    print(f"done: {last.stage} -> {last.dir}")


if __name__ == "__main__":
    main()
