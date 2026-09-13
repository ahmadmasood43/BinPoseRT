#!/usr/bin/env python
"""Run one ablation row (D12): ``uv run python tools/run.py experiment=A0 dataset=tless``.

Every run writes a BOP CSV, ``run_manifest.json`` and reports under ``outputs/``; stages whose
outputs are cached are skipped, GPU stages that are not cached stop the run with the adapter
command that fills them. ``pipeline.plan_only=true`` prints the plan and exits.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binposert.pipeline import MissingGpuArtefact, Pipeline  # noqa: E402

log = logging.getLogger("binposert.run")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(config, dict)
    pipeline = Pipeline(config)
    print(pipeline.describe())
    if config["pipeline"].get("plan_only"):
        return 0
    try:
        result = pipeline.run()
    except MissingGpuArtefact as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(2)
    if result.report is not None:
        r = result.report
        print(
            f"\nAR = {r['ar']:.4f}  (VSD {r['ar_vsd']:.4f} · MSSD {r['ar_mssd']:.4f} · "
            f"MSPD {r['ar_mspd']:.4f})  n_gt={r['n_gt']}"
        )
        print(f"results: {result.results_csv}")
    print(f"manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    main()
