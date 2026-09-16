"""Hydra entry point used by ``tools/run.py``: ``experiment=A0 dataset=tless``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from binposert.data import BopDataset
from binposert.pipeline.cache import StageCache, canonical_json
from binposert.pipeline.manifest import RunManifest
from binposert.pipeline.stages import RunResult, StageContext, resolve_refs, run_pipeline

log = logging.getLogger("binposert.pipeline")


def make_dataset(cfg: dict[str, Any], repo_root: Path) -> BopDataset:
    d = cfg["dataset"]
    root = Path(d["root"])
    if not root.is_absolute():
        root = repo_root / root
    return BopDataset(
        root,
        split=d["split"],
        models_dir=d.get("models_dir"),
        targets=d.get("targets"),
        continuous_symmetry_steps=int(d.get("continuous_symmetry_steps", 36)),
    )


def make_context(
    cfg: DictConfig | dict[str, Any], repo_root: str | Path | None = None
) -> StageContext:
    plain: dict[str, Any] = (
        OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)  # type: ignore[assignment]
    )
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    dataset = make_dataset(plain, root)
    outputs = Path(plain.get("outputs_root", "outputs"))
    if not outputs.is_absolute():
        outputs = root / outputs
    cache = StageCache(outputs, dataset.name, dataset.split)
    return StageContext(
        dataset=dataset,
        cache=cache,
        cfg=plain,
        repo_root=root,
        run_external=bool(plain.get("run_external", False)),
    )


def run(cfg: DictConfig | dict[str, Any], repo_root: str | Path | None = None) -> RunResult:
    ctx = make_context(cfg, repo_root)
    stages = list(ctx.cfg["stages"])
    experiment = str(ctx.cfg.get("experiment", {}).get("name", "adhoc"))
    config_hash = canonical_json(
        {
            k: ctx.cfg[k]
            for k in ("dataset", "segmenter", "estimator", "refiner", "evaluate", "stages")
            if k in ctx.cfg
        }
    )
    import hashlib

    manifest = RunManifest(
        experiment=experiment,
        dataset=ctx.dataset.name,
        split=ctx.dataset.split,
        config=ctx.cfg,
        config_hash=hashlib.sha256(config_hash.encode()).hexdigest()[:16],
        seed=int(ctx.cfg.get("seed", 0)),
    )
    result = run_pipeline(ctx, stages, manifest)
    last = result.refs[stages[-1]]
    manifest.write(last.dir / "run_manifest.json")
    run_dir = ctx.cache.base.parent.parent / "runs" / f"{experiment}_{ctx.dataset.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(run_dir / "run_manifest.json")
    log.info("run manifest: %s", run_dir / "run_manifest.json")
    return result


def planned_refs(
    cfg: DictConfig | dict[str, Any], repo_root: str | Path | None = None
) -> dict[str, Any]:
    ctx = make_context(cfg, repo_root)
    return resolve_refs(ctx, list(ctx.cfg["stages"]))
