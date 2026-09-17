"""Stage implementations and the DAG runner (D12).

Every stage is ``run(ctx, out_dir, upstream) -> None`` and is a pure function of the upstream stage
directories, its own config and its version string. GPU stages (CNOS, FoundPose, MegaPose) are
*external*: their config carries a command template that invokes an adapter from ``adapters/``
inside the estimator's environment. On a machine without that environment the runner refuses to
run them and prints the exact command instead; the cached output is rsync'd in from the GPU machine.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.evaluate import PosePrediction, evaluate_localisation, write_bop_csv
from binposert.evaluate.stratified import stratify_by_visibility
from binposert.pipeline.artefacts import (
    DETECTIONS_FILE,
    HYPOTHESES_FILE,
    DetectionRecord,
    DetectionWriter,
    HypothesisRecord,
    read_detections,
    read_hypotheses_table,
    write_hypotheses,
)
from binposert.pipeline.cache import StageCache, StageRef
from binposert.pipeline.manifest import RunManifest
from binposert.pose.synthetic import GroundTruthPerturbedEstimator
from binposert.segment import GroundTruthSegmenter
from binposert.types import QualitySignals

log = logging.getLogger("binposert.pipeline")

STAGE_ORDER = [
    "segment",
    "coarse_pose",
    "refine",
    "associate",
    "fuse",
    "confidence",
    "nbv",
    "evaluate",
]
STAGE_VERSIONS = {
    "segment": "1",
    "coarse_pose": "1",
    "refine": "4",
    "associate": "1",
    "fuse": "1",
    "evaluate": "8",
}


class ExternalStageMissing(RuntimeError):
    """A GPU stage's cached output is absent and running it here is not allowed."""


@dataclass
class StageContext:
    dataset: BopDataset
    cache: StageCache
    cfg: dict[str, Any]
    repo_root: Path
    run_external: bool = False
    external_env: dict[str, str] = field(default_factory=dict)

    @property
    def dataset_cfg(self) -> dict[str, Any]:
        d: dict[str, Any] = self.cfg["dataset"]
        return d


StageFn = Callable[[StageContext, Path, dict[str, StageRef]], None]


# ----------------------------------------------------------------------------- helpers


def default_workers() -> int:
    """``n_workers: 0`` in a stage config: half the logical cores, at least one. Gamma ran 15
    workers on a 16-thread machine next to a GPU stage for hours and the host crashed five times in
    four days (docs/milestone_gamma_decision.md G21); half the cores keeps the thermal and memory
    load of a stage where a shared workstation can sustain it. Set ``n_workers`` explicitly to
    use more."""
    return max(1, (os.cpu_count() or 2) // 2)


def hashable_config(section: dict[str, Any]) -> dict[str, Any]:
    """The part of a stage's config that determines its output (machine-specific keys excluded)."""
    return {k: v for k, v in section.items() if k not in ("command", "env", "kind")}


def _render_command(
    template: list[str],
    ctx: StageContext,
    out_dir: Path,
    upstream: dict[str, StageRef],
    params: Any,
) -> list[str]:
    ds = ctx.dataset
    subst = {
        "repo": str(ctx.repo_root),
        "out_dir": str(out_dir),
        "dataset_root": str(ds.root),
        "dataset_name": ds.name,
        "split": ds.split,
        "models_dir": str(ds.models_dir),
        "targets": str(ds.targets_file) if ds.targets_file is not None else "",
        "params_json": json.dumps(params, sort_keys=True),
    }
    for name, ref in upstream.items():
        subst[f"upstream_{name}"] = str(ref.dir)
    return [str(part).format(**subst) for part in template]


def run_external(
    ctx: StageContext,
    section: dict[str, Any],
    out_dir: Path,
    upstream: dict[str, StageRef],
    expected_file: str,
) -> None:
    cmd = _render_command(
        list(section["command"]), ctx, out_dir, upstream, section.get("params", {})
    )
    pretty = " ".join(shlex.quote(c) for c in cmd)
    if not ctx.run_external:
        raise ExternalStageMissing(
            f"stage '{section['name']}' is a GPU stage and its cache is missing at {out_dir}.\n"
            f"Run it on the GPU machine (or pass run_external=true here):\n  {pretty}\n"
            f"then rsync {out_dir} to this machine."
        )
    log.info("running external stage: %s", pretty)
    env = {
        **os.environ,
        "PYTHONPATH": str(ctx.repo_root),
        **ctx.external_env,
        **section.get("env", {}),
    }
    subprocess.run(cmd, cwd=ctx.repo_root, env=env, check=True)
    if not (out_dir / expected_file).exists():
        raise RuntimeError(f"external stage finished but {out_dir / expected_file} is missing")


# ----------------------------------------------------------------------------- stages


def segment_stage(ctx: StageContext, out_dir: Path, upstream: dict[str, StageRef]) -> None:
    section = ctx.cfg["segmenter"]
    if section.get("kind", "internal") == "external":
        run_external(ctx, section, out_dir, upstream, DETECTIONS_FILE)
        return
    if section["name"] != "gt":
        raise ValueError(f"unknown internal segmenter {section['name']!r}")
    params = section.get("params", {})
    seg = GroundTruthSegmenter(
        ctx.dataset, min_visible_fraction=params.get("min_visible_fraction", 0.1)
    )
    writer = DetectionWriter(out_dir)
    ds = ctx.dataset
    for scene_id in ds.scene_ids:
        for image_id in ds.image_ids(scene_id):
            view, _ = ds.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
            object_ids = ds.target_object_ids(scene_id, image_id) or ds.object_ids
            t0 = time.perf_counter()
            dets = seg.segment(view, object_ids)
            dt = time.perf_counter() - t0
            for d in dets:
                writer.add(DetectionRecord(scene_id, image_id, d, time_s=dt))
    writer.close()


def coarse_pose_stage(ctx: StageContext, out_dir: Path, upstream: dict[str, StageRef]) -> None:
    section = ctx.cfg["estimator"]
    if section.get("kind", "internal") == "external":
        run_external(ctx, section, out_dir, upstream, HYPOTHESES_FILE)
        return
    if section["name"] != "gt_perturbed":
        raise ValueError(f"unknown internal estimator {section['name']!r}")
    params = section.get("params", {})
    est = GroundTruthPerturbedEstimator(
        ctx.dataset,
        t_sigma_mm=float(params.get("t_sigma_mm", 0.0)),
        r_sigma_deg=float(params.get("r_sigma_deg", 0.0)),
        seed=int(params.get("seed", 0)),
    )
    seg_dir = upstream["segment"].dir
    ds = ctx.dataset
    recs: list[HypothesisRecord] = []
    for scene_id in ds.scene_ids:
        for image_id in ds.image_ids(scene_id):
            view, _ = ds.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
            dets = read_detections(seg_dir, scene_id, image_id, load_masks=False)
            t0 = time.perf_counter()
            hyps = est.estimate(view, dets)
            dt = time.perf_counter() - t0
            recs.extend(HypothesisRecord(scene_id, image_id, h, time_s=dt) for h in hyps)
    write_hypotheses(out_dir, recs)


def refine_stage(ctx: StageContext, out_dir: Path, upstream: dict[str, StageRef]) -> None:
    from binposert.pipeline.refine_stage import params_from_config, run_refine

    section = ctx.cfg["refiner"]
    n_workers = int(section.get("n_workers", 0)) or default_workers()
    summary = run_refine(
        ctx.dataset,
        upstream["segment"].dir,
        upstream["coarse_pose"].dir,
        out_dir,
        params_from_config(section),
        n_workers=n_workers,
    )
    with open(out_dir / "refine_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "refined %d hypotheses, rejection rate %.3f (%s), median %.3f s",
        summary["n_hypotheses"],
        summary["rejection_rate"],
        summary["reasons"],
        summary["median_seconds"],
    )


def associate_stage(ctx: StageContext, out_dir: Path, upstream: dict[str, StageRef]) -> None:
    from binposert.pipeline.multiview_stage import associate_params_from_config, run_associate

    source = upstream.get("refine") or upstream["coarse_pose"]
    summary = run_associate(
        ctx.dataset, source.dir, out_dir, associate_params_from_config(ctx.cfg["multiview"])
    )
    with open(out_dir / "associate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "%d groups of %d views -> %d tracks from %d hypotheses (views per track %s)",
        summary["n_groups"],
        summary["n_views"],
        summary["n_tracks"],
        summary["n_hypotheses"],
        summary["views_per_track"],
    )


def fuse_stage(ctx: StageContext, out_dir: Path, upstream: dict[str, StageRef]) -> None:
    from binposert.pipeline.multiview_stage import fuse_params_from_config, run_fuse

    section = ctx.cfg["fusion"]
    n_workers = int(section.get("n_workers", 0)) or default_workers()
    summary = run_fuse(
        ctx.dataset,
        upstream["segment"].dir,
        upstream["associate"].dir,
        out_dir,
        fuse_params_from_config(section),
        n_workers=n_workers,
    )
    with open(out_dir / "fuse_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("fused %d tracks (%s): %s", summary["n_tracks"], summary["method"], summary)


def evaluate_stage(ctx: StageContext, out_dir: Path, upstream: dict[str, StageRef]) -> None:
    section = ctx.cfg["evaluate"]
    source = upstream.get("fuse") or upstream.get("refine") or upstream["coarse_pose"]
    table = read_hypotheses_table(source.dir)
    images: set[tuple[int, int]] | None = None
    if source.stage == "fuse":
        # multi-view rows are scored on the images of their view groups only
        from binposert.pipeline.multiview_stage import GROUPS_FILE

        with open(source.dir / GROUPS_FILE) as f:
            groups = json.load(f)
        images = {(int(s), int(i)) for s, gs in groups.items() for g in gs for i in g}
        write_targets_subset(ctx.dataset, images, out_dir / TARGETS_SUBSET_FILE)
    preds, times = predictions_from_table(
        table, score_signal=section.get("score_signal", "pose_score")
    )
    ds = ctx.dataset
    method = section.get("method_name", "binposert")
    bop_name = ctx.dataset_cfg.get("bop_name") or ds.name  # bop_toolkit's dataset name
    csv_path = out_dir / f"{method}_{bop_name}-{ctx.dataset_cfg.get('bop_split', ds.split)}.csv"
    write_bop_csv(csv_path, preds)

    n_workers = int(section.get("n_workers", 0)) or default_workers()
    report = evaluate_localisation(
        preds,
        ds,
        n_model_points=int(section.get("n_model_points", 2000)),
        with_vsd=bool(section.get("with_vsd", True)),
        n_workers=n_workers,
        images=images,
    )
    rows = pd.DataFrame(report.rows)
    rows.to_parquet(out_dir / "gt_rows.parquet", index=False)
    bins = [tuple(b) for b in section.get("visibility_bins", [[0.1, 0.3], [0.3, 0.6], [0.6, 1.0]])]
    strat = stratify_by_visibility(rows, bins)
    summary = {
        "method": method,
        "bop_csv": csv_path.name,
        "n_predictions": len(preds),
        "n_images": len(images) if images is not None else None,
        "n_gt": report.n_gt,
        "ar": report.ar,
        "ar_vsd": report.ar_vsd,
        "ar_mssd": report.ar_mssd,
        "ar_mspd": report.ar_mspd,
        "per_object": {str(k): v for k, v in report.per_object.items()},
        "by_visibility": strat,
        "image_times_s": {"median": float(np.nanmedian(list(times.values()))) if times else None},
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "summary.md", "w") as f:
        f.write(format_summary(summary))
    log.info(
        "AR %.4f (VSD %.4f MSSD %.4f MSPD %.4f) on %d GT",
        report.ar,
        report.ar_vsd,
        report.ar_mssd,
        report.ar_mspd,
        report.n_gt,
    )


TARGETS_SUBSET_FILE = "targets_subset.json"


def write_targets_subset(ds: BopDataset, images: set[tuple[int, int]], path: Path) -> None:
    """The dataset's BOP targets restricted to ``images`` so ``tools/bop_eval.sh`` (bop_toolkit)
    scores exactly the ground truth the core evaluator scored."""
    items: list[dict[str, int]] = []
    if ds.targets is not None:
        for (sid, iid), objs in sorted(ds.targets.items()):
            if (sid, iid) in images:
                items.extend(
                    {"scene_id": sid, "im_id": iid, "obj_id": oid, "inst_count": n}
                    for oid, n in sorted(objs.items())
                )
    else:
        for sid, iid in sorted(images):
            items.append({"scene_id": sid, "im_id": iid})
    with open(path, "w") as f:
        json.dump(items, f)


def predictions_from_table(
    table: pd.DataFrame, score_signal: str = "pose_score"
) -> tuple[list[PosePrediction], dict[tuple[int, int], float]]:
    """Hypotheses table -> BOP predictions. BOP wants one time per image, so the maximum recorded
    ``time_s`` of an image is used for all of its predictions."""
    from binposert.pipeline.artefacts import columns_to_transform

    times: dict[tuple[int, int], float] = {}
    for key_raw, grp in table.groupby(["scene_id", "image_id"]):
        s, i = (int(x) for x in cast(tuple[Any, Any], key_raw))
        t = grp["time_s"].max()
        times[(s, i)] = float(t) if pd.notna(t) else -1.0
    preds: list[PosePrediction] = []
    for _, row in table.iterrows():
        score = _score(row, score_signal)
        key = (int(row["scene_id"]), int(row["image_id"]))
        preds.append(
            PosePrediction(
                scene_id=key[0],
                image_id=key[1],
                object_id=int(row["object_id"]),
                score=score,
                T_camera_object=columns_to_transform(row),
                time_s=times[key],
            )
        )
    return preds, times


def _score(row: Any, score_signal: str) -> float:
    for name in (score_signal, "pose_score", "seg_score"):
        if name in QualitySignals.field_names() or name in row:
            v = row.get(name, np.nan)
            if pd.notna(v):
                return float(v)
    return 1.0


def format_summary(s: dict[str, Any]) -> str:
    lines = [
        f"# {s['method']}",
        "",
        f"AR = {s['ar']:.4f}  "
        f"(VSD {s['ar_vsd']:.4f} · MSSD {s['ar_mssd']:.4f} · MSPD {s['ar_mspd']:.4f})",
        f"{s['n_predictions']} predictions, {s['n_gt']} GT, BOP CSV `{s['bop_csv']}`",
        "",
        "| visibility | n GT | AR | AR_VSD | AR_MSSD | AR_MSPD | success (MSSD < 0.1d) |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in s["by_visibility"]:
        lines.append(
            f"| [{b['lo']:.2f}, {b['hi']:.2f}) | {b['n_gt']} | {b['ar']:.4f} | {b['ar_vsd']:.4f} | "
            f"{b['ar_mssd']:.4f} | {b['ar_mspd']:.4f} | {b['success_0.1d']:.4f} |"
        )
    return "\n".join(lines) + "\n"


STAGES: dict[str, StageFn] = {
    "segment": segment_stage,
    "coarse_pose": coarse_pose_stage,
    "refine": refine_stage,
    "associate": associate_stage,
    "fuse": fuse_stage,
    "evaluate": evaluate_stage,
}

STAGE_INPUTS: dict[str, list[str]] = {
    "segment": [],
    "coarse_pose": ["segment"],
    "refine": ["segment", "coarse_pose"],
    "associate": ["coarse_pose", "refine"],
    "fuse": ["segment", "associate"],
    "evaluate": ["coarse_pose", "refine", "fuse"],
}


DEPTH_STAGES = ("refine", "fuse", "evaluate")


def stage_config(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "segment":
        return hashable_config(cfg["segmenter"])
    if stage == "coarse_pose":
        return hashable_config(cfg["estimator"])
    if stage == "refine":
        section = hashable_config(cfg["refiner"])
    elif stage == "associate":
        section = hashable_config(cfg["multiview"])
    elif stage == "fuse":
        section = hashable_config(cfg["fusion"])
    else:
        section = hashable_config(cfg.get(stage, {}))
    # a depth↔RGB shift in the dataset config changes what every depth-reading stage sees
    shift = cfg.get("dataset", {}).get("depth_shift_px")
    if stage in DEPTH_STAGES and shift:
        section = {**section, "depth_shift_px": [float(x) for x in shift]}
    return section


# ----------------------------------------------------------------------------- runner


@dataclass
class RunResult:
    refs: dict[str, StageRef]
    manifest: RunManifest


def resolve_refs(ctx: StageContext, stages: list[str]) -> dict[str, StageRef]:
    """Compute the cache location of every requested stage without running anything."""
    refs: dict[str, StageRef] = {}
    for name in stages:
        ups = [refs[u] for u in STAGE_INPUTS[name] if u in refs]
        refs[name] = ctx.cache.ref(name, STAGE_VERSIONS[name], stage_config(ctx.cfg, name), ups)
    return refs


def run_pipeline(ctx: StageContext, stages: list[str], manifest: RunManifest) -> RunResult:
    refs: dict[str, StageRef] = {}
    for name in stages:
        if name not in STAGES:
            raise ValueError(f"stage {name!r} is not implemented yet")
        ups = {u: refs[u] for u in STAGE_INPUTS[name] if u in refs}
        version = STAGE_VERSIONS[name]
        config = stage_config(ctx.cfg, name)
        ref = ctx.cache.ref(name, version, config, list(ups.values()))
        t0 = time.perf_counter()
        if ref.complete:
            log.info("stage %-12s cached  %s", name, ref.dir)
            manifest.record_stage(name, ref.hash, ref.dir, True, 0.0)
        else:
            log.info("stage %-12s running %s", name, ref.dir)
            ctx.cache.begin(ref, version, config, list(ups.values()))
            STAGES[name](ctx, ref.dir, ups)
            ctx.cache.finish(ref)
            manifest.record_stage(name, ref.hash, ref.dir, False, time.perf_counter() - t0)
        refs[name] = ref
    return RunResult(refs=refs, manifest=manifest)
