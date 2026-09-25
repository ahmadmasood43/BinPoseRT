#!/usr/bin/env python
"""D13 frozen update-path benchmark (``configs/benchmark.yaml`` protocol).

    OMP_NUM_THREADS=1 uv run python tools/benchmark.py --dataset tless --row A8_k4

Drives ``iter_track_samples`` from ``profile_update_path.py`` to the configured
warmup + timed track count, enforces ``max_load_average``, samples process memory,
and writes ``outputs/<dataset>_benchmark_<row>.json`` (schema_version 2, superset of
the Δ16 profile JSON so the two files can be diffed directly).

Full-pipeline mode (``--full-pipeline``) is a separate leg: it runs one
``tools/run.py`` subprocess on a small scene subset and reads timing back from
``run_manifest.json``.  It must NOT run concurrently with any sweep workers
(``bench`` step in ``run_deployment.sh`` enforces this).
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO))

from profile_update_path import STAGES, iter_track_samples, pct  # noqa: E402

from binposert.confidence import ConfidenceModel  # noqa: E402
from binposert.multiview import FusionParams, associate, fuse_track, lift_to_world  # noqa: E402
from binposert.pipeline.artefacts import (  # noqa: E402
    HypothesisRecord,
    hypothesis_from_row,
    hypothesis_to_row,
)
from binposert.pipeline.confidence_stage import (  # noqa: E402
    aggregate_members,
    params_from_config,
    score_members,
    with_diameter,
)
from binposert.pipeline.multiview_stage import (  # noqa: E402
    associate_params_from_config,
    scene_weights,
)
from binposert.pipeline.refine_stage import params_from_config as refine_params  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.refine import Refiner  # noqa: E402
from binposert.types import ObjectTrack  # noqa: E402

# ---------------------------------------------------------------------------
# Memory sampler
# ---------------------------------------------------------------------------


def _sample_memory_mb(interval: float, stop: threading.Event) -> list[float]:
    """Sample RSS in MB every ``interval`` seconds until ``stop`` is set."""
    samples: list[float] = []
    while not stop.is_set():
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux: kB; macOS: bytes
            if sys.platform != "darwin":
                rss = rss * 1024
            samples.append(rss / 1024 / 1024)
        except Exception:
            pass
        stop.wait(interval)
    return samples


def _peak_vram_mb() -> float:
    """Query nvidia-smi for the current process's VRAM usage (0 if unavailable)."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(pid):
                return float(parts[1])
    except Exception:
        pass
    return 0.0


def _proc_tree_rss_mb(pid: int) -> float:
    """Sum VmRSS (MB) of ``pid`` and every descendant via ``/proc`` (Linux only).  Needed for the
    full-pipeline leg: the external CNOS/FoundPose adapters run as further subprocesses of
    ``tools/run.py``, so RUSAGE_SELF on our own process (used by the update-path leg) would miss
    them entirely."""
    try:
        children_of: dict[int, list[int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
                children_of.setdefault(ppid, []).append(int(entry.name))
            except Exception:
                continue
        total_kb = 0
        stack, seen = [pid], set()
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            try:
                for line in (Path("/proc") / str(p) / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total_kb += int(line.split()[1])
                        break
            except Exception:
                pass
            stack.extend(children_of.get(p, []))
        return total_kb / 1024.0
    except Exception:
        return 0.0


def _gpu_mem_used_mb() -> float:
    """Whole-GPU memory used (MB), summed across all GPUs (0 if unavailable).  Whole-GPU rather
    than per-PID: the external adapters are several subprocess hops from `tools/run.py`'s own
    PID, so PID-matching (as `_peak_vram_mb` does for the in-process update-path leg) is fragile
    here; the caller subtracts a pre-run baseline to isolate this run's contribution."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode()
        return sum(float(x) for x in out.split())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Update-path benchmark
# ---------------------------------------------------------------------------


def run_update_path(
    manifest: dict,
    bench_cfg: dict,
    *,
    outputs: Path,
    dataset_name: str,
    row: str,
) -> dict[str, Any]:
    """Run the update-path benchmark leg and return the result dict."""
    proto = bench_cfg["protocol"]
    up = bench_cfg["update_path"]
    warmup_tracks = proto["warmup_tracks"]
    timed_tracks = proto["timed_tracks"]
    max_load = proto["max_load_average"]

    load1 = os.getloadavg()[0]
    if load1 > max_load:
        raise RuntimeError(
            f"Load average {load1:.2f} exceeds max_load_average {max_load} "
            f"(configs/benchmark.yaml).  Wait for the machine to quiet down."
        )

    cfg = manifest["config"]
    dataset = make_dataset(cfg, REPO)
    r_params = refine_params(cfg["refiner"])
    a_params = associate_params_from_config(cfg["multiview"], REPO)
    c_params = params_from_config(cfg["confidence"], REPO)
    model_h = ConfidenceModel.load(c_params.model_h)
    model_f = ConfidenceModel.load(c_params.model_f)
    refiners: dict[int, Refiner] = {}

    times: dict[str, list[float]] = {s: [] for s in STAGES}
    per_track_total: list[float] = []
    n_warmup_tracks = 0
    n_timed_tracks = 0
    load_start = os.getloadavg()

    # Memory sampler runs in background throughout the timed portion.
    mem_stop = threading.Event()
    mem_samples: list[float] = []
    mem_thread: threading.Thread | None = None

    for gi, sample in enumerate(
        iter_track_samples(
            dataset,
            manifest,
            seed=up.get("seed", 0),
            max_groups=None,  # we stop by track count below
        )
    ):
        # Lazy Refiner construction outside the timer (warm by end of warmup phase).
        for row_h in sample.hyps_rows:
            hyp = hypothesis_from_row(row_h)
            if hyp.object_id not in refiners:
                refiners[hyp.object_id] = Refiner(dataset.load_model(hyp.object_id), r_params)

        warm = n_warmup_tracks < warmup_tracks
        sid, image_ids = sample.scene_id, sample.image_ids
        views, masks, hyps_rows = sample.views, sample.masks, sample.hyps_rows

        # Start memory sampler at first timed group.
        if not warm and mem_thread is None:
            mem_thread = threading.Thread(
                target=lambda: mem_samples.extend(
                    _sample_memory_mb(
                        proto["memory"]["sample_interval_s"] if "memory" in proto else 0.2, mem_stop
                    )
                ),
                daemon=True,
            )
            mem_thread.start()

        # --- refine ---
        refined_rows = []
        t_refine_group = 0.0
        for row_h in hyps_rows:
            hyp = hypothesis_from_row(row_h)
            iid = int(row_h.image_id)
            t0 = time.perf_counter()
            out = refiners[hyp.object_id].refine(views[iid], masks[(iid, hyp.detection_id)], hyp)
            dt = time.perf_counter() - t0
            t_refine_group += dt
            if not warm:
                times["refine"].append(dt)
            r = dict(row_h)
            r.update(hypothesis_to_row(HypothesisRecord(sid, iid, out.hypothesis, dt)))
            refined_rows.append(r)

        refined = pd.DataFrame(refined_rows)
        models = {int(o): dataset.load_model(int(o)) for o in refined.object_id.unique()}

        # --- associate ---
        t0 = time.perf_counter()
        weights = scene_weights(refined, dataset, a_params.weights).to_dict()
        by_view: dict[str, list[Any]] = {}
        for iid in image_ids:
            v = views[iid]
            by_view[v.camera_id] = [
                lift_to_world(hypothesis_from_row(r), v.T_world_camera, float(weights[i]))
                for i, r in refined[refined.image_id == iid].iterrows()
            ]
        res = associate(by_view, models, a_params.association)
        t_assoc = time.perf_counter() - t0
        n_tracks = max(1, len(res.tracks))

        # --- fuse ---
        fused_rows = []
        t_fuse_group = 0.0
        for t in res.tracks:
            members = res.members[t.track_id]
            t0 = time.perf_counter()
            out_f = fuse_track(
                ObjectTrack(t.track_id, t.object_id, [m.hypothesis for m in members]),
                members,
                models[t.object_id],
                FusionParams(method="mean"),
            )
            dt = time.perf_counter() - t0
            t_fuse_group += dt
            if not warm:
                times["fuse"].append(dt)
            fused_rows.append(
                {
                    "scene_id": sid,
                    "group_id": 0,
                    "track_id": t.track_id,
                    "object_id": t.object_id,
                    "n_members": len(members),
                    "n_aligned": out_f.n_aligned,
                    "weight_sum": float(out_f.weights.sum()),
                    **out_f.fused.signals.to_row(),
                }
            )

        # --- confidence ---
        t0 = time.perf_counter()
        member_rows = []
        for t in res.tracks:
            for m in res.members[t.track_id]:
                h = m.hypothesis
                mr = refined[
                    (refined.image_id == int(h.camera_id))
                    & (refined.detection_id == h.detection_id)
                    & (refined.hypothesis_id == h.hypothesis_id)
                ].iloc[0]
                member_rows.append({**dict(mr), "group_id": 0, "track_id": t.track_id})
        members_df = pd.DataFrame(member_rows)
        p_h = score_members(members_df, dataset, model_h)
        agg = aggregate_members(members_df, p_h)
        fused_df = with_diameter(pd.DataFrame(fused_rows), dataset).merge(
            agg, on=["scene_id", "group_id", "track_id"]
        )
        model_f.predict_proba(fused_df)
        t_conf = time.perf_counter() - t0

        if not warm:
            times["associate"].extend([t_assoc / n_tracks] * n_tracks)
            times["confidence"].extend([t_conf / n_tracks] * n_tracks)
            per_track_total.extend(
                [(t_refine_group + t_assoc + t_fuse_group + t_conf) / n_tracks] * n_tracks
            )
            n_timed_tracks += n_tracks
        else:
            n_warmup_tracks += n_tracks

        label = " (warm-up)" if warm else ""
        print(
            f"group {gi + 1} scene {sid} views {image_ids}: {len(hyps_rows)} hyps, "
            f"{n_tracks} tracks, refine {t_refine_group:.2f} s, assoc {t_assoc * 1000:.0f} ms, "
            f"fuse {t_fuse_group * 1000:.0f} ms, conf {t_conf * 1000:.0f} ms{label} "
            f"[warmup {n_warmup_tracks}/{warmup_tracks}, timed {n_timed_tracks}/{timed_tracks}]"
        )

        if n_timed_tracks >= timed_tracks:
            break

    mem_stop.set()
    if mem_thread is not None:
        mem_thread.join(timeout=2)

    load_end = os.getloadavg()
    peak_rss = max(mem_samples) if mem_samples else 0.0
    peak_vram = _peak_vram_mb()

    return {
        "schema_version": proto["schema_version"],
        "dataset": dataset_name,
        "row": row,
        "protocol": proto,
        "n_warmup_tracks_actual": n_warmup_tracks,
        "n_timed_tracks": n_timed_tracks,
        "single_threaded": os.environ.get("OMP_NUM_THREADS") == "1",
        "cuda_sync": proto["cuda_sync"],
        "load_average_start": load_start,
        "load_average_end": list(load_end),
        "peak_rss_mb": round(peak_rss, 1),
        "peak_vram_mb": peak_vram,
        "per_stage_ms": {k: pct(v) for k, v in times.items()},
        "per_track_total_ms": pct(per_track_total),
        "per_hypothesis_refine_ms": pct(times["refine"]),
        "target_p95_ms": up["target_p95_ms"],
    }


# ---------------------------------------------------------------------------
# Full-pipeline benchmark (D13's second budget, P3 / P18): a separate leg from the
# update-path leg above.  It runs one real `tools/run.py` invocation from raw images
# through `evaluate` on a small scene subset, so it is the only leg that measures the
# GPU-backed segment/coarse_pose stages and true end-to-end wall time.
# ---------------------------------------------------------------------------


def build_subset_targets(
    ds: Any, n_scenes: int, n_images_per_scene: int, seed: int
) -> set[tuple[int, int]]:
    """Deterministic ``n_scenes`` x ``n_images_per_scene`` slice of ``ds``'s own targets."""
    rng = np.random.default_rng(seed)
    scene_ids = sorted(ds.scene_ids)
    n_s = min(n_scenes, len(scene_ids))
    chosen_scenes = sorted(int(s) for s in rng.choice(scene_ids, size=n_s, replace=False))
    subset: set[tuple[int, int]] = set()
    for sid in chosen_scenes:
        image_ids = sorted(ds.image_ids(sid))
        n_i = min(n_images_per_scene, len(image_ids))
        chosen_images = rng.choice(image_ids, size=n_i, replace=False)
        subset.update((sid, int(iid)) for iid in chosen_images)
    return subset


def run_full_pipeline(
    bench_cfg: dict[str, Any],
    repo: Path,
    dataset_name: str,
    row: str,
    force: bool = False,
) -> dict[str, Any]:
    """Run (or reuse) one `tools/run.py` subprocess on a small subset and report timing.

    Only `segment`, `coarse_pose` and `refine` have genuine per-image (or per-image-aggregated)
    timing in their own artefacts (`time_s` / `seconds` columns keyed by scene_id, image_id);
    `associate`/`fuse`/`confidence`/`evaluate` operate on multi-view groups, not single images, so
    they are reported as a stage total rather than a fabricated per-image distribution (P18)."""
    from binposert.data.bop import BopDataset
    from binposert.pipeline.artefacts import read_detections_table, read_hypotheses_table
    from binposert.pipeline.stages import write_targets_subset

    fp = bench_cfg["full_pipeline"]
    n_scenes = int(fp["subset_scenes"])
    n_images_per_scene = int(fp["subset_images_per_scene"])
    outputs_root = repo / fp["outputs_root"]
    seed = int(fp["seed"])

    ds_cfg = yaml.safe_load((repo / "configs" / "dataset" / f"{dataset_name}.yaml").read_text())
    ds_root = Path(ds_cfg["root"])
    if not ds_root.is_absolute():
        ds_root = repo / ds_root
    ds = BopDataset(
        ds_root,
        split=ds_cfg["split"],
        models_dir=ds_cfg.get("models_dir"),
        targets=ds_cfg.get("targets"),
        name=ds_cfg.get("name"),
    )

    subset = build_subset_targets(ds, n_scenes, n_images_per_scene, seed)
    outputs_root.mkdir(parents=True, exist_ok=True)
    subset_path = outputs_root / "full_pipeline_targets_subset.json"
    write_targets_subset(ds, subset, subset_path)

    manifest_path = outputs_root / "runs" / f"{row}_{dataset_name}" / "run_manifest.json"
    rss_samples: list[float] = []
    vram_samples: list[float] = []
    gpu_baseline = 0.0

    if manifest_path.exists() and not force:
        print(f"full-pipeline manifest exists at {manifest_path}, reusing (pass --force to re-run)")
    else:
        cmd = [
            "uv",
            "run",
            "python",
            str(repo / "tools" / "run.py"),
            "experiment=A8",
            f"dataset={dataset_name}",
            "multiview.params.groups.n_views=4",
            f"dataset.targets={subset_path}",
            f"outputs_root={fp['outputs_root']}",
            "run_external=true",
            f"experiment.name={row}",
            f"seed={seed}",
        ]
        print("running:", " ".join(cmd))

        gpu_baseline = _gpu_mem_used_mb()
        stop = threading.Event()
        proc = subprocess.Popen(cmd, cwd=repo)

        def _sampler() -> None:
            while not stop.is_set():
                rss_samples.append(_proc_tree_rss_mb(proc.pid))
                vram_samples.append(_gpu_mem_used_mb())
                stop.wait(bench_cfg["memory"]["sample_interval_s"])

        sampler_thread = threading.Thread(target=_sampler, daemon=True)
        sampler_thread.start()
        ret = proc.wait()
        stop.set()
        sampler_thread.join(timeout=2)
        if ret != 0:
            raise RuntimeError(f"tools/run.py subprocess failed (exit {ret}): {' '.join(cmd)}")

    manifest = json.loads(manifest_path.read_text())
    stages = manifest["stages"]

    def _stage_dir(name: str) -> Path:
        return Path(stages[name]["dir"])

    # `time_s` in detections/hypotheses is CUMULATIVE by design (predictions_from_table needs one
    # "time per image" per BOP19's convention): hypotheses.time_s = segment.time_s + FoundPose's own
    # marginal cost (adapters/foundpose_cli.py:467-468, `time_s=det_time + dt`). Verified against
    # the full A8_k4 1000-image run: coarse_pose.time_s >= segment.time_s for all 1000 images.
    # coarse_pose's OWN marginal cost is therefore the per-image difference, not the raw column.
    seg_df = read_detections_table(_stage_dir("segment"))
    seg_per_image_s = seg_df.groupby(["scene_id", "image_id"])["time_s"].first()

    pose_df = read_hypotheses_table(_stage_dir("coarse_pose"))
    pose_cum_per_image_s = pose_df.groupby(["scene_id", "image_id"])["time_s"].first()
    pose_marginal = (pose_cum_per_image_s - seg_per_image_s).dropna()

    refine_details_path = _stage_dir("refine") / "refine_details.parquet"
    refine_per_image: list[float] = []
    if refine_details_path.exists():
        rdf = pd.read_parquet(refine_details_path)
        refine_per_image = rdf.groupby(["scene_id", "image_id"])["seconds"].sum().tolist()

    peak_rss = max(rss_samples) if rss_samples else manifest.get("peak_rss_mb", 0.0)
    peak_vram = max(0.0, max(vram_samples) - gpu_baseline) if vram_samples else 0.0

    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "row": row,
        "n_scenes": n_scenes,
        "n_images_per_scene": n_images_per_scene,
        "n_images_total": len(subset),
        "seed": seed,
        "subset_targets_file": str(subset_path),
        "run_manifest": str(manifest_path),
        "wall_seconds": manifest.get("wall_seconds"),
        "peak_rss_mb": round(peak_rss, 1),
        "peak_vram_mb": round(peak_vram, 1),
        "per_image_ms": {
            "segment": pct(seg_per_image_s.tolist()),
            "coarse_pose_marginal": pct(pose_marginal.tolist()),
            "refine": pct(refine_per_image),
        },
        "stage_total_seconds": {name: info["seconds"] for name, info in stages.items()},
        "note": (
            "per_image_ms.segment and .refine are genuinely per-image (real time_s / seconds "
            "columns in their own artefacts, grouped by scene_id, image_id). "
            "per_image_ms.coarse_pose_marginal is FoundPose's OWN cost with segment's cumulative "
            "time subtracted out (hypotheses.time_s = segment.time_s + FoundPose's dt, by design, "
            "for BOP19's one-time-per-image convention -- see predictions_from_table). "
            "associate/fuse/confidence/evaluate operate on multi-view groups rather than single "
            "images, so only their stage_total_seconds is reported (P18/P19)."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--row", default=None)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--bench-cfg", type=Path, default=REPO / "configs" / "benchmark.yaml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run the full-pipeline leg (D13's second budget) instead of the "
        "update-path leg: one tools/run.py subprocess on a small scene subset.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="--full-pipeline only: re-run even if the row's manifest exists.",
    )
    args = ap.parse_args()

    bench_cfg = yaml.safe_load(args.bench_cfg.read_text())

    # Apply thread env from protocol before loading anything.
    for k, v in bench_cfg["protocol"].get("thread_env", {}).items():
        os.environ[k] = str(v)

    if args.full_pipeline:
        row = args.row or "full_pipeline_subset"
        result = run_full_pipeline(bench_cfg, REPO, args.dataset, row, force=args.force)
        out = args.out or (args.outputs / f"{args.dataset}_full_pipeline_{row}.json")
    else:
        row = args.row or "A8_k4"
        manifest = json.loads(
            (args.outputs / "runs" / f"{row}_{args.dataset}" / "run_manifest.json").read_text()
        )
        result = run_update_path(
            manifest,
            bench_cfg,
            outputs=args.outputs,
            dataset_name=args.dataset,
            row=row,
        )
        out = args.out or (args.outputs / f"{args.dataset}_benchmark_{row}.json")

    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
