#!/usr/bin/env python
"""Profile the update path (D13: ``refine -> associate -> fuse -> confidence`` for one ObjectTrack,
given cached Detections and coarse hypotheses) in Python, from a finished A8 row.

    uv run python tools/profile_update_path.py --dataset tless --row A8_k4 --n-groups 60

Every view group of the row is a Scene update: its member hypotheses are refined one by one
(depth and masks are loaded outside the timer — they are the cached inputs), the group's
hypotheses are associated, every track is fused and scored. Per-stage times are attributed per
track (a group's association / scoring time divided by its track count) and reported as
median / p90 / p95 against the 200 ms target. The first ``--profile-groups`` timed groups run
their refine calls under cProfile for the hot-spot list (the candidates for the C++ ports, D6)
and are excluded from the timings. Run with ``OMP_NUM_THREADS=1`` for the D13 batch-1 protocol
and on a quiet machine: the numbers are wall-clock.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.confidence import ConfidenceModel  # noqa: E402
from binposert.multiview import FusionParams, associate, fuse_track, lift_to_world  # noqa: E402
from binposert.pipeline.artefacts import (  # noqa: E402
    HypothesisRecord,
    hypothesis_from_row,
    hypothesis_to_row,
    load_mask,
    read_detections_table,
    read_hypotheses_table,
)
from binposert.pipeline.confidence_stage import (  # noqa: E402
    aggregate_members,
    params_from_config,
    score_members,
    with_diameter,
)
from binposert.pipeline.multiview_stage import (  # noqa: E402
    GROUPS_FILE,
    associate_params_from_config,
    scene_weights,
)
from binposert.pipeline.refine_stage import params_from_config as refine_params  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.refine import Refiner  # noqa: E402
from binposert.types import ObjectTrack  # noqa: E402

STAGES = ["refine", "associate", "fuse", "confidence"]


def pct(x: list[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float) * 1000.0
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--row", default="A8_k4")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--n-groups", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5, help="groups timed but discarded")
    ap.add_argument(
        "--profile-groups",
        type=int,
        default=10,
        help="groups (after warm-up) whose refine calls run under cProfile; the timings of "
        "the other groups are taken without the profiler's overhead",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    manifest = json.loads(
        (args.outputs / "runs" / f"{args.row}_{args.dataset}" / "run_manifest.json").read_text()
    )
    cfg, st = manifest["config"], manifest["stages"]
    dataset = make_dataset(cfg, REPO)
    seg_dir, pose_dir = Path(st["segment"]["dir"]), Path(st["coarse_pose"]["dir"])
    assoc_dir = Path(st["associate"]["dir"])
    groups = json.loads((assoc_dir / GROUPS_FILE).read_text())
    all_groups = [(int(s), g) for s, gs in groups.items() for g in gs]
    load0 = os.getloadavg()
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(
        len(all_groups), size=min(args.n_groups + args.warmup, len(all_groups)), replace=False
    )
    chosen = [all_groups[i] for i in pick]

    r_params = refine_params(cfg["refiner"])
    a_params = associate_params_from_config(cfg["multiview"], REPO)
    c_params = params_from_config(cfg["confidence"], REPO)
    model_h, model_f = (
        ConfidenceModel.load(c_params.model_h),
        ConfidenceModel.load(c_params.model_f),
    )
    dets = read_detections_table(seg_dir)
    coarse = read_hypotheses_table(pose_dir)
    refiners: dict[int, Refiner] = {}
    times: dict[str, list[float]] = {s: [] for s in STAGES}
    per_track_total: list[float] = []
    prof = cProfile.Profile()
    n_tracks_total = 0
    for gi, (sid, image_ids) in enumerate(chosen):
        warm = gi < args.warmup
        profiled = not warm and gi < args.warmup + args.profile_groups
        # cached inputs: views with depth, masks, coarse hypotheses
        views, masks, hyps_rows = {}, {}, []
        for iid in image_ids:
            v, _ = dataset.load_view(sid, iid, load_rgb=False, load_depth=True)
            views[iid] = v
            d_img = dets[(dets.scene_id == sid) & (dets.image_id == iid)].set_index("detection_id")
            h_img = coarse[(coarse.scene_id == sid) & (coarse.image_id == iid)]
            for _, row in h_img.iterrows():
                masks[(iid, int(row.detection_id))] = load_mask(
                    seg_dir, str(d_img.loc[int(row.detection_id), "mask_path"])
                )
                hyps_rows.append(row)
        # refine, per hypothesis
        refined_rows = []
        t_refine_group = 0.0
        for row in hyps_rows:
            hyp = hypothesis_from_row(row)
            if hyp.object_id not in refiners:
                refiners[hyp.object_id] = Refiner(dataset.load_model(hyp.object_id), r_params)
            iid = int(row.image_id)
            t0 = time.perf_counter()
            if profiled:
                prof.enable()
            out = refiners[hyp.object_id].refine(views[iid], masks[(iid, hyp.detection_id)], hyp)
            if profiled:
                prof.disable()
            dt = time.perf_counter() - t0
            t_refine_group += dt
            if not warm and not profiled:
                times["refine"].append(dt)
            r = dict(row)
            r.update(hypothesis_to_row(HypothesisRecord(sid, iid, out.hypothesis, dt)))
            refined_rows.append(r)
        refined = pd.DataFrame(refined_rows)
        models = {int(o): dataset.load_model(int(o)) for o in refined.object_id.unique()}
        # associate, per group
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
        # fuse, per track
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
            if not warm and not profiled:
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
        # confidence, per group (batched), attributed per track
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
        if not warm and not profiled:
            times["associate"].extend([t_assoc / n_tracks] * n_tracks)
            times["confidence"].extend([t_conf / n_tracks] * n_tracks)
            per_track_total.extend(
                [(t_refine_group + t_assoc + t_fuse_group + t_conf) / n_tracks] * n_tracks
            )
            n_tracks_total += n_tracks
        print(
            f"group {gi + 1}/{len(chosen)} scene {sid} views {image_ids}: {len(hyps_rows)} hyps, "
            f"{n_tracks} tracks, refine {t_refine_group:.2f} s, assoc {t_assoc * 1000:.0f} ms, "
            f"fuse {t_fuse_group * 1000:.0f} ms, conf {t_conf * 1000:.0f} ms"
            + (" (warm-up)" if warm else " (profiled)" if profiled else "")
        )
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
    result = {
        "dataset": args.dataset,
        "row": args.row,
        "n_groups": len(chosen) - args.warmup - args.profile_groups,
        "n_profiled_groups": args.profile_groups,
        "n_tracks": n_tracks_total,
        "single_threaded": os.environ.get("OMP_NUM_THREADS") == "1",
        "load_average_start": load0,
        "per_stage_ms": {k: pct(v) for k, v in times.items()},
        "per_track_total_ms": pct(per_track_total),
        "per_hypothesis_refine_ms": pct(times["refine"]),
        "target_p95_ms": 200.0,
        "profile_refine_top": s.getvalue(),
    }
    out = args.out or (args.outputs / f"{args.dataset}_update_path_profile.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "profile_refine_top"}, indent=1))
    print(s.getvalue()[:4000])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
