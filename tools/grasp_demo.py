#!/usr/bin/env python
"""Simulated pick (D14): the transform chain ``T_robot_gripper = T_robot_world @ T_world_object
@ T_object_gripper`` on the FusedPoses of one scene of an A9 (or A8) row, drawn with Open3D.

    uv run python tools/grasp_demo.py --dataset xyzibd --author   # seeds models/grasps/xyzibd.json
    uv run python tools/grasp_demo.py --dataset xyzibd --row A9_nbv_b4_g0 --scene 0

The Grasp Poses live in ``models/grasps/<dataset>.json`` (one ``T_object_gripper`` per
ObjectModel; ``--author`` seeds the file from the bounding boxes, after which it is edited by
hand). ``T_robot_world`` has no dataset value: the robot base is placed by ``--robot-base``
(world mm). Writes ``<out>/pick_scene<id>.png`` and ``pick_scene<id>.json`` (every pick with its
chain) and prints the chain of the most confident accepted picks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.pipeline.artefacts import columns_to_transform  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.transforms import invert, make_T  # noqa: E402
from binposert.types import FusedPose, QualitySignals, Verdict  # noqa: E402
from binposert.viz.grasp import (  # noqa: E402
    author_bbox_grasp,
    chain_text,
    load_grasps,
    plan_picks,
    render_pick,
    save_grasps,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--row", default="A9_nbv_b4_g0")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--grasps", default=None, help="default models/grasps/<dataset>.json")
    ap.add_argument("--author", action="store_true", help="seed the grasp file from the bboxes")
    ap.add_argument("--force", action="store_true", help="overwrite an existing grasp file")
    ap.add_argument("--robot-base", type=float, nargs=3, default=(-600.0, 0.0, 0.0))
    ap.add_argument("--out", default=None, help="default outputs/<dataset>_epsilon_pick")
    ap.add_argument("--n-gripper", type=int, default=3)
    args = ap.parse_args()
    outputs = Path(args.outputs)
    manifest = json.loads(
        (outputs / "runs" / f"{args.row}_{args.dataset}" / "run_manifest.json").read_text()
    )
    dataset = make_dataset(manifest["config"], REPO)
    grasp_path = Path(args.grasps or REPO / "models" / "grasps" / f"{args.dataset}.json")
    if args.author:
        if grasp_path.exists() and not args.force:
            print(f"{grasp_path} exists (use --force to overwrite)")
        else:
            grasps = {oid: author_bbox_grasp(dataset.load_model(oid)) for oid in dataset.object_ids}
            save_grasps(grasp_path, grasps)
            print(f"wrote {len(grasps)} Grasp Poses to {grasp_path}")
    grasps = load_grasps(grasp_path)

    stage = manifest["stages"].get("nbv") or manifest["stages"]["confidence"]
    d = Path(stage["dir"])
    fused_df = pd.read_parquet(d / "fused.parquet")
    fused_df = fused_df[fused_df.scene_id == args.scene]
    if "nbv" in manifest["stages"]:
        episodes = json.loads((d / "episodes.json").read_text())
        used = next(e["views_used"] for e in episodes if e["scene_id"] == args.scene)
    else:
        fused_df = fused_df[fused_df.group_id == 0]
        used = [int(i) for i in str(fused_df.image_ids.iloc[0]).split(",")]
    fused = [
        FusedPose(
            track_id=int(r.track_id),
            object_id=int(r.object_id),
            T_world_object=columns_to_transform(r),
            confidence=float(r.confidence),
            verdict=Verdict(str(r.verdict)),
            signals=QualitySignals.from_row(
                {k: r[k] for k in QualitySignals.field_names() if k in fused_df}
            ),
        )
        for _, r in fused_df.iterrows()
    ]
    models = {oid: dataset.load_model(oid) for oid in sorted({fp.object_id for fp in fused})}
    views = [dataset.load_view(args.scene, i, load_rgb=False, load_depth=False)[0] for i in used]
    T_world_robot = make_T(np.eye(3), np.asarray(args.robot_base, dtype=float))
    T_robot_world = invert(T_world_robot)
    picks = plan_picks(fused, grasps, T_robot_world)
    out = Path(args.out or outputs / f"{args.dataset}_epsilon_pick")
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"pick_scene{args.scene}.png"
    render_pick(models, fused, picks, views, grasps, png, T_robot_world, n_gripper=args.n_gripper)
    accepted = [p for p in picks if p.verdict == Verdict.ACCEPT]
    for p in accepted[: args.n_gripper]:
        print(chain_text(p, grasps[p.object_id], T_robot_world))
        print()
    with open(out / f"pick_scene{args.scene}.json", "w") as f:
        json.dump(
            {
                "row": args.row,
                "scene_id": args.scene,
                "views_used": used,
                "T_robot_world": T_robot_world.ravel().tolist(),
                "n_fused": len(fused),
                "n_accepted": len(accepted),
                "picks": [
                    {
                        "track_id": p.track_id,
                        "object_id": p.object_id,
                        "confidence": p.confidence,
                        "verdict": p.verdict.value,
                        "T_world_object": p.T_world_object.ravel().tolist(),
                        "T_robot_gripper": p.T_robot_gripper.ravel().tolist(),
                    }
                    for p in picks
                ],
            },
            f,
            indent=1,
        )
    print(f"{len(fused)} FusedPoses, {len(accepted)} accepted picks; figure {png}")


if __name__ == "__main__":
    main()
