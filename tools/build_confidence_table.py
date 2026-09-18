#!/usr/bin/env python
"""Build the labelled tables the ConfidenceModels are fitted and evaluated on (Delta, D11), from
the cached Gamma rows of one dataset and nothing else.

    uv run python tools/build_confidence_table.py --dataset tless
    uv run python tools/build_confidence_table.py --dataset xyzibd --rows A6_k1_mean A6_k2_mean

Writes ``outputs/confidence/<dataset>/``:

* ``hypotheses.parquet`` — every refined PoseHypothesis of the dataset (the refine stage the rows
  share) with its label: ``success = MSSD < 0.1 · diameter`` against the nearest same-object
  ground truth of its View, plus the Model H extras;
* ``fused.parquet`` — every FusedPose of every listed row (one row per view count) with its
  world-frame label and the row it came from (``row``, ``k``);
* ``members.parquet`` — the (row, track) -> hypothesis links, so Model H probabilities can be
  aggregated per track when Model F is fitted;
* ``build.json`` — provenance (stage hashes, counts, base rates).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.confidence import label_fused, label_hypotheses  # noqa: E402
from binposert.pipeline.artefacts import read_hypotheses_table  # noqa: E402
from binposert.pipeline.multiview_stage import FUSED_FILE, TRACKS_FILE  # noqa: E402
from binposert.pipeline.run import make_dataset  # noqa: E402
from binposert.pipeline.stages import default_workers  # noqa: E402

MEMBER_COLUMNS = [
    "row",
    "k",
    "scene_id",
    "group_id",
    "track_id",
    "image_id",
    "object_id",
    "detection_id",
    "hypothesis_id",
    "weight",
    "rejection_reason",
]


def manifest_of(outputs: Path, name: str, dataset: str) -> dict[str, Any]:
    p = outputs / "runs" / f"{name}_{dataset}" / "run_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"row {name} on {dataset} has no run manifest at {p}")
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="tless")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument(
        "--rows",
        nargs="+",
        default=["A6_k1_mean", "A6_k2_mean", "A6_k3_mean", "A6_k4_mean"],
        help="fused rows (mean fusion, unperturbed extrinsics) whose FusedPoses are labelled",
    )
    ap.add_argument("--n-workers", type=int, default=0, help="0 = half the cores")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n_workers = args.n_workers or default_workers()
    out = args.out or (args.outputs / "confidence" / args.dataset)
    out.mkdir(parents=True, exist_ok=True)

    manifests = {name: manifest_of(args.outputs, name, args.dataset) for name in args.rows}
    first = manifests[args.rows[0]]
    dataset = make_dataset(first["config"], REPO)
    refine_hashes = {m["stages"]["refine"]["hash"] for m in manifests.values()}
    if len(refine_hashes) != 1:
        raise SystemExit(f"rows do not share one refine stage: {refine_hashes}")
    refine_dir = Path(first["stages"]["refine"]["dir"])

    t0 = time.perf_counter()
    hyps = read_hypotheses_table(refine_dir)
    hyps_lab = label_hypotheses(hyps, dataset, n_model_points=0, n_workers=n_workers)
    hyps_lab.insert(0, "dataset", args.dataset)
    hyps_lab.to_parquet(out / "hypotheses.parquet", index=False)
    t_h = time.perf_counter() - t0
    print(
        f"hypotheses: {len(hyps_lab)} rows, success rate {hyps_lab.success.mean():.3f}, "
        f"no GT {int((~hyps_lab.gt_present).sum())}, {t_h:.0f} s"
    )

    fused_parts: list[pd.DataFrame] = []
    member_parts: list[pd.DataFrame] = []
    rows_info: dict[str, Any] = {}
    for name, m in manifests.items():
        k = int(m["config"]["multiview"]["params"]["groups"]["n_views"])
        noise = m["config"]["multiview"]["params"].get("extrinsic_noise", {})
        if float(noise.get("t_mm", 0)) or float(noise.get("deg", 0)):
            raise SystemExit(f"{name} has perturbed extrinsics; its FusedPoses cannot be labelled")
        fuse_dir = Path(m["stages"]["fuse"]["dir"])
        assoc_dir = Path(m["stages"]["associate"]["dir"])
        fused = pd.read_parquet(fuse_dir / FUSED_FILE)
        if len(fused) == 0:
            raise SystemExit(f"{name} wrote no FusedPoses (fusion=none?)")
        t1 = time.perf_counter()
        lab = label_fused(fused, dataset, n_model_points=0, n_workers=n_workers)
        lab.insert(0, "k", k)
        lab.insert(0, "row", name)
        lab.insert(0, "dataset", args.dataset)
        fused_parts.append(lab)
        tracks = pd.read_parquet(assoc_dir / TRACKS_FILE)
        mem = tracks[[c for c in MEMBER_COLUMNS if c in tracks.columns]].copy()
        mem.insert(0, "k", k)
        mem.insert(0, "row", name)
        member_parts.append(mem[MEMBER_COLUMNS])
        rows_info[name] = {
            "k": k,
            "fuse_hash": m["stages"]["fuse"]["hash"],
            "associate_hash": m["stages"]["associate"]["hash"],
            "n_fused": int(len(lab)),
            "success_rate": float(lab.success.mean()),
            "seconds": time.perf_counter() - t1,
        }
        print(
            f"{name}: {len(lab)} FusedPoses, success rate {lab.success.mean():.3f}, "
            f"{rows_info[name]['seconds']:.0f} s"
        )
    fused_all = pd.concat(fused_parts, ignore_index=True)
    for c in ("joint_reason",):
        if c in fused_all:
            fused_all[c] = fused_all[c].astype(object)
    fused_all.to_parquet(out / "fused.parquet", index=False)
    members = pd.concat(member_parts, ignore_index=True)
    members["rejection_reason"] = members["rejection_reason"].astype(object)
    members.to_parquet(out / "members.parquet", index=False)

    info = {
        "dataset": args.dataset,
        "split": dataset.split,
        "refine_hash": first["stages"]["refine"]["hash"],
        "refine_dir": str(refine_dir),
        "n_hypotheses": int(len(hyps_lab)),
        "hypothesis_success_rate": float(hyps_lab.success.mean()),
        "n_hypotheses_without_gt": int((~hyps_lab.gt_present).sum()),
        "scenes": sorted(int(s) for s in hyps_lab.scene_id.unique()),
        "rows": rows_info,
        "n_fused": int(len(fused_all)),
        "seconds": time.perf_counter() - t0,
    }
    (out / "build.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out} ({info['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
