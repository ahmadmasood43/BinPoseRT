#!/usr/bin/env python
"""FoundPose adapter: fills a ``coarse_pose`` stage directory with a PoseHypotheses artefact (D3).

The public FoundPose release is the *coarse* pipeline (template retrieval + DINOv2 patch
correspondences + PnP-RANSAC), which is exactly the ``stage=coarse`` input Beta's refinement
consumes. The upstream scripts are driven unchanged; this adapter only

1. writes our upstream Detections (GT masks for A0, CNOS for A1) in the CNOS JSON format FoundPose
   reads from ``$BOP_PATH/detections/cnos-fastsam/cnos-fastsam_<dataset>-test.json``,
2. generates templates + object representations once per dataset (gen_templates / gen_repre),
3. runs ``scripts/infer.py`` with an opts JSON built from ``estimator.opts`` in the request,
4. converts ``inference/<dataset>_bp-<hash>/<obj>/estimated-poses.json`` into the frozen Parquet
   schema, mapping FoundPose's ``inst_id`` (index into detections sorted by score) back to our
   ``detection_id``.

FoundPose iterates ``<dataset>/test_targets_bop19.json``; when the dataset has none, one is
generated from the request. Environment: ``BOP_PATH`` (dataset parent dir), ``FOUNDPOSE_REPO``,
``FOUNDPOSE_OUTPUT_PATH`` (see docker/foundpose/Dockerfile).

Usage::

    python adapters/foundpose_cli.py run --request <stage dir>/adapter_request.json
    python adapters/foundpose_cli.py collect --request …   # step 4 only, after a manual infer run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters._common import (  # noqa: E402
    Request,
    clean_output_dir,
    read_json,
    run_command,
    setup_logging,
    write_json,
)
from binposert import artefacts  # noqa: E402

log = logging.getLogger("adapter.foundpose")

REPO = Path(os.environ.get("FOUNDPOSE_REPO", "/opt/foundpose"))
OUTPUT_PATH = Path(os.environ.get("FOUNDPOSE_OUTPUT_PATH", "/scratch/foundpose"))
DETECTION_INDEX_FILE = "detection_index.json"
REPRE_VERSION = "v1"


def version_tag(req: Request) -> str:
    return f"bp-{req.hash}"


# ----------------------------------------------------------------------------- step 1: detections


def export_detections(req: Request, det_table: pd.DataFrame, det_dir: Path) -> dict[str, list[int]]:
    """Write CNOS-format JSON for FoundPose; return ``{"scene/im/obj": [detection_id, ...]}`` in the
    order FoundPose will index them (score descending, stable)."""
    bop_path = Path(os.environ["BOP_PATH"])
    out = bop_path / "detections" / "cnos-fastsam" / f"cnos-fastsam_{req.dataset_name}-test.json"
    if out.exists() and not out.with_suffix(".json.orig").exists():
        shutil.copy(out, out.with_suffix(".json.orig"))  # keep the official file around
    entries: list[dict[str, Any]] = []
    index: dict[str, list[int]] = defaultdict(list)
    keys = set(req.image_keys)
    df = det_table[
        [
            (int(s), int(i)) in keys
            for s, i in zip(det_table.scene_id, det_table.image_id)  # noqa: B905 (py3.9)
        ]
    ]
    for (s, i, o), g in df.groupby(["scene_id", "image_id", "object_id"], sort=True):
        g = g.sort_values(["score", "detection_id"], ascending=[False, True], kind="stable")
        for r in g.itertuples(index=False):
            mask = artefacts.read_mask(det_dir, str(r.mask_path))
            entries.append(
                {
                    "scene_id": int(s),
                    "image_id": int(i),
                    "category_id": int(o),
                    "bbox": [int(r.bbox_x), int(r.bbox_y), int(r.bbox_w), int(r.bbox_h)],
                    "score": float(r.score),
                    "time": float(r.time_s) if not np.isnan(r.time_s) else 0.0,
                    "segmentation": artefacts.mask_to_rle(mask),
                    "binposert_detection_id": int(r.detection_id),  # ignored upstream
                }
            )
            index[f"{int(s)}/{int(i)}/{int(o)}"].append(int(r.detection_id))
    write_json(out, entries)
    log.info("wrote %d detections → %s", len(entries), out)
    return dict(index)


def ensure_targets_file(req: Request, det_table: pd.DataFrame) -> None:
    """FoundPose only visits (scene, image, object) triples listed in test_targets_bop19.json."""
    path = req.dataset_root / "test_targets_bop19.json"
    if path.exists():
        return
    targets = []
    if req.objects is not None:
        for (s, i), objs in sorted(req.objects.items()):
            for o, n in sorted(objs.items()):
                targets.append({"im_id": i, "inst_count": n, "obj_id": o, "scene_id": s})
    else:
        counts = det_table.groupby(["scene_id", "image_id", "object_id"]).size()
        for (s, i, o), n in counts.items():
            targets.append(
                {"im_id": int(i), "inst_count": int(n), "obj_id": int(o), "scene_id": int(s)}
            )
    write_json(path, targets)
    log.warning("generated %s from the request (%d targets)", path, len(targets))


# ----------------------------------------------------------------------------- steps 2–3: upstream


def ensure_representation(req: Request, object_ids: list[int]) -> None:
    tpl = req.config.get("templates", {})
    repre_dir = OUTPUT_PATH / "object_repre" / REPRE_VERSION / req.dataset_name
    if repre_dir.exists() and all((repre_dir / str(o)).exists() for o in object_ids):
        log.info("object representations present under %s", repre_dir)
        return
    gen_templates = {
        "gen_templates_opts": {
            "version": REPRE_VERSION,
            "object_dataset": req.dataset_name,
            "object_lids": object_ids,
            "num_viewspheres": int(tpl.get("num_viewspheres", 1)),
            "min_num_viewpoints": int(tpl.get("min_num_viewpoints", 57)),
            "num_inplane_rotations": int(tpl.get("num_inplane_rotations", 14)),
            "images_per_view": 1,
            "max_num_triangles": 20000,
            "back_face_culling": False,
            "texture_size": [1024, 1024],
            "ssaa_factor": 4.0,
            "background_type": "black",
            "light_type": "multi_directional",
            "features_patch_size": 14,
            "crop": True,
            "crop_rel_pad": float(req.config["opts"].get("crop_rel_pad", 0.2)),
            "crop_size": list(req.config["opts"].get("crop_size", [420, 420])),
        }
    }
    gen_repre = {
        "gen_repre_opts": {
            "version": REPRE_VERSION,
            "templates_version": REPRE_VERSION,
            "object_dataset": req.dataset_name,
            "object_lids": object_ids,
            "extractor_name": str(req.config["opts"]["extractor_name"]),
            "grid_cell_size": float(req.config["opts"].get("grid_cell_size", 14.0)),
            "apply_pca": True,
            "pca_components": int(tpl.get("pca_components", 256)),
            "cluster_features": True,
            "cluster_num": int(tpl.get("cluster_num", 2048)),
            "template_desc_opts": {"desc_type": "tfidf"},
        }
    }
    cfg_dir = req.output_dir / "foundpose_configs"
    write_json(cfg_dir / "gen_templates.json", gen_templates)
    write_json(cfg_dir / "gen_repre.json", gen_repre)
    run_command(
        [
            sys.executable,
            "scripts/gen_templates.py",
            "--opts-path",
            str(cfg_dir / "gen_templates.json"),
        ],
        cwd=REPO,
    )
    run_command(
        [sys.executable, "scripts/gen_repre.py", "--opts-path", str(cfg_dir / "gen_repre.json")],
        cwd=REPO,
    )


def run_infer(req: Request, object_ids: list[int]) -> float:
    opts = dict(req.config.get("opts", {}))
    infer = {
        "infer_opts": {
            "version": version_tag(req),
            "repre_version": REPRE_VERSION,
            "object_dataset": req.dataset_name,
            "object_lids": object_ids,
            "use_detections": True,
            "save_estimates": True,
            "vis_results": False,
            "vis_feat_map": False,
            "vis_for_paper": False,
            "debug": False,
            **opts,
        }
    }
    cfg_dir = req.output_dir / "foundpose_configs"
    write_json(cfg_dir / "infer.json", infer)
    return run_command(
        [sys.executable, "scripts/infer.py", "--opts-path", str(cfg_dir / "infer.json")], cwd=REPO
    )


# ----------------------------------------------------------------------------- step 4: collect


def collect(req: Request, det_table: pd.DataFrame, index: dict[str, list[int]]) -> int:
    inference_dir = OUTPUT_PATH / "inference" / f"{req.dataset_name}_{version_tag(req)}"
    if not inference_dir.exists():
        raise FileNotFoundError(f"no FoundPose output under {inference_dir}")
    seg_score = {
        (int(r.scene_id), int(r.image_id), int(r.detection_id)): float(r.score)
        for r in det_table.itertuples(index=False)
    }
    writer = artefacts.PoseHypothesesWriter(req.output_dir, source="foundpose")
    n = 0
    unmapped = 0
    for obj_dir in sorted(p for p in inference_dir.iterdir() if p.is_dir()):
        results_path = obj_dir / "estimated-poses.json"
        if not results_path.exists():
            log.warning("%s has no estimated-poses.json", obj_dir)
            continue
        for e in read_json(results_path):
            s, i, o = int(e["scene_id"]), int(e["img_id"]), int(e["obj_id"])
            inst = int(e["inst_id"])
            ids = index.get(f"{s}/{i}/{o}", [])
            if inst >= len(ids):
                unmapped += 1
                continue
            det_id = ids[inst]
            R = np.asarray(e["R"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(e["t"], dtype=np.float64).reshape(3)
            if np.linalg.norm(t) < 10.0:  # metres slipped through: BOP poses are in mm
                t = t * 1000.0
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            times = e.get("time", {})
            t_img = float(sum(times.values()) if isinstance(times, dict) else times)
            t_img += float(e.get("cnos_time", 0.0) or 0.0)
            writer.add_pose(
                scene_id=s,
                image_id=i,
                object_id=o,
                detection_id=det_id,
                hypothesis_id=int(e.get("hypothesis_id", 0)),
                T_camera_object=T,
                time_s=t_img,
                pose_score=float(e["score"]),
                seg_score=seg_score.get((s, i, det_id), float("nan")),
            )
            n += 1
    if unmapped:
        log.warning("%d FoundPose estimates could not be mapped to a detection", unmapped)
    writer.finish(
        stage="coarse_pose",
        config=req.config,
        hash=req.hash,
        n_images=len(req.image_keys),
        inputs={k: str(v) for k, v in req.inputs.items()},
        upstream=req.config.get("upstream"),
        foundpose_inference_dir=str(inference_dir),
    )
    return n


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--request", required=True, type=Path)
    args = ap.parse_args(argv)

    req = Request.load(args.request)
    if req.stage != "coarse_pose" or req.config.get("name") != "foundpose":
        raise SystemExit(
            f"request is for {req.stage}/{req.config.get('name')}, not coarse_pose/foundpose"
        )
    det_dir = req.inputs["detections"]
    det_table = artefacts.read_detections_table(det_dir)
    object_ids = sorted(int(o) for o in det_table.object_id.unique())

    if args.cmd == "run":
        clean_output_dir(req)
        index = export_detections(req, det_table, det_dir)
        write_json(req.output_dir / DETECTION_INDEX_FILE, index)
        ensure_targets_file(req, det_table)
        ensure_representation(req, object_ids)
        seconds = run_infer(req, object_ids)
        log.info("infer.py took %.1f s", seconds)
    else:
        index = read_json(req.output_dir / DETECTION_INDEX_FILE)
    n = collect(req, det_table, index)
    artefacts.mark_success(req.output_dir)
    log.info("wrote %d hypotheses → %s", n, req.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
