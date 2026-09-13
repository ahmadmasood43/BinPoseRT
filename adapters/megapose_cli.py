#!/usr/bin/env python
"""MegaPose adapter: fills a ``coarse_pose`` stage directory with PoseHypotheses (D3, A5).

Runs ``megapose-1.0-RGB-multi-hypothesis`` (coarse classifier → ``n_pose_hypotheses`` refined by
MegaPose's own render-and-compare refiner → scored) on every Detection of the upstream artefact and
caches *all* scored hypotheses per Detection as ``stage=coarse`` (MegaPose's refiner is part of the
comparator; our depth Refinement in Beta is what is under test). Depth is never passed in.

Environment: ``MEGAPOSE_DATA_DIR`` (checkpoints under ``megapose-models/``), see
docker/megapose/Dockerfile. Object meshes are the BOP ``models`` (mm) of the request's dataset.

Usage::

    python adapters/megapose_cli.py run --request <stage dir>/adapter_request.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters._common import (  # noqa: E402
    Request,
    clean_output_dir,
    rgb_path,
    scene_camera,
    setup_logging,
)
from binposert import artefacts  # noqa: E402

log = logging.getLogger("adapter.megapose")


def make_object_dataset(models_dir: Path, object_ids: list[int]) -> Any:
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset

    objects = [
        RigidObject(
            label=f"obj_{o:06d}", mesh_path=models_dir / f"obj_{o:06d}.ply", mesh_units="mm"
        )
        for o in object_ids
    ]
    return RigidObjectDataset(objects)


def run(req: Request) -> int:
    import torch
    from megapose.datasets.scene_dataset import ObjectData
    from megapose.inference.types import ObservationTensor
    from megapose.inference.utils import make_detections_from_object_data
    from megapose.utils.load_model import NAMED_MODELS, load_named_model
    from PIL import Image

    det_dir = req.inputs["detections"]
    det_table = artefacts.read_detections_table(det_dir)
    object_ids = sorted(int(o) for o in det_table.object_id.unique())
    models_dir_name = str(req.dataset.get("models_dir") or "models")
    models_dir = req.dataset_root / models_dir_name
    if not models_dir.is_dir():
        models_dir = req.dataset_root / "models"
    object_dataset = make_object_dataset(models_dir, object_ids)

    model_name = str(req.config["checkpoints"]["model"])
    opts = dict(req.config.get("opts", {}))
    params = dict(NAMED_MODELS[model_name]["inference_parameters"])
    params["n_pose_hypotheses"] = int(
        opts.get("n_pose_hypotheses", params.get("n_pose_hypotheses", 5))
    )
    params["n_refiner_iterations"] = int(
        opts.get("n_refiner_iterations", params.get("n_refiner_iterations", 5))
    )
    log.info("loading %s (%s)", model_name, params)
    estimator = load_named_model(
        model_name, object_dataset, bsz_images=int(opts.get("bsz_images", 64))
    ).cuda()

    writer = artefacts.PoseHypothesesWriter(req.output_dir, source="megapose")
    n = 0
    by_image = {
        (int(s), int(i)): g.sort_values("detection_id")
        for (s, i), g in det_table.groupby(["scene_id", "image_id"], sort=True)
    }
    for scene_id, image_id in req.image_keys:
        rows = by_image.get((scene_id, image_id))
        if rows is None or len(rows) == 0:
            continue
        objs = req.object_ids(scene_id, image_id)
        if objs is not None:
            rows = rows[rows.object_id.isin(objs)]
        cam = scene_camera(req.dataset_root, req.split, scene_id)[str(image_id)]
        K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        rgb = np.asarray(
            Image.open(rgb_path(req.dataset_root, req.split, scene_id, image_id)).convert("RGB")
        )
        object_data = [
            ObjectData(
                label=f"obj_{int(r.object_id):06d}",
                bbox_modal=np.array(
                    [r.bbox_x, r.bbox_y, r.bbox_x + r.bbox_w, r.bbox_y + r.bbox_h], dtype=np.float64
                ),
            )
            for r in rows.itertuples(index=False)
        ]
        det_ids = [int(r.detection_id) for r in rows.itertuples(index=False)]
        seg_scores = [float(r.score) for r in rows.itertuples(index=False)]
        detections = make_detections_from_object_data(object_data).cuda()
        observation = ObservationTensor.from_numpy(rgb, None, K).cuda()

        t0 = time.perf_counter()
        with torch.no_grad():
            _, extra = estimator.run_inference_pipeline(
                observation, detections=detections, **params
            )
        torch.cuda.synchronize()
        t_img = time.perf_counter() - t0

        scored = extra["scoring"]["preds"]  # every refined hypothesis with pose_score / pose_logit
        infos = scored.infos
        poses = scored.poses.cpu().numpy()  # T_camera_object in metres
        for inst in sorted(infos["instance_id"].unique()):
            sel = infos.index[infos["instance_id"] == inst].tolist()
            sel.sort(key=lambda k: -float(infos.loc[k, "pose_score"]))
            for h, k in enumerate(sel):
                T = np.array(poses[k], dtype=np.float64)
                T[:3, 3] *= 1000.0
                writer.add_pose(
                    scene_id=scene_id,
                    image_id=image_id,
                    object_id=int(rows.iloc[int(inst)].object_id),
                    detection_id=det_ids[int(inst)],
                    hypothesis_id=h,
                    T_camera_object=T,
                    time_s=t_img,
                    pose_score=float(infos.loc[k, "pose_score"]),
                    seg_score=seg_scores[int(inst)],
                )
                n += 1
        log.info("scene %d image %d: %d detections in %.2f s", scene_id, image_id, len(rows), t_img)
    writer.finish(
        stage="coarse_pose",
        config=req.config,
        hash=req.hash,
        n_images=len(req.image_keys),
        inputs={k: str(v) for k, v in req.inputs.items()},
        upstream=req.config.get("upstream"),
        model=model_name,
        inference_parameters=params,
    )
    return n


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--request", required=True, type=Path)
    args = ap.parse_args(argv)
    req = Request.load(args.request)
    if req.stage != "coarse_pose" or req.config.get("name") != "megapose":
        raise SystemExit(
            f"request is for {req.stage}/{req.config.get('name')}, not coarse_pose/megapose"
        )
    clean_output_dir(req)
    n = run(req)
    artefacts.mark_success(req.output_dir)
    log.info("wrote %d hypotheses → %s", n, req.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
