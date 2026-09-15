#!/usr/bin/env python
"""MegaPose adapter: Detections artefacts -> coarse PoseHypotheses artefacts (D3 comparator, A5).

Runs inside docker/megapose (or envs/megapose). Uses the upstream ``PoseEstimator`` pipeline
(coarse render-and-compare + refiner, ``megapose-1.0-RGB-multi-hypothesis`` by default) with the
Detection bounding boxes as input, one image at a time. The output pose is MegaPose's final refined
pose but is tagged ``stage=coarse`` in our vocabulary: the project's own Refinement stage (Beta)
runs after it, and the comparison of A5 with A1 is estimator-vs-estimator.

    python adapters/megapose_cli.py --dataset-root data/bop/tless --split test_primesense \
        --targets data/bop/tless/test_targets_bop19.json --detections <segment_dir> \
        --out <stage_dir> --params '{"model": "megapose-1.0-RGB-multi-hypothesis"}'
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters.common import (  # noqa: E402
    AdapterArgs,
    depth_path,
    parse_args,
    rgb_path,
    target_images,
    target_objects,
    write_adapter_info,
)

log = logging.getLogger("adapters.megapose")

MODELS_URL = "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/"


def bootstrap(args: AdapterArgs) -> None:
    src = args.upstream / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    data_dir = args.checkpoints / "megapose"
    os.environ.setdefault("MEGAPOSE_DATA_DIR", str(data_dir))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # megapose.config assumes a conda env; any prefix with bin/python satisfies it.
    os.environ.setdefault("CONDA_PREFIX", str(Path(sys.executable).parents[1]))
    models = data_dir / "megapose-models"
    if not models.is_dir() or not any(models.iterdir()):
        raise SystemExit(
            f"MegaPose checkpoints missing under {models}. Download them once with\n"
            f"  wget -r -np -nH --cut-dirs=2 -R 'index.html*' -P {data_dir} {MODELS_URL}\n"
            "(only <run>/checkpoint.pth.tar and config.yaml are needed, ~85 MB per RGB model)."
        )


def make_object_dataset(args: AdapterArgs, lids: list[int]) -> Any:
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset

    models_dir = args.dataset_root / "models_cad"
    if not models_dir.is_dir():
        models_dir = args.dataset_root / "models"
    with open(models_dir / "models_info.json") as f:
        info = json.load(f)
    objects = [
        RigidObject(
            label=f"obj_{lid:06d}",
            mesh_path=models_dir / f"obj_{lid:06d}.ply",
            mesh_units="mm",
            mesh_diameter=float(info[str(lid)]["diameter"]) / 1000.0,
        )
        for lid in lids
    ]
    return RigidObjectDataset(objects)


def run(args: AdapterArgs) -> None:
    import pandas as pd
    import torch

    # The panda3d render workers are forked and hand tensors back through torch queues; with the
    # default "file_descriptor" strategy share_memory_() deadlocks in the child (upstream's own
    # training script sets this too).
    torch.multiprocessing.set_sharing_strategy("file_system")
    bootstrap(args)
    from megapose.inference.types import ObservationTensor, PandasTensorCollection
    from megapose.utils.load_model import NAMED_MODELS, load_named_model

    from binposert.pipeline.artefacts import (
        HypothesisRecord,
        read_detections_table,
        write_hypotheses,
    )
    from binposert.types import PoseHypothesis, QualitySignals, Stage

    model_name = str(args.params.get("model", "megapose-1.0-RGB-multi-hypothesis"))
    info = NAMED_MODELS[model_name]
    infer_kw = dict(info["inference_parameters"])
    infer_kw["n_refiner_iterations"] = int(
        args.params.get("n_refiner_iterations", infer_kw["n_refiner_iterations"])
    )
    infer_kw["n_pose_hypotheses"] = int(
        args.params.get("n_pose_hypotheses", infer_kw["n_pose_hypotheses"])
    )

    assert args.detections is not None
    dets = read_detections_table(args.detections)
    pairs = target_images(args)
    pair_set = set(pairs)
    dets = dets[
        [(int(s), int(i)) in pair_set for s, i in zip(dets.scene_id, dets.image_id, strict=True)]
    ]
    tobj = target_objects(args)
    lids = sorted(int(o) for o in dets.object_id.unique())
    log.info(
        "%d target images, %d detections, %d objects, model %s",
        len(pairs),
        len(dets),
        len(lids),
        model_name,
    )

    object_dataset = make_object_dataset(args, lids)
    estimator = load_named_model(model_name, object_dataset).cuda()

    recs: list[HypothesisRecord] = []
    stats = {"n_instances": 0, "n_poses": 0, "n_images": len(pairs)}
    t_start = time.time()
    for idx, (scene_id, image_id) in enumerate(pairs):
        group = dets[(dets.scene_id == scene_id) & (dets.image_id == image_id)]
        if tobj is not None:
            keep = []
            for lid, sub in group.groupby("object_id"):
                n_target = tobj.get((scene_id, image_id), {}).get(int(lid), 0)
                if n_target > 0:
                    keep.append(sub.sort_values("score", ascending=False).head(n_target))
            group = pd.concat(keep) if keep else group.iloc[0:0]
        if len(group) == 0:
            continue
        with open(args.dataset_root / args.split / f"{scene_id:06d}" / "scene_camera.json") as f:
            cam = json.load(f)[str(image_id)]
        K = np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3)
        import cv2

        bgr = cv2.imread(str(rgb_path(args, scene_id, image_id)), cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        depth = None
        if info["requires_depth"]:
            dp = depth_path(args, scene_id, image_id)
            if dp is None:
                raise RuntimeError(f"{model_name} needs depth but scene {scene_id} has none")
            depth = (
                cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32)
                * float(cam.get("depth_scale", 1.0))
                / 1000.0
            )
        observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()
        boxes = np.stack(
            [
                [r.bbox_x, r.bbox_y, r.bbox_x + r.bbox_w, r.bbox_y + r.bbox_h]
                for r in group.itertuples()
            ]
        ).astype(np.float32)
        infos = pd.DataFrame(
            dict(
                label=[f"obj_{int(o):06d}" for o in group.object_id],
                batch_im_id=0,
                instance_id=np.arange(len(group)),
            )
        )
        detections = PandasTensorCollection(infos=infos, bboxes=torch.as_tensor(boxes)).cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            output, extra = estimator.run_inference_pipeline(
                observation, detections=detections, **infer_kw
            )
        dt = time.perf_counter() - t0
        stats["n_instances"] += len(group)
        poses = output.poses.cpu().numpy()  # (N, 4, 4) T_camera_object in metres
        out_infos = output.infos
        scores = (
            out_infos["pose_score"].to_numpy()
            if "pose_score" in out_infos
            else np.full(len(poses), np.nan)
        )
        logits = (
            out_infos["coarse_logit"].to_numpy()
            if "coarse_logit" in out_infos
            else np.full(len(poses), np.nan)
        )
        rows = list(group.itertuples())
        det_time = float(np.nanmax(group["time_s"].to_numpy(dtype=float))) if len(group) else 0.0
        det_time = det_time if np.isfinite(det_time) else 0.0
        for k in range(len(poses)):
            inst = int(out_infos["instance_id"].iloc[k])
            row = rows[inst]
            T = np.asarray(poses[k], dtype=np.float64)
            T[:3, 3] *= 1000.0  # m -> mm
            sig = QualitySignals(seg_score=float(row.score), pose_score=float(scores[k]))
            hyp = PoseHypothesis(
                camera_id=str(row.camera_id),
                object_id=int(row.object_id),
                detection_id=int(row.detection_id),
                hypothesis_id=0,
                T_camera_object=T,
                stage=Stage.COARSE,
                signals=sig,
                source=f"megapose:{model_name}",
            )
            recs.append(HypothesisRecord(scene_id, image_id, hyp, time_s=det_time + dt))
            stats["n_poses"] += 1
        if idx % 20 == 0:
            el = time.time() - t_start
            log.info(
                "[%d/%d] scene %d image %d: %d poses "
                "(%.1fs elapsed, %.2fs/img, coarse_logit mean %.2f)",
                idx + 1,
                len(pairs),
                scene_id,
                image_id,
                len(poses),
                el,
                el / (idx + 1),
                float(np.nanmean(logits)) if len(logits) else float("nan"),
            )
    write_hypotheses(args.out, recs)
    write_adapter_info(
        args,
        "megapose",
        {**stats, "model": model_name, "inference": infer_kw, "seconds": time.time() - t_start},
    )
    log.info("wrote %d hypotheses to %s", len(recs), args.out)
    _stop_renderers(estimator)


def _stop_renderers(estimator: Any) -> None:
    """Send the panda3d workers their stop token; otherwise Python's atexit joins them forever."""
    seen: set[int] = set()
    for model in (estimator.coarse_model, estimator.refiner_model):
        renderer = getattr(model, "renderer", None)
        if renderer is None or id(renderer) in seen or not hasattr(renderer, "stop"):
            continue
        seen.add(id(renderer))
        try:
            renderer.stop()
        except Exception as e:  # noqa: BLE001 — shutdown only
            log.warning("renderer stop failed: %s", e)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args("megapose", __doc__ or "", needs_detections=True)
    run(args)


if __name__ == "__main__":
    main()
