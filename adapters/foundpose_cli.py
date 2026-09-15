#!/usr/bin/env python
"""FoundPose adapter: Detections artefacts -> coarse PoseHypotheses artefacts (D3, D12, D15).

Runs inside docker/foundpose (or envs/foundpose). Templates and object representations are built
with the upstream scripts (``gen_templates``, ``gen_repre``) once per dataset and cached under
--work. Inference re-uses the upstream building blocks (crop camera, DINOv2 feature extraction,
tf-idf template retrieval, cyclic-buddy correspondences, PnP-RANSAC) but *not* ``scripts/infer.py``:
that script consults ground-truth annotations to drop detections and to evaluate, and reads its
detections from a fixed CNOS file. Here every Detection of the upstream ``segment`` stage is
processed identically, whatever produced it (GT masks for A0, CNOS for A1), and no ground truth is
touched.

    python adapters/foundpose_cli.py --dataset-root data/bop/tless --split test_primesense \
        --targets data/bop/tless/test_targets_bop19.json --detections <segment_dir> \
        --out <stage_dir> --params '{...}'
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
    ensure_symlink,
    parse_args,
    rgb_path,
    target_images,
    target_objects,
    write_adapter_info,
)

log = logging.getLogger("adapters.foundpose")

VERSION = "binposert"  # name of the templates / repre version directories under --work
DINOV2_CHECKPOINT = "dinov2_vitl14_pretrain.pth"


# ----------------------------------------------------------------------------- environment


def bootstrap(args: AdapterArgs, dataset_name: str) -> Any:
    """Put FoundPose and its submodules on sys.path and point bop_toolkit at --work."""
    up = args.upstream
    for p in (up, up / "external" / "bop_toolkit", up / "external" / "dinov2", up / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    datasets_dir = args.work / "datasets"
    ds_link = datasets_dir / dataset_name
    ds_link.mkdir(parents=True, exist_ok=True)
    for entry in sorted(args.dataset_root.iterdir()):
        if entry.name.startswith(".unpacked"):
            continue
        ensure_symlink(ds_link / entry.name, entry)
    os.environ["BOP_PATH"] = str(datasets_dir)  # read by bop_toolkit_lib.config at import
    torch_home = args.checkpoints / "torch"
    ensure_symlink(
        torch_home / "hub" / "checkpoints" / DINOV2_CHECKPOINT, args.checkpoints / DINOV2_CHECKPOINT
    )
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import bop_toolkit_lib.config as bop_config

    bop_config.datasets_path = str(datasets_dir)
    bop_config.output_path = str(args.work / "output")
    Path(bop_config.output_path).mkdir(parents=True, exist_ok=True)
    return bop_config


def object_ids_for(args: AdapterArgs, dataset_name: str) -> list[int]:
    with open(args.dataset_root / "models_cad" / "models_info.json") as f:
        all_ids = sorted(int(k) for k in json.load(f))
    tobj = target_objects(args)
    if tobj is None:
        return all_ids
    wanted = {
        o for (s, _), objs in tobj.items() for o in objs if args.scenes is None or s in args.scenes
    }
    return [o for o in all_ids if o in wanted]


# ----------------------------------------------------------------------------- onboarding


def ensure_templates(
    args: AdapterArgs, bop_config: Any, dataset_name: str, lids: list[int]
) -> Path:
    import gen_templates

    tp = args.params.get("templates", {})
    base = Path(bop_config.output_path) / "templates" / VERSION / dataset_name
    missing = [lid for lid in lids if not (base / str(lid) / "_DONE").exists()]
    if missing:
        opts = gen_templates.GenTemplatesOpts(
            version=VERSION,
            object_dataset=dataset_name,
            object_lids=missing,
            num_viewspheres=int(tp.get("num_viewspheres", 1)),
            min_num_viewpoints=int(tp.get("min_num_viewpoints", 57)),
            num_inplane_rotations=int(tp.get("num_inplane_rotations", 14)),
            images_per_view=1,
            max_num_triangles=int(tp.get("max_num_triangles", 20000)),
            back_face_culling=False,
            texture_size=(1024, 1024),
            ssaa_factor=float(tp.get("ssaa_factor", 4.0)),
            background_type="black",
            light_type="multi_directional",
            features_patch_size=14,
            crop=True,
            crop_rel_pad=float(tp.get("crop_rel_pad", 0.2)),
            crop_size=tuple(tp.get("crop_size", [420, 420])),
            save_templates=True,
            overwrite=True,
            debug=False,
        )
        log.info("rendering templates for %d objects: %s", len(missing), missing)
        t0 = time.time()
        gen_templates.synthesize_templates(opts)
        for lid in missing:
            (base / str(lid) / "_DONE").write_text(f"{time.time() - t0:.1f}s\n")
        log.info("templates done in %.1fs", time.time() - t0)
    return base


def ensure_repre(
    args: AdapterArgs, bop_config: Any, dataset_name: str, lids: list[int], extractor: Any
) -> Path:
    import gen_repre
    from utils import repre_util

    rp = args.params.get("repre", {})
    base = Path(bop_config.output_path) / "object_repre"
    for lid in lids:
        repre_dir = Path(
            repre_util.get_object_repre_dir_path(str(base), VERSION, dataset_name, lid)
        )
        if (repre_dir / "repre.pth").exists() and (repre_dir / "_DONE").exists():
            continue
        opts = gen_repre.GenRepreOpts(
            version=VERSION,
            templates_version=VERSION,
            object_dataset=dataset_name,
            object_lids=[lid],
            extractor_name=args.params["extractor"],
            grid_cell_size=float(rp.get("grid_cell_size", 14.0)),
            apply_pca=True,
            pca_components=int(rp.get("pca_components", 256)),
            cluster_features=True,
            cluster_num=int(rp.get("cluster_num", 2048)),
            template_desc_opts=repre_util.TemplateDescOpts(desc_type="tfidf"),
            overwrite=True,
            debug=False,
        )
        log.info("building representation for object %d", lid)
        t0 = time.time()
        gen_repre.generate_repre(opts, dataset_name, lid, args.device, extractor)
        (repre_dir / "_DONE").write_text(f"{time.time() - t0:.1f}s\n")
    return base


# ----------------------------------------------------------------------------- inference


class ObjectContext:
    """Everything FoundPose needs per object, loaded once."""

    def __init__(self, repre_dir: str, device: str, infer_params: dict[str, Any]) -> None:
        import torch
        from utils import knn_util, repre_util

        self.repre = repre_util.load_object_repre(repre_dir=repre_dir, tensor_device=device)
        self.words_index = knn_util.KNN(
            k=self.repre.template_desc_opts.tfidf_knn_k,
            metric=self.repre.template_desc_opts.tfidf_knn_metric,
        )
        self.words_index.fit(self.repre.feat_cluster_centroids)
        self.template_indices = []
        for template_id in range(len(self.repre.template_cameras_cam_from_model)):
            ids = torch.nonzero(self.repre.feat_to_template_ids == template_id).flatten()
            index = knn_util.KNN(k=1, metric="l2")
            index.fit(self.repre.feat_vectors[ids].cpu())
            self.template_indices.append(index)
        self.vertices = self.repre.vertices.cpu().numpy()
        self.infer = infer_params


def estimate_instance(
    ctx: ObjectContext,
    extractor: Any,
    image_np: np.ndarray,
    mask: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    K: np.ndarray,
    device: str,
    grid_cell_size: float,
    crop_size: tuple[int, int],
    crop_rel_pad: float,
) -> dict[str, Any] | None:
    """One Detection -> pose dict (T_camera_object 4x4, signals) or None when PnP fails."""
    import cv2
    import torch
    from utils import corresp_util, feature_util, pnp_util
    from utils import misc as misc_util
    from utils.misc import array_to_tensor, warp_image
    from utils.structs import AlignedBox2f, PinholePlaneCameraModel

    h, w = mask.shape
    camera = PinholePlaneCameraModel(
        width=w, height=h, f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]), T_world_from_eye=np.eye(4)
    )
    x, y, bw, bh = bbox_xywh
    box = AlignedBox2f(left=x, top=y, right=x + bw, bottom=y + bh)
    crop_box = misc_util.calc_crop_box(box=box, make_square=True)
    crop_camera = misc_util.construct_crop_camera(
        box=crop_box,
        camera_model_c2w=camera,
        viewport_size=crop_size,
        viewport_rel_pad=crop_rel_pad,
    )
    interp = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
    image_crop = warp_image(
        src_camera=camera, dst_camera=crop_camera, src_image=image_np, interpolation=interp
    )
    mask_crop = warp_image(
        src_camera=camera,
        dst_camera=crop_camera,
        src_image=mask.astype(np.uint8),
        interpolation=cv2.INTER_NEAREST,
    )
    if mask_crop.sum() == 0:
        return None

    grid = feature_util.generate_grid_points(grid_size=crop_size, cell_size=grid_cell_size).to(
        device
    )
    image_t = array_to_tensor(image_crop).to(torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    feature_map = extractor(image_t)["feature_maps"][0]
    query_points = feature_util.filter_points_by_mask(grid, array_to_tensor(mask_crop).to(device))
    if len(query_points) == 0:
        return None
    query_features = feature_util.sample_feature_map_at_points(
        feature_map_chw=feature_map,
        points=query_points,
        image_size=(image_crop.shape[1], image_crop.shape[0]),
    ).contiguous()
    if query_features.shape[1] != ctx.repre.feat_vectors.shape[1] and len(
        ctx.repre.feat_raw_projectors
    ):
        from utils import projector_util

        query_features = projector_util.project_features(
            feat_vectors=query_features, projectors=ctx.repre.feat_raw_projectors
        ).contiguous()

    corresp = corresp_util.establish_correspondences(
        query_points=query_points,
        query_features=query_features,
        object_repre=ctx.repre,
        template_matching_type="tfidf",
        template_knn_indices=ctx.template_indices,
        feat_matching_type="cyclic_buddies",
        top_n_templates=int(ctx.infer.get("match_top_n_templates", 5)),
        top_k_buddies=int(ctx.infer.get("match_top_k_buddies", 300)),
        visual_words_knn_index=ctx.words_index,
        debug=False,
    )

    best: dict[str, Any] | None = None
    for c in corresp:
        if len(c["coord_2d"]) < 6:
            continue
        ok, R, t, inliers, quality = pnp_util.estimate_pose(
            corresp=c,
            camera_c2w=crop_camera,
            pnp_type="opencv",
            pnp_ransac_iter=int(ctx.infer.get("pnp_ransac_iter", 400)),
            pnp_inlier_thresh=float(ctx.infer.get("pnp_inlier_thresh", 10.0)),
            pnp_required_ransac_conf=0.99,
            pnp_refine_lm=True,
        )
        if not ok:
            continue
        if best is None or quality > best["quality"]:
            best = {
                "R": np.asarray(R, dtype=np.float64).reshape(3, 3),
                "t": np.asarray(t, dtype=np.float64).reshape(3),
                "quality": float(quality),
                "inliers": None if inliers is None else np.asarray(inliers).reshape(-1),
                "corresp": c,
            }
    if best is None:
        return None

    # The PnP pose is expressed in the crop camera; move it to the original camera frame.
    T_crop_object = np.eye(4)
    T_crop_object[:3, :3] = best["R"]
    T_crop_object[:3, 3] = best["t"]
    T_camera_object = crop_camera.T_world_from_eye @ T_crop_object  # world == original camera

    # Many-to-many-aware inlier ratio (upstream's BOP score) and mean inlier reprojection error.
    c = best["corresp"]
    coord_2d = (
        c["coord_2d"].cpu().numpy() if hasattr(c["coord_2d"], "cpu") else np.asarray(c["coord_2d"])
    )
    nn_ids = (
        c["nn_vertex_ids"].cpu().numpy()
        if hasattr(c["nn_vertex_ids"], "cpu")
        else np.asarray(c["nn_vertex_ids"])
    )
    ids_2d = (
        c["coord_2d_ids"].cpu().numpy()
        if hasattr(c["coord_2d_ids"], "cpu")
        else np.asarray(c["coord_2d_ids"])
    )
    verts_crop = (T_crop_object[:3, :3] @ ctx.vertices[nn_ids].T).T + T_crop_object[:3, 3]
    proj = crop_camera.eye_to_window(verts_crop)
    dist = np.linalg.norm(coord_2d - proj, axis=1)
    thresh = float(ctx.infer.get("pnp_inlier_thresh", 10.0))
    inlier = dist <= thresh
    uniq = np.unique(ids_2d)
    hit = np.zeros(len(uniq))
    for k, u in enumerate(uniq):
        hit[k] = float(np.any(inlier[ids_2d == u]))
    score = float(hit.mean()) if len(uniq) else 0.0
    return {
        "T_camera_object": T_camera_object,
        "pose_score": score,
        "n_inliers": float(inlier.sum()),
        "reproj_error_px": float(dist[inlier].mean()) if inlier.any() else float("nan"),
        "n_corresp": int(len(coord_2d)),
        "template_id": int(c["template_id"]),
        "template_score": float(c["template_score"]),
        "quality": best["quality"],
    }


def run(args: AdapterArgs) -> None:
    import torch

    dataset_name = args.dataset_root.name
    bop_config = bootstrap(args, dataset_name)
    from utils import feature_util, repre_util

    from binposert.pipeline.artefacts import (
        HYPOTHESES_FILE,
        HYPOTHESIS_COLUMNS,
        HypothesisRecord,
        hypothesis_to_row,
        load_mask,
        read_detections_table,
    )
    from binposert.types import PoseHypothesis, QualitySignals, Stage

    lids = object_ids_for(args, dataset_name)
    log.info("objects: %s", lids)
    ensure_templates(args, bop_config, dataset_name, lids)
    if args.extra.get("onboard_only"):
        log.info("templates ready for %d objects (--onboard-only)", len(lids))
        return
    extractor = feature_util.make_feature_extractor(args.params["extractor"])
    extractor.to(args.device)
    repre_base = ensure_repre(args, bop_config, dataset_name, lids, extractor)

    infer_params = dict(args.params.get("infer", {}))
    crop_size = tuple(args.params.get("templates", {}).get("crop_size", [420, 420]))
    crop_rel_pad = float(args.params.get("templates", {}).get("crop_rel_pad", 0.2))
    grid_cell_size = float(args.params.get("repre", {}).get("grid_cell_size", 14.0))
    num_preds_factor = float(infer_params.get("num_preds_factor", 1.0))

    assert args.detections is not None
    dets = read_detections_table(args.detections)
    pairs = target_images(args)
    pair_set = set(pairs)
    tobj = target_objects(args)
    dets = dets[
        [(int(s), int(i)) in pair_set for s, i in zip(dets.scene_id, dets.image_id, strict=True)]
    ]
    log.info("%d target images, %d detections", len(pairs), len(dets))

    import cv2
    import pandas as pd

    stats = {"n_instances": 0, "n_poses": 0, "n_no_pose": 0}
    t_start = time.time()
    cams: dict[int, dict[str, Any]] = {}
    partial_dir = args.out / "partial"  # per-object checkpoints so a restart resumes
    partial_dir.mkdir(parents=True, exist_ok=True)
    for lid in lids:
        part = partial_dir / f"obj_{lid:06d}.parquet"
        if part.exists():
            df_part = pd.read_parquet(part)
            log.info("object %d: %d poses restored from %s", lid, len(df_part), part)
            stats["n_poses"] += len(df_part)
            continue
        obj_recs: list[HypothesisRecord] = []
        ctx = ObjectContext(
            repre_util.get_object_repre_dir_path(str(repre_base), VERSION, dataset_name, lid),
            args.device,
            infer_params,
        )
        sel = dets[dets.object_id == lid]
        t_obj = time.time()
        n_obj = 0
        for (scene_id, image_id), group in sel.groupby(["scene_id", "image_id"], sort=True):
            scene_id, image_id = int(scene_id), int(image_id)
            if tobj is not None:
                n_target = tobj.get((scene_id, image_id), {}).get(lid, 0)
                if n_target == 0:
                    continue
                group = group.sort_values("score", ascending=False).head(
                    max(1, int(num_preds_factor * n_target))
                )
            if scene_id not in cams:
                with open(
                    args.dataset_root / args.split / f"{scene_id:06d}" / "scene_camera.json"
                ) as f:
                    cams[scene_id] = json.load(f)
            K = np.asarray(cams[scene_id][str(image_id)]["cam_K"], dtype=np.float64).reshape(3, 3)
            bgr = cv2.imread(str(rgb_path(args, scene_id, image_id)), cv2.IMREAD_COLOR)
            image_np = np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32) / 255.0
            for _, row in group.iterrows():
                mask = load_mask(args.detections, str(row["mask_path"]))
                t0 = time.perf_counter()
                with torch.no_grad():
                    res = estimate_instance(
                        ctx,
                        extractor,
                        image_np,
                        mask,
                        (int(row.bbox_x), int(row.bbox_y), int(row.bbox_w), int(row.bbox_h)),
                        K,
                        args.device,
                        grid_cell_size,
                        crop_size,
                        crop_rel_pad,
                    )
                dt = time.perf_counter() - t0
                stats["n_instances"] += 1
                n_obj += 1
                if res is None:
                    stats["n_no_pose"] += 1
                    continue
                stats["n_poses"] += 1
                hyp = PoseHypothesis(
                    camera_id=str(row["camera_id"]),
                    object_id=lid,
                    detection_id=int(row["detection_id"]),
                    hypothesis_id=0,
                    T_camera_object=res["T_camera_object"],
                    stage=Stage.COARSE,
                    signals=QualitySignals(
                        seg_score=float(row["score"]),
                        pose_score=res["pose_score"],
                        reproj_error_px=res["reproj_error_px"],
                        n_inliers=res["n_inliers"],
                    ),
                    source="foundpose",
                )
                det_time = float(row["time_s"]) if np.isfinite(float(row["time_s"])) else 0.0
                obj_recs.append(HypothesisRecord(scene_id, image_id, hyp, time_s=det_time + dt))
        pd.DataFrame(
            [hypothesis_to_row(r) for r in obj_recs], columns=HYPOTHESIS_COLUMNS
        ).to_parquet(part, index=False)
        log.info(
            "object %d: %d instances in %.1fs (%.2fs each); total %d poses so far",
            lid,
            n_obj,
            time.time() - t_obj,
            (time.time() - t_obj) / max(n_obj, 1),
            stats["n_poses"],
        )
        del ctx
        torch.cuda.empty_cache()

    table = pd.concat(
        [pd.read_parquet(partial_dir / f"obj_{lid:06d}.parquet") for lid in lids],
        ignore_index=True,
    )
    table["rejection_reason"] = table["rejection_reason"].astype(object)
    table.to_parquet(args.out / HYPOTHESES_FILE, index=False)
    write_adapter_info(
        args,
        "foundpose",
        {**stats, "n_images": len(pairs), "seconds": time.time() - t_start, "version": VERSION},
    )
    log.info("wrote %d hypotheses to %s (%s)", len(table), args.out, stats)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args("foundpose", __doc__ or "", needs_detections=True)
    run(args)


if __name__ == "__main__":
    main()
