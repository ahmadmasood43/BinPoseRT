#!/usr/bin/env python
"""CNOS adapter: BOP images -> Detections artefacts (D4, D12, D15).

Runs inside docker/cnos (or envs/cnos): SAM ViT-H proposals, DINOv2 ViT-L/14 CLS descriptors,
template matching against pyrender templates of the CAD models, per-object NMS. The upstream
model code (``src.model.detector.CNOS.test_step``) is used unchanged; this file only prepares the
directory layout CNOS expects, renders the templates once, feeds the target images and converts
the per-image ``.npz`` outputs into ``detections.parquet`` + mask PNGs.

    python adapters/cnos_cli.py --dataset-root data/bop/tless --split test_primesense \
        --targets data/bop/tless/test_targets_bop19.json --out <stage_dir> --params '{...}'
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from adapters.common import (  # noqa: E402
    AdapterArgs,
    ensure_symlink,
    parse_args,
    rgb_path,
    target_images,
    write_adapter_info,
)

log = logging.getLogger("adapters.cnos")

DINOV2_REPO_URL = "https://github.com/facebookresearch/dinov2.git"
DINOV2_REPO_COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"  # same as FoundPose's submodule
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DINOV2_CHECKPOINT = "dinov2_vitl14_pretrain.pth"
TEMPLATE_RENDER_LEVEL = 2  # CNOS renders level 2 and subsamples the requested level from it


def prepare_layout(args: AdapterArgs, dataset_name: str) -> Path:
    """CNOS expects <root>/datasets/<dataset>/{models_cad,...}, <root>/pretrained/segment-anything
    and <root>/datasets/templates_pyrender/<dataset>. Build that from symlinks under --work."""
    root = args.work
    ds_link = root / "datasets" / dataset_name
    ds_link.mkdir(parents=True, exist_ok=True)
    for entry in sorted(args.dataset_root.iterdir()):
        if entry.name.startswith(".unpacked"):
            continue
        ensure_symlink(ds_link / entry.name, entry)
    ensure_symlink(
        root / "pretrained" / "segment-anything" / SAM_CHECKPOINT, args.checkpoints / SAM_CHECKPOINT
    )
    torch_home = args.checkpoints / "torch"
    ensure_symlink(
        torch_home / "hub" / "checkpoints" / DINOV2_CHECKPOINT, args.checkpoints / DINOV2_CHECKPOINT
    )
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    return root


def ensure_dinov2_checkout(args: AdapterArgs) -> Path:
    """torch.hub cannot resolve a commit SHA as a GitHub ref, so pin DINOv2 via a local checkout."""
    d = args.upstream.parent / "dinov2"
    if not (d / "hubconf.py").exists():
        subprocess.run(["git", "clone", "-q", DINOV2_REPO_URL, str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", DINOV2_REPO_COMMIT], check=True)
    return d


def render_templates(args: AdapterArgs, dataset_name: str) -> Path:
    tdir = args.work / "datasets" / "templates_pyrender" / dataset_name
    marker = tdir / "_RENDERED"
    if marker.exists():
        return tdir
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.render_template_with_pyrender",
        f"dataset_name={dataset_name}",
        f"user.local_root_dir={args.work}",
        f"level={TEMPLATE_RENDER_LEVEL}",
        "num_workers=4",
        'gpus="0"',  # a bare 0 is parsed as int and rejected by os.environ
        "disable_output=true",
    ]
    log.info("rendering templates: %s", " ".join(cmd))
    # The upstream script spawns `python -m src.poses.pyrender` per object via os.system.
    env = {
        **os.environ,
        "PYOPENGL_PLATFORM": "egl",
        "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
    }
    subprocess.run(cmd, cwd=args.upstream, env=env, check=True)
    counts = {
        p.name: len(list(p.glob("*.png")))
        for p in tdir.iterdir()
        if p.is_dir() and p.name.startswith("obj_")
    }
    if not counts or min(counts.values()) == 0:
        raise RuntimeError(f"template rendering produced empty object directories under {tdir}")
    marker.write_text(
        f"level={TEMPLATE_RENDER_LEVEL} objects={len(counts)} images={sum(counts.values())}\n"
    )
    return tdir


def build_model(args: AdapterArgs, template_dir: Path, dataset_name: str):  # type: ignore[no-untyped-def]
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    p = args.params
    overrides = [
        f"dataset_name={dataset_name}",
        f"user.local_root_dir={args.work}",
        "model.onboarding_config.rendering_type=pyrender",
        f"model.onboarding_config.level_templates={int(p.get('level_templates', 0))}",
        f"model.matching_config.aggregation_function={p.get('aggregation', 'avg_5')}",
        f"model.matching_config.confidence_thresh={float(p.get('confidence_thresh', 0.15))}",
        f"model.post_processing_config.nms_thresh={float(p.get('nms_thresh', 0.25))}",
        f"model.descriptor_model.model.repo_or_dir={ensure_dinov2_checkout(args)}",
        "+model.descriptor_model.model.source=local",
        f"save_dir={args.work / 'results'}",
        "name_exp=binposert",
    ]
    if p.get("segmentor", "sam") == "fastsam":
        overrides.append("model=cnos_fast")
    else:
        thresh = float(p.get("stability_score_thresh", 0.97))
        overrides.append(f"model.segmentor_model.stability_score_thresh={thresh}")
    with initialize_config_dir(version_base=None, config_dir=str(args.upstream / "configs")):
        cfg = compose(config_name="run_inference", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    log.info("CNOS config: %s", OmegaConf.to_yaml(cfg.model.onboarding_config))

    ref_cfg = cfg.data.reference_dataloader.copy()
    ref_cfg.template_dir = str(template_dir)
    ref_dataset = instantiate(ref_cfg)

    model = instantiate(cfg.model)
    model.ref_dataset = ref_dataset
    model.ref_obj_names = cfg.data.datasets[dataset_name].obj_names
    model.dataset_name = dataset_name
    seg_name = cfg.model.segmentor_model._target_.split(".")[-1]
    model.name_prediction_file = (
        f"{seg_name}_template_pyrender{cfg.model.onboarding_config.level_templates}"
        f"_agg{cfg.model.matching_config.aggregation_function}_{dataset_name}"
    )
    model = model.to(torch.device(args.device))
    model.eval()
    return model, cfg


def run(args: AdapterArgs) -> None:
    import torch
    import torchvision.transforms as T
    from PIL import Image

    sys.path.insert(0, str(args.upstream))
    os.chdir(args.upstream)  # CNOS resolves predefined poses relative to its project root

    dataset_name = args.dataset_root.name
    prepare_layout(args, dataset_name)
    template_dir = render_templates(args, dataset_name)
    model, cfg = build_model(args, template_dir, dataset_name)

    from binposert.pipeline.artefacts import DetectionRecord, DetectionWriter, camera_id_for
    from binposert.types import Detection

    min_pixels = int(args.params.get("min_mask_pixels", 50))
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )
    pred_dir = Path(model.log_dir) / "predictions" / dataset_name / model.name_prediction_file
    writer = DetectionWriter(args.out)
    pairs = target_images(args)
    log.info("%d target images", len(pairs))
    t_start = time.time()
    n_dets = 0
    with torch.no_grad():
        for idx, (scene_id, image_id) in enumerate(pairs):
            npz = pred_dir / f"scene{scene_id}_frame{image_id}.npz"
            if idx == 0 or not npz.exists():  # idx 0 also loads the reference descriptors
                image = Image.open(rgb_path(args, scene_id, image_id)).convert("RGB")
                batch = {
                    "image": transform(image).unsqueeze(0).to(model.device),
                    "scene_id": [str(scene_id)],
                    "frame_id": [str(image_id)],
                }
                model.test_step(batch, idx)  # per-image npz: a restarted run resumes from here
            data = np.load(npz)
            scores = np.asarray(data["score"], dtype=np.float64)
            order = np.argsort(-scores, kind="stable")
            runtime = float(data["time"])
            det_id = 0
            for k in order:
                mask = np.asarray(data["segmentation"][k]).astype(bool)
                if mask.sum() < min_pixels:
                    continue
                writer.add(
                    DetectionRecord(
                        scene_id,
                        image_id,
                        Detection(
                            camera_id=camera_id_for(image_id),
                            object_id=int(data["category_id"][k]),
                            mask=mask,
                            score=float(scores[k]),
                            detection_id=det_id,
                        ),
                        time_s=runtime,
                    )
                )
                det_id += 1
            n_dets += det_id
            if idx % 20 == 0:
                el = time.time() - t_start
                log.info(
                    "[%d/%d] scene %d image %d: %d detections (%.1fs elapsed, %.2fs/img)",
                    idx + 1,
                    len(pairs),
                    scene_id,
                    image_id,
                    det_id,
                    el,
                    el / (idx + 1),
                )
    writer.close()
    write_adapter_info(
        args,
        "cnos",
        {
            "n_images": len(pairs),
            "n_detections": n_dets,
            "seconds": time.time() - t_start,
            "dinov2_repo_commit": DINOV2_REPO_COMMIT,
            "template_dir": str(template_dir),
            "prediction_name": model.name_prediction_file,
        },
    )
    log.info("wrote %d detections for %d images to %s", n_dets, len(pairs), args.out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args("cnos", __doc__ or "")
    run(args)


if __name__ == "__main__":
    main()
