"""Adapters (adapters/*_cli.py) driven end-to-end on the mini fixture without a GPU:
the CNOS importer on a hand-built BOP detection JSON and the FoundPose collector on a fake
``estimated-poses.json``. The upstream-run branches are exercised on the GPU machine only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from binposert import artefacts
from binposert.pipeline import MissingGpuArtefact, Pipeline
from binposert.pipeline.stages import ADAPTER_REQUEST_FILE
from binposert.segment import CachedSegmenter
from tests.conftest import compose_config

REPO = Path(__file__).resolve().parents[1]


def _adapter(name: str, *args: str, env: dict[str, str] | None = None) -> str:
    cmd = [sys.executable, str(REPO / "adapters" / name), *args]
    out = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=300,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def _bop_detections_from_gt(mini_bop, jitter_scores: bool = True) -> list[dict]:
    """The BOP'23 default-detection JSON layout, built from the fixture's visible GT masks."""
    entries = []
    for s in mini_bop.scene_ids:
        for i in mini_bop.image_ids(s):
            _, gts = mini_bop.load_view(s, i, load_rgb=False, load_depth=False)
            for g in gts:
                mask = mini_bop.gt_mask(s, i, g.gt_index)
                ys, xs = np.nonzero(mask)
                entries.append(
                    {
                        "scene_id": s,
                        "image_id": i,
                        "category_id": g.object_id,
                        "bbox": [
                            int(xs.min()),
                            int(ys.min()),
                            int(np.ptp(xs) + 1),
                            int(np.ptp(ys) + 1),
                        ],
                        "score": 0.9 - 0.1 * g.gt_index if jitter_scores else 1.0,
                        "time": 0.35,
                        "segmentation": artefacts.mask_to_rle(mask),
                    }
                )
    return entries


def test_cnos_import_fills_segment_stage(tmp_path, mini_bop):
    cfg = compose_config("experiment=A1", "dataset=mini_bop", f"outputs_root={tmp_path}")
    with pytest.raises(MissingGpuArtefact) as exc:
        Pipeline(cfg).run()
    seg_plan = exc.value.plan
    assert seg_plan.name == "segment" and "cnos_cli.py" in str(exc.value)
    request = seg_plan.directory / ADAPTER_REQUEST_FILE

    json_path = tmp_path / "cnos-fastsam_mini_bop-test.json"
    json_path.write_text(json.dumps(_bop_detections_from_gt(mini_bop)))
    _adapter("cnos_cli.py", "import", "--request", str(request), "--json", str(json_path))

    assert seg_plan.is_complete
    seg = CachedSegmenter(seg_plan.directory)
    assert seg.name == "cnos-fastsam"
    view, gts = mini_bop.load_view(1, 0, load_rgb=False, load_depth=False)
    dets = seg.segment(view)
    assert len(dets) == len(gts)
    # detection ids follow descending score; masks survive the RLE round trip exactly
    assert [d.score for d in dets] == sorted([d.score for d in dets], reverse=True)
    for d in dets:
        g = next(g for g in gts if abs(0.9 - 0.1 * g.gt_index - d.score) < 1e-9)
        np.testing.assert_array_equal(d.mask, mini_bop.gt_mask(1, 0, g.gt_index))
    table = artefacts.read_detections_table(seg_plan.directory)
    np.testing.assert_allclose(table["time_s"], 0.35)
    assert artefacts.read_meta(seg_plan.directory)["upstream"]["commit"].startswith("298d1f33")

    # the next GPU stage (FoundPose) is now the one that stops the run
    with pytest.raises(MissingGpuArtefact) as exc2:
        Pipeline(cfg).run()
    assert exc2.value.plan.name == "coarse_pose"


def test_foundpose_collect_maps_inst_ids_back_to_detections(tmp_path, mini_bop, monkeypatch):
    """Simulate FoundPose's per-object estimated-poses.json (GT poses, inst_id = rank by score)
    and let the adapter's ``collect`` build the artefact; evaluation must then give AR 1."""
    cfg = compose_config("experiment=A0", "dataset=mini_bop", f"outputs_root={tmp_path}")
    with pytest.raises(MissingGpuArtefact) as exc:
        Pipeline(cfg).run()
    plan = exc.value.plan
    request_path = plan.directory / ADAPTER_REQUEST_FILE
    det_dir = plan.inputs["detections"]
    det = artefacts.read_detections_table(det_dir)

    # what export_detections() would have recorded: per (scene, image, object) the detection ids
    # in FoundPose's order (score descending, then detection_id)
    index: dict[str, list[int]] = {}
    for (s, i, o), g in det.groupby(["scene_id", "image_id", "object_id"], sort=True):
        g = g.sort_values(["score", "detection_id"], ascending=[False, True], kind="stable")
        index[f"{s}/{i}/{o}"] = [int(d) for d in g.detection_id]
    (plan.directory / "detection_index.json").write_text(json.dumps(index))

    out_root = tmp_path / "foundpose_out"
    inference = out_root / "inference" / f"mini_bop_bp-{plan.hash}"
    per_object: dict[int, list[dict]] = {}
    for key, det_ids in index.items():
        s, i, o = (int(x) for x in key.split("/"))
        _, gts = mini_bop.load_view(s, i, load_rgb=False, load_depth=False)
        for inst_id, det_id in enumerate(det_ids):
            T = gts[det_id].T_camera_object
            per_object.setdefault(o, []).append(
                {
                    "scene_id": str(s),
                    "img_id": str(i),
                    "obj_id": str(o),
                    "inst_id": str(inst_id),
                    "hypothesis_id": "0",
                    "score": str(0.5 + 0.01 * det_id),
                    "R": T[:3, :3].tolist(),
                    "t": T[:3, 3].reshape(3, 1).tolist(),
                    "time": {"corresp": 0.1, "pose_coarse": 0.2},
                    "cnos_time": 0.3,
                }
            )
    for o, rows in per_object.items():
        d = inference / str(o)
        d.mkdir(parents=True)
        (d / "estimated-poses.json").write_text(json.dumps(rows))

    _adapter(
        "foundpose_cli.py",
        "collect",
        "--request",
        str(request_path),
        env={"FOUNDPOSE_OUTPUT_PATH": str(out_root), "BOP_PATH": str(tmp_path)},
    )
    assert plan.is_complete
    table = artefacts.read_pose_hypotheses_table(plan.directory)
    assert len(table) == len(det)
    assert (table["source"] == "foundpose").all() and (table["stage"] == "coarse").all()
    np.testing.assert_allclose(table["time_s"], 0.6)
    assert (table["seg_score"] == 1.0).all()  # joined back from the GT detections
    merged = table.merge(det, on=["scene_id", "image_id", "detection_id"], suffixes=("", "_det"))
    assert (merged["object_id"] == merged["object_id_det"]).all()

    result = Pipeline(cfg).run()
    assert result.report is not None and result.report["ar"] == pytest.approx(1.0)
