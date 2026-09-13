"""Stage DAG + content-addressed cache (D12): hashes, _SUCCESS skipping, GPU handshake, run.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from binposert import artefacts
from binposert.evaluate import read_bop_csv
from binposert.pipeline import MissingGpuArtefact, Pipeline, stage_hash
from binposert.pipeline.stages import ADAPTER_REQUEST_FILE
from binposert.types import QualitySignals
from tests.conftest import compose_config

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- hashing


def test_stage_hash_is_stable_and_sensitive():
    ds = {"name": "tless", "split": "test", "targets": "t.json", "scene_ids": None}
    cfg = {"name": "gt", "impl": "local", "min_visible_fraction": 0.0, "root": "/a/path"}
    h = stage_hash("segment", "1", cfg, ds, {})
    assert len(h) == 16
    assert stage_hash("segment", "1", dict(cfg), dict(ds), {}) == h
    # path-like keys never enter the hash; ints and integral floats are the same value
    assert stage_hash("segment", "1", {**cfg, "root": "/elsewhere"}, ds, {}) == h
    assert stage_hash("segment", "1", {**cfg, "min_visible_fraction": 0}, ds, {}) == h
    # anything else does
    assert stage_hash("segment", "2", cfg, ds, {}) != h
    assert stage_hash("segment", "1", {**cfg, "min_visible_fraction": 0.1}, ds, {}) != h
    assert stage_hash("segment", "1", cfg, {**ds, "scene_ids": [1]}, {}) != h
    assert stage_hash("segment", "1", cfg, ds, {"segment": "abc"}) != h
    assert stage_hash("coarse_pose", "1", cfg, ds, {}) != h


def test_plan_hashes_chain_through_inputs(tmp_path):
    cfg = compose_config("experiment=smoke", f"outputs_root={tmp_path}")
    p = Pipeline(cfg)
    names = [s.name for s in p.plan]
    assert names == ["segment", "coarse_pose", "evaluate"]
    seg, coarse, ev = p.plan
    assert coarse.input_hashes == {"segment": seg.hash}
    assert ev.input_hashes == {"coarse_pose": coarse.hash}
    assert coarse.inputs == {"detections": seg.directory}
    assert seg.directory == tmp_path / "mini_bop" / "test" / "segment" / seg.hash

    # a change upstream re-keys every downstream stage; a change downstream keys only itself
    cfg2 = compose_config("experiment=smoke", f"outputs_root={tmp_path}", "segmenter.version=2")
    p2 = Pipeline(cfg2)
    assert [s.hash != t.hash for s, t in zip(p.plan, p2.plan, strict=True)] == [True, True, True]
    cfg3 = compose_config("experiment=smoke", f"outputs_root={tmp_path}", "evaluate.with_vsd=false")
    p3 = Pipeline(cfg3)
    assert [s.hash != t.hash for s, t in zip(p.plan, p3.plan, strict=True)] == [False, False, True]


# ----------------------------------------------------------------------------- execution


def test_smoke_run_is_deterministic_and_cached(tmp_path):
    cfg = compose_config("experiment=smoke", f"outputs_root={tmp_path}")
    r1 = Pipeline(cfg).run(experiment="smoke")
    assert r1.completed and r1.status == {s: "ran" for s in ("segment", "coarse_pose", "evaluate")}
    assert r1.report is not None and 0.0 < r1.report["ar"] < 1.0
    assert r1.report["n_gt"] == 12 and r1.report["n_predictions"] == 12
    assert (
        r1.results_csv is not None and r1.results_csv.name == "smoke-gt-synthetic_mini_bop-test.csv"
    )
    assert len(read_bop_csv(r1.results_csv)) == 12
    manifest = json.loads(r1.manifest_path.read_text())
    assert manifest["experiment"] == "smoke"
    assert {s["name"]: s["status"] for s in manifest["stages"]} == r1.status
    assert manifest["dataset"]["n_images"] == 4 and manifest["summary"]["n_gt"] == 12
    assert "commit" in manifest["git"] and manifest["peak_rss_mb"] > 0
    ev_dir = next(p for p in r1.stages if p.name == "evaluate").directory
    assert (ev_dir / "gallery_worst_mssd.png").exists()
    assert (ev_dir / "rows.parquet").exists() and (ev_dir / "report.md").exists()
    report_bytes = (ev_dir / "report.json").read_bytes()
    results_bytes = (ev_dir / "results.csv").read_bytes()

    # second run: nothing recomputed, identical artefacts
    r2 = Pipeline(cfg).run(experiment="smoke")
    assert r2.status == {s: "cached" for s in ("segment", "coarse_pose", "evaluate")}
    assert (ev_dir / "report.json").read_bytes() == report_bytes

    # a fresh outputs root recomputes the same numbers bit for bit (D12 determinism)
    cfg_b = compose_config("experiment=smoke", f"outputs_root={tmp_path / 'again'}")
    r3 = Pipeline(cfg_b).run(experiment="smoke")
    ev_dir_b = next(p for p in r3.stages if p.name == "evaluate").directory
    assert ev_dir_b.name == ev_dir.name
    assert (ev_dir_b / "report.json").read_bytes() == report_bytes
    # only the measured wall time may differ between two runs
    strip = lambda b: [ln.rsplit(b",", 1)[0] for ln in b.splitlines()]  # noqa: E731
    assert strip((ev_dir_b / "results.csv").read_bytes()) == strip(results_bytes)


def test_incomplete_stage_directory_is_rerun(tmp_path):
    cfg = compose_config(
        "experiment=smoke", f"outputs_root={tmp_path}", "pipeline.stop_after=segment"
    )
    r = Pipeline(cfg).run()
    assert r.status == {"segment": "ran", "coarse_pose": "skipped", "evaluate": "skipped"}
    seg = r.stages[0]
    (seg.directory / "_SUCCESS").unlink()  # simulate a crash / partial rsync
    (seg.directory / "junk.txt").write_text("stale")
    r = Pipeline(cfg).run()
    assert r.status["segment"] == "ran"
    assert not (seg.directory / "junk.txt").exists() and seg.is_complete


def test_gpu_stage_stops_with_request_and_accepts_adapter_output(tmp_path, mini_bop):
    """A0 on the fixture: segment(gt) runs locally, coarse_pose(foundpose) is a GPU stage.
    Simulate the adapter writing GT poses in the frozen schema; the rerun must evaluate to AR 1."""
    cfg = compose_config("experiment=A0", "dataset=mini_bop", f"outputs_root={tmp_path}")
    with pytest.raises(MissingGpuArtefact) as exc:
        Pipeline(cfg).run(experiment="A0")
    plan = exc.value.plan
    assert plan.name == "coarse_pose" and plan.impl == "gpu"
    assert "foundpose_cli.py" in str(exc.value)
    request = json.loads((plan.directory / ADAPTER_REQUEST_FILE).read_text())
    assert request["hash"] == plan.hash and request["stage"] == "coarse_pose"
    assert request["inputs"]["detections"] == str(plan.inputs["detections"])
    assert sorted(map(tuple, request["image_keys"])) == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert request["config"]["upstream"]["commit"].startswith("3103473b")

    # "adapter": one hypothesis per cached detection, at the GT pose
    det = artefacts.read_detections_table(Path(request["inputs"]["detections"]))
    writer = artefacts.PoseHypothesesWriter(plan.directory, source="foundpose")
    for r in det.to_dict("records"):
        _, gts = mini_bop.load_view(r["scene_id"], r["image_id"], load_rgb=False, load_depth=False)
        gt = gts[r["detection_id"]]
        writer.add_pose(
            r["scene_id"],
            r["image_id"],
            r["object_id"],
            r["detection_id"],
            0,
            gt.T_camera_object,
            time_s=0.2,
            pose_score=0.9,
            seg_score=r["score"],
        )
    writer.finish(stage="coarse_pose")
    artefacts.mark_success(plan.directory)

    result = Pipeline(cfg).run(experiment="A0")
    assert result.status == {"segment": "cached", "coarse_pose": "cached", "evaluate": "ran"}
    assert result.report is not None
    assert result.report["ar"] == pytest.approx(1.0)
    assert result.report["source"] == "foundpose"
    preds = read_bop_csv(result.results_csv)
    assert all(p.time_s == pytest.approx(0.2) for p in preds)


def test_run_py_end_to_end_on_mini_fixture(tmp_path):
    cmd = [
        sys.executable,
        str(REPO / "tools" / "run.py"),
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "evaluate.gallery_n_worst=0",
    ]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "AR = " in out.stdout and "manifest:" in out.stdout
    runs = list((tmp_path / "mini_bop" / "test" / "runs" / "smoke").glob("*/run_manifest.json"))
    assert len(runs) == 1
    csvs = list(runs[0].parent.glob("*.csv"))
    assert csvs and csvs[0].name == "smoke-gt-synthetic_mini_bop-test.csv"

    # a GPU experiment exits 2 with the adapter instructions
    cmd = [sys.executable, str(REPO / "tools" / "run.py"), "experiment=A1", "dataset=mini_bop"]
    out = subprocess.run(
        cmd + [f"outputs_root={tmp_path}"], cwd=REPO, capture_output=True, text=True
    )
    assert out.returncode == 2
    assert "cnos_cli.py" in out.stderr


def test_synthetic_estimator_error_matches_config(tmp_path):
    cfg = compose_config(
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "estimator.sigma_t_mm=0.0",
        "estimator.sigma_rot_deg=0.0",
        "evaluate.gallery_n_worst=0",
    )
    r = Pipeline(cfg).run()
    assert r.report is not None and r.report["ar"] == pytest.approx(1.0)
    table = artefacts.read_pose_hypotheses_table(r.stages[1].directory)
    assert set(table.columns) >= set(QualitySignals.field_names())
    assert np.isnan(table["icp_fitness"]).all()
    assert (table["stage"] == "coarse").all()
