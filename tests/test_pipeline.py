import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from binposert import transforms as tf
from binposert.data import BopDataset
from binposert.pipeline import (
    DetectionRecord,
    DetectionWriter,
    HypothesisRecord,
    StageCache,
    read_detections,
    read_hypotheses,
    stage_hash,
    write_hypotheses,
)
from binposert.pipeline.artefacts import (
    HYPOTHESES_FILE,
    hypothesis_from_row,
    hypothesis_to_row,
    read_detections_table,
    read_hypotheses_table,
)
from binposert.pipeline.run import make_context, planned_refs, run
from binposert.pipeline.stages import ExternalStageMissing
from binposert.pose import CachedPoseEstimator
from binposert.segment import CachedSegmenter, GroundTruthSegmenter
from binposert.types import Detection, PoseHypothesis, QualitySignals, Stage

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name="config", overrides=list(overrides))


# ----------------------------------------------------------------------------- cache


def test_stage_hash_is_stable_and_sensitive():
    cfg = {"name": "gt", "params": {"min_visible_fraction": 0.1}, "version": 1}
    h1 = stage_hash("segment", "1", cfg, [])
    h2 = stage_hash("segment", "1", json.loads(json.dumps(cfg)), [])
    assert h1 == h2 and len(h1) == 16
    assert stage_hash("segment", "1", {**cfg, "params": {"min_visible_fraction": 0.2}}, []) != h1
    assert stage_hash("segment", "2", cfg, []) != h1  # version bump
    assert stage_hash("segment", "1", cfg, ["abc"]) != h1  # upstream change
    # key order and tuple/list spelling do not matter
    assert stage_hash("s", "1", {"b": (1, 2), "a": 1}, []) == stage_hash(
        "s", "1", {"a": 1, "b": [1, 2]}, []
    )


def test_stage_is_skipped_when_success_marker_exists(tmp_path):
    cache = StageCache(tmp_path, "ds", "test")
    ref = cache.ref("segment", "1", {"name": "gt"}, [])
    assert not ref.complete
    cache.begin(ref, "1", {"name": "gt"}, [])
    (ref.dir / "partial.txt").write_text("x")
    assert not ref.complete
    # an incomplete directory is wiped on the next begin()
    cache.begin(ref, "1", {"name": "gt"}, [])
    assert not (ref.dir / "partial.txt").exists()
    cache.finish(ref)
    assert ref.complete and cache.ref("segment", "1", {"name": "gt"}, []).complete


# ----------------------------------------------------------------------------- artefacts


def test_detection_and_hypothesis_artefacts_round_trip(tmp_path, mini_bop: BopDataset):
    seg = GroundTruthSegmenter(mini_bop)
    view, gts = mini_bop.load_view(1, 0, load_rgb=False, load_depth=False)
    dets = seg.segment(view, mini_bop.object_ids)
    assert [d.object_id for d in dets] == [1, 1, 5]
    seg_dir = tmp_path / "segment"
    w = DetectionWriter(seg_dir)
    for d in dets:
        w.add(DetectionRecord(1, 0, d, time_s=0.5))
    df = w.close()
    assert len(df) == 3 and (seg_dir / "masks" / "000001" / "000000_000002.png").exists()
    back = read_detections(seg_dir, 1, 0)
    for a, b in zip(dets, back, strict=True):
        assert (a.object_id, a.detection_id, a.camera_id) == (
            b.object_id,
            b.detection_id,
            b.camera_id,
        )
        np.testing.assert_array_equal(a.mask, b.mask)
        assert a.bbox_xywh == b.bbox_xywh
    assert read_detections(seg_dir, 2, 0) == []

    rng = np.random.default_rng(0)
    hyps = [
        PoseHypothesis(
            camera_id="000000",
            object_id=d.object_id,
            detection_id=d.detection_id,
            hypothesis_id=k,
            T_camera_object=tf.random_rigid(rng),
            stage=Stage.COARSE,
            signals=QualitySignals(seg_score=d.score, pose_score=0.3 * k, n_inliers=12),
            source="foundpose",
            rejection_reason=None if k == 0 else "test",
        )
        for d in dets
        for k in range(2)
    ]
    pose_dir = tmp_path / "coarse_pose"
    write_hypotheses(pose_dir, [HypothesisRecord(1, 0, h, 1.25) for h in hyps])
    table = read_hypotheses_table(pose_dir)
    assert len(table) == 6 and set(QualitySignals.field_names()) <= set(table.columns)
    back_h = read_hypotheses(pose_dir, 1, 0)
    for a, b in zip(hyps, back_h, strict=True):
        assert (a.detection_id, a.hypothesis_id, a.stage, a.source) == (
            b.detection_id,
            b.hypothesis_id,
            b.stage,
            b.source,
        )
        assert a.rejection_reason == b.rejection_reason
        np.testing.assert_allclose(a.T_camera_object, b.T_camera_object)
        assert b.signals.n_inliers == 12 and np.isnan(b.signals.icp_fitness)
    # single-row round trip through the column mapping
    row = hypothesis_to_row(HypothesisRecord(1, 0, hyps[0]))
    assert hypothesis_from_row(row).rejection_reason is None


def test_cached_plugins_replay_adapter_outputs(tmp_path, mini_bop: BopDataset):
    from binposert.pipeline.artefacts import write_stage_info

    view, _ = mini_bop.load_view(2, 1, load_rgb=False, load_depth=False)
    dets = GroundTruthSegmenter(mini_bop).segment(view, [5])
    seg_dir = tmp_path / "segment"
    w = DetectionWriter(seg_dir)
    for d in dets:
        w.add(DetectionRecord(2, 1, d))
    w.close()
    write_stage_info(seg_dir, {"config": {"name": "cnos"}})
    cached = CachedSegmenter(seg_dir)
    assert cached.name == "cnos"
    assert [d.detection_id for d in cached.segment(view, [1, 5])] == [d.detection_id for d in dets]
    assert cached.segment(view, [1]) == []

    pose_dir = tmp_path / "coarse_pose"
    h = PoseHypothesis("000001", 5, dets[0].detection_id, 0, np.eye(4), Stage.COARSE, source="x")
    write_hypotheses(pose_dir, [HypothesisRecord(2, 1, h)])
    write_stage_info(pose_dir, {"config": {"name": "foundpose"}})
    est = CachedPoseEstimator(pose_dir)
    assert est.name == "foundpose"
    assert len(est.estimate(view, dets)) == 1
    other = Detection("000001", 5, np.zeros((1, 1), bool), 1.0, detection_id=99)
    assert est.estimate(view, [other]) == []


# ----------------------------------------------------------------------------- end to end


def test_smoke_experiment_runs_and_caches(tmp_path):
    cfg = _compose("experiment=smoke", f"outputs_root={tmp_path}")
    result = run(cfg, repo_root=REPO)
    ev = result.refs["evaluate"].dir
    report = json.loads((ev / "report.json").read_text())
    assert report["ar"] == pytest.approx(1.0) and report["n_gt"] == 12
    assert (ev / "binposert-smoke_mini_bop-test.csv").exists()
    assert (ev / "run_manifest.json").exists() and (ev / "summary.md").exists()
    manifest = json.loads((ev / "run_manifest.json").read_text())
    assert manifest["experiment"] == "smoke" and set(manifest["stages"]) == {
        "segment",
        "coarse_pose",
        "evaluate",
    }
    assert all(not s["cached"] for s in manifest["stages"].values())
    bins = {(b["lo"], b["hi"]): b["n_gt"] for b in report["by_visibility"]}
    assert bins == {(0.1, 0.3): 0, (0.3, 0.6): 2, (0.6, 1.0): 10}

    # second run: everything cached, same directories
    again = run(cfg, repo_root=REPO)
    assert all(s["cached"] for s in again.manifest.stages.values())
    assert {k: r.dir for k, r in again.refs.items()} == {k: r.dir for k, r in result.refs.items()}

    # a config change downstream re-uses upstream stages and recomputes the rest
    noisy = _compose(
        "experiment=smoke", f"outputs_root={tmp_path}", "estimator.params.t_sigma_mm=3"
    )
    third = run(noisy, repo_root=REPO)
    assert third.refs["segment"].dir == result.refs["segment"].dir
    assert third.refs["coarse_pose"].dir != result.refs["coarse_pose"].dir
    assert (
        third.manifest.stages["segment"]["cached"]
        and not third.manifest.stages["coarse_pose"]["cached"]
    )
    rep3 = json.loads((third.refs["evaluate"].dir / "report.json").read_text())
    assert rep3["ar_mspd"] < 1.0 or rep3["ar_mssd"] < 1.0


def test_run_py_cli_on_mini_fixture(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run.py"),
            "experiment=smoke",
            f"outputs_root={tmp_path}",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "done: evaluate" in out.stdout


def test_external_stage_without_cache_refuses_and_explains(tmp_path):
    cfg = _compose("experiment=A1", "dataset=mini_bop", f"outputs_root={tmp_path}")
    with pytest.raises(ExternalStageMissing) as exc:
        run(cfg, repo_root=REPO)
    msg = str(exc.value)
    assert "adapters/cnos_cli.py" in msg and "--out" in msg and "rsync" in msg


def test_external_stage_command_and_hash_exclude_machine_paths(tmp_path):
    a = _compose("experiment=A0", "dataset=mini_bop", f"outputs_root={tmp_path}")
    b = _compose(
        "experiment=A0",
        "dataset=mini_bop",
        f"outputs_root={tmp_path}",
        "estimator.command=[docker,run,foundpose]",
    )
    ra, rb = planned_refs(a, REPO), planned_refs(b, REPO)
    assert ra["coarse_pose"].hash == rb["coarse_pose"].hash
    c = _compose(
        "experiment=A0", "dataset=mini_bop", f"outputs_root={tmp_path}", "estimator.version=x"
    )
    assert planned_refs(c, REPO)["coarse_pose"].hash != ra["coarse_pose"].hash
    # GT-mask (A0) and CNOS (A1) runs share nothing downstream of segment
    d = _compose("experiment=A1", "dataset=mini_bop", f"outputs_root={tmp_path}")
    assert planned_refs(d, REPO)["coarse_pose"].hash != ra["coarse_pose"].hash


def test_external_stage_runs_a_fake_adapter(tmp_path, mini_bop: BopDataset):
    """A stub adapter writes hypotheses via the artefact API; the runner must accept its output."""
    stub = tmp_path / "stub_adapter.py"
    stub.write_text(
        "import argparse, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from binposert.data import BopDataset\n"
        "from binposert.pipeline import HypothesisRecord, read_detections, write_hypotheses\n"
        "from binposert.types import PoseHypothesis, QualitySignals, Stage\n"
        "ap = argparse.ArgumentParser()\n"
        "for a in ('--dataset-root','--split','--targets','--detections','--out','--params'):\n"
        "    ap.add_argument(a)\n"
        "ns = ap.parse_args()\n"
        "ds = BopDataset(ns.dataset_root, ns.split)\n"
        "recs = []\n"
        "for s in ds.scene_ids:\n"
        "    for i in ds.image_ids(s):\n"
        "        gts = {g.gt_index: g for g in ds.ground_truth(s, i)}\n"
        "        for d in read_detections(ns.detections, s, i, load_masks=False):\n"
        "            T = gts[d.detection_id].T_camera_object\n"
        "            h = PoseHypothesis(d.camera_id, d.object_id, d.detection_id, 0, T,\n"
        "                               Stage.COARSE, QualitySignals(pose_score=0.5),\n"
        "                               source='stub')\n"
        "            recs.append(HypothesisRecord(s, i, h, 0.1))\n"
        "write_hypotheses(ns.out, recs)\n"
    )
    cfg = _compose(
        "experiment=A0",
        "dataset=mini_bop",
        f"outputs_root={tmp_path}",
        "run_external=true",
        "evaluate.n_model_points=300",
        "evaluate.with_vsd=false",
    )
    cfg.estimator.command = [
        sys.executable,
        str(stub),
        "--dataset-root",
        "{dataset_root}",
        "--split",
        "{split}",
        "--detections",
        "{upstream_segment}",
        "--out",
        "{out_dir}",
        "--params",
        "{params_json}",
    ]
    result = run(cfg, repo_root=REPO)
    assert (result.refs["coarse_pose"].dir / HYPOTHESES_FILE).exists()
    report = json.loads((result.refs["evaluate"].dir / "report.json").read_text())
    assert report["ar_mssd"] == pytest.approx(1.0)
    assert len(read_detections_table(result.refs["segment"].dir)) == 12


def test_context_uses_targets_file_when_configured(tmp_path):
    cfg = _compose("dataset=tless", f"outputs_root={tmp_path}")
    if not (REPO / "data/bop/tless/test_primesense").is_dir():
        pytest.skip("T-LESS not downloaded on this machine")
    ctx = make_context(cfg, REPO)
    assert ctx.dataset.targets is not None
    assert len(ctx.dataset.scene_ids) == 20 and len(ctx.dataset.image_ids(1)) == 50
    assert ctx.dataset.target_object_ids(1, 1) == [2, 25, 29, 30]


def test_refine_stage_improves_noisy_coarse_poses_on_the_fixture(tmp_path):
    base = (
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "estimator.params.t_sigma_mm=4",
        "estimator.params.r_sigma_deg=6",
        "evaluate.with_vsd=false",
    )
    before = run(_compose(*base), repo_root=REPO)
    after = run(_compose(*base, "stages=[segment,coarse_pose,refine,evaluate]"), repo_root=REPO)
    rep_b = json.loads((before.refs["evaluate"].dir / "report.json").read_text())
    rep_a = json.loads((after.refs["evaluate"].dir / "report.json").read_text())
    assert rep_b["ar_mssd"] < 0.95 and rep_a["ar_mssd"] > rep_b["ar_mssd"] + 0.1
    assert after.refs["coarse_pose"].dir == before.refs["coarse_pose"].dir  # upstream reused
    ref_dir = after.refs["refine"].dir
    details = pd.read_parquet(ref_dir / "refine_details.parquet")
    assert len(details) == 12 and details["accepted"].sum() >= 11  # one occluded instance is gated
    assert np.isfinite(details["z_shift_mm"]).all() and (details["z_init_overlap"] > 0).all()
    summary = json.loads((ref_dir / "refine_summary.json").read_text())
    assert summary["n_hypotheses"] == 12 and summary["rejection_rate"] <= 1 / 12
    hyps = read_hypotheses_table(ref_dir)
    assert (hyps["stage"] == "refined").all() and hyps["icp_fitness"].min() > 0.9
    assert hyps["source"].str.endswith("+point_to_plane").all()
    # a different refiner config re-uses coarse_pose and gets its own refine directory
    gicp = run(
        _compose(*base, "stages=[segment,coarse_pose,refine,evaluate]", "refiner=gicp"),
        repo_root=REPO,
    )
    assert gicp.refs["refine"].dir != ref_dir and gicp.manifest.stages["coarse_pose"]["cached"]


# ----------------------------------------------------------------------------- multi-view stages


def test_multiview_stages_fuse_two_views_on_the_fixture(tmp_path):
    base = (
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "estimator.params.t_sigma_mm=4",
        "estimator.params.r_sigma_deg=6",
        "evaluate.with_vsd=false",
        "multiview.params.groups.n_views=2",
    )
    mv = "stages=[segment,coarse_pose,associate,fuse,evaluate]"
    single = run(_compose(*base, mv, "fusion=none"), repo_root=REPO)
    mean = run(_compose(*base, mv, "fusion=mean"), repo_root=REPO)
    best = run(_compose(*base, mv, "fusion=best"), repo_root=REPO)
    assert mean.refs["associate"].dir == single.refs["associate"].dir  # association shared
    assert mean.manifest.stages["associate"]["cached"]

    assoc = mean.refs["associate"].dir
    groups = json.loads((assoc / "groups.json").read_text())
    assert groups == {"1": [[0, 1]], "2": [[0, 1]]}
    tracks = pd.read_parquet(assoc / "tracks.parquet")
    # 2 scenes x 3 objects, each seen in both Views -> 6 tracks of 2 members
    sizes = tracks.groupby(["scene_id", "track_id"]).size()
    assert len(sizes) == 6 and (sizes == 2).all()
    assert (tracks.groupby(["scene_id", "track_id"]).object_id.nunique() == 1).all()
    assert (tracks.groupby(["scene_id", "track_id"]).image_id.nunique() == 2).all()
    summary = json.loads((assoc / "associate_summary.json").read_text())
    assert summary["n_tracks"] == 6 and summary["n_gated_out"] > 0

    fused = pd.read_parquet(mean.refs["fuse"].dir / "fused.parquet")
    assert len(fused) == 6 and (fused.n_views == 2).all() and (fused.n_members == 2).all()
    assert fused.dispersion_mm.between(0.5, 15).all()
    hyps = read_hypotheses_table(mean.refs["fuse"].dir)
    assert len(hyps) == 12 and hyps["source"].eq("fused:mean").all()
    # the fused pose is one world pose: its two projections agree through the extrinsics
    ds = BopDataset(REPO / "tests/fixtures/mini_bop", split="test")
    for (sid, _tid), grp in hyps.groupby(["scene_id", "hypothesis_id"]):
        worlds = []
        for _, r in grp.iterrows():
            view, _ = ds.load_view(int(sid), int(r.image_id), load_rgb=False, load_depth=False)
            worlds.append(view.T_world_camera @ hypothesis_from_row(r).T_camera_object)
        assert np.allclose(worlds[0], worlds[1], atol=1e-6)

    rep = {
        k: json.loads((r.refs["evaluate"].dir / "report.json").read_text())
        for k, r in {"none": single, "mean": mean, "best": best}.items()
    }
    assert all(v["n_images"] == 4 and v["n_gt"] == 12 for v in rep.values())
    assert rep["none"]["ar_mssd"] < 0.95  # the noisy single views leave room
    assert rep["mean"]["ar_mssd"] >= rep["none"]["ar_mssd"]
    assert rep["mean"]["ar_mssd"] >= rep["best"]["ar_mssd"] - 0.05

    # joint polish on top of the mean: every track accepted, error shrinks to the ICP floor
    joint = run(_compose(*base, mv, "fusion=mean_joint_icp", "fusion.n_workers=1"), repo_root=REPO)
    fj = pd.read_parquet(joint.refs["fuse"].dir / "fused.parquet")
    assert fj.joint_accepted.astype(bool).sum() >= 5 and fj.multiview_residual_mm.notna().all()
    # every track of a group polishes against depth, not only the first (View cache keyed on depth)
    assert (fj.joint_n_scene_points > 0).all() and not (fj.joint_reason == "no_depth").any()
    rep_j = json.loads((joint.refs["evaluate"].dir / "report.json").read_text())
    assert rep_j["ar_mssd"] >= rep["mean"]["ar_mssd"]
    assert rep_j["ar_mssd"] > 0.95


def test_extrinsic_noise_changes_the_association_hash_and_degrades_fusion(tmp_path):
    base = (
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "evaluate.with_vsd=false",
        "multiview.params.groups.n_views=2",
        "stages=[segment,coarse_pose,associate,fuse,evaluate]",
    )
    clean = run(_compose(*base), repo_root=REPO)
    noisy = run(
        _compose(
            *base,
            "multiview.params.extrinsic_noise.t_mm=10",
            "multiview.params.extrinsic_noise.deg=2",
        ),
        repo_root=REPO,
    )
    assert noisy.refs["associate"].dir != clean.refs["associate"].dir
    rep_c = json.loads((clean.refs["evaluate"].dir / "report.json").read_text())
    rep_n = json.loads((noisy.refs["evaluate"].dir / "report.json").read_text())
    assert rep_c["ar_mssd"] == pytest.approx(1.0)
    assert rep_n["ar_mssd"] < rep_c["ar_mssd"]
    fused = pd.read_parquet(noisy.refs["fuse"].dir / "fused.parquet")
    assert fused.dispersion_mm.min() > 2.0  # the perturbation shows up as disagreement


def test_strided_groups_cover_images_once():
    from binposert.pipeline.multiview_stage import strided_groups

    ids = list(range(50))
    for k in (1, 2, 3, 4):
        gs = strided_groups(ids, k)
        flat = [i for g in gs for i in g]
        assert len(gs) == 50 // k and all(len(g) == k for g in gs)
        assert len(flat) == len(set(flat))
        assert gs[0] == [j * (50 // k) for j in range(k)]
    assert strided_groups(ids, 2, max_groups=3) == [[0, 25], [1, 26], [2, 27]]
    assert strided_groups([1, 2], 3) == []
