"""Next-best-view (D14): the score prefers the View that can decide what is uncertain, the loop
follows its policy and stopping rule, and the ``nbv`` stage runs on the fixture."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from binposert.active import (
    LoopParams,
    NbvScoreParams,
    NbvScorer,
    TrackBelief,
    hypothesis_set,
    run_episode,
    scaled_camera,
    silhouette_disagreement,
    strided_order,
    track_beliefs,
)
from binposert.active.loop import Belief
from binposert.pipeline.run import run
from binposert.transforms import rotvec_T
from binposert.types import View
from tests.synth import box_model, cylinder_model, look_at_T_world_camera, simple_K
from tests.test_pipeline import CONFIGS, REPO

K = simple_K()
SIZE = (240, 320)


def _view(image_id: int, eye, target=(0, 0, 0)) -> View:
    return View(
        camera_id=f"{image_id:06d}",
        K=K,
        T_world_camera=look_at_T_world_camera(eye, target),
        rgb=None,
        depth=None,
        image_size=SIZE,
        scene_id=0,
        image_id=image_id,
    )


def _belief(track_id, model, T_world_object, camera: View, confidence=0.0) -> TrackBelief:
    return TrackBelief(
        track_id,
        model.object_id,
        T_world_object,
        confidence,
        [T_world_object],
        [camera.T_world_camera],
    )


# ----------------------------------------------------------------------------- score pieces


def test_silhouette_disagreement_is_one_minus_iou_and_empty_masks_agree():
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    b = np.zeros_like(a)
    b[4:8, 4:8] = True
    assert silhouette_disagreement([a, a]) == 0.0
    assert silhouette_disagreement([a, ~a]) == 1.0
    inter, union = 4, 16 + 16 - 4
    assert silhouette_disagreement([a, b]) == pytest.approx(1 - inter / union)
    assert silhouette_disagreement([np.zeros_like(a), np.zeros_like(a)]) == 0.0
    assert silhouette_disagreement([a]) == 0.0


def test_scaled_camera_keeps_the_pixel_centre_convention():
    K2, size = scaled_camera(K, SIZE, 0.5)
    assert size == (120, 160)
    assert K2[0, 0] == K[0, 0] * 0.5
    # the centre of the full-resolution image maps to the centre of the half-resolution one
    u = (K[0, 2] + 0.5) * 0.5 - 0.5
    assert K2[0, 2] == pytest.approx(u)


def test_hypothesis_set_is_members_plus_samples_along_the_observing_ray():
    model = box_model(extents=(60.0, 40.0, 20.0))
    cam = _view(0, (0, 0, 500))
    T = np.eye(4)
    b = _belief(0, model, T, cam)
    p = NbvScoreParams(n_samples=200, sigma_ray=0.05, sigma_lateral=0.01, sigma_rot_deg=0.0)
    hyps = hypothesis_set(b, model.diameter, p, np.random.default_rng(0))
    assert len(hyps) == 200 and np.allclose(hyps[0], T)
    # samples are spread along the camera's optical axis (world z here) more than across it
    d = np.asarray([h[:3, 3] for h in hyps[1:]])
    assert d[:, 2].std() > 3 * d[:, 0].std()
    assert d[:, 2].std() == pytest.approx(0.05 * model.diameter, rel=0.3)


def test_symmetric_members_do_not_disagree():
    """Two members of a cylinder track that differ by a turn about its axis render the same
    silhouette everywhere: the symmetry is aligned away and the score sees no uncertainty."""
    model = cylinder_model(steps=36)
    cam = _view(0, (400, 0, 300))
    T = rotvec_T([1, 0, 0], 30.0)
    T_turned = T @ rotvec_T([0, 0, 1], 130.0)  # on the 10° sampling grid (ADR-0003)
    beliefs = track_beliefs(
        [{"track_id": 0, "object_id": model.object_id, "T_world_object": T, "confidence": 0.0}],
        {0: [(T, cam.T_world_camera), (T_turned, cam.T_world_camera)]},
        {model.object_id: model},
    )
    assert np.allclose(beliefs[0].members[0], beliefs[0].members[1], atol=1e-6)
    scorer = NbvScorer({model.object_id: model}, NbvScoreParams(n_samples=2, render_scale=1.0))
    candidate = _view(1, (0, 400, 300))
    score, rows = scorer.score_view(candidate, beliefs, scorer.hypothesis_sets(beliefs, 0))
    assert rows[0]["U"] == 0.0 and score.score == 0.0


# ----------------------------------------------------------------------------- the D14 claim


def test_depth_ambiguity_is_decided_by_the_view_across_the_ray():
    """A single-view track is uncertain along its camera's ray. The candidate that looks across
    that ray sees the samples as a lateral spread (disagreement); the candidate that looks along
    the same ray sees only a change of scale. The first wins (D14: disagreement in one view ->
    that view wins)."""
    model = box_model(extents=(60.0, 40.0, 20.0), symmetric=False)
    first = _view(0, (0, 0, 500))  # looks straight down world z
    b = _belief(0, model, np.eye(4), first, confidence=0.2)
    scorer = NbvScorer(
        {model.object_id: model},
        NbvScoreParams(n_samples=8, sigma_ray=0.1, sigma_lateral=0.0, sigma_rot_deg=0.0),
    )
    across = _view(1, (500, 0, 60))
    along = _view(2, (0, 0, 800))
    sets = scorer.hypothesis_sets([b], 0)
    s_across, rows_across = scorer.score_view(across, [b], sets)
    s_along, rows_along = scorer.score_view(along, [b], sets)
    assert rows_across[0]["U"] > 3 * rows_along[0]["U"]
    assert s_across.score > 0.0 and s_across.score > s_along.score >= 0.0
    assert rows_across[0]["weight"] == pytest.approx(0.8)
    # the full candidate ranking says the same
    ranked = scorer.score_candidates(0, [along, across], [b])
    assert max(ranked, key=lambda c: c.score).image_id == 1


def test_an_occluded_view_scores_lower_and_confident_tracks_do_not_count():
    small = box_model(object_id=1, extents=(40.0, 40.0, 20.0), symmetric=False)
    big = box_model(object_id=2, extents=(200.0, 20.0, 200.0), symmetric=False)  # a wall
    first = _view(0, (0, 0, 500))
    T_small = np.eye(4)
    T_big = rotvec_T([1, 0, 0], 0.0, t=(0.0, -150.0, 60.0))  # standing on the -y side
    uncertain = _belief(0, small, T_small, first, confidence=0.0)
    confident = _belief(1, big, T_big, first, confidence=1.0)
    scorer = NbvScorer({1: small, 2: big}, NbvScoreParams(n_samples=6, render_scale=0.5))
    # the view from -y looks through the wall at the small box; the view from +y does not
    occluded = _view(1, (0, -500, 120))
    clear = _view(2, (0, 500, 120))
    beliefs = [uncertain, confident]
    sets = scorer.hypothesis_sets(beliefs, 0)
    s_occ, rows_occ = scorer.score_view(occluded, beliefs, sets)
    s_clear, rows_clear = scorer.score_view(clear, beliefs, sets)
    assert rows_occ[0]["V"] < 0.5 < rows_clear[0]["V"]
    assert s_occ.score < s_clear.score
    # the confident big box contributes nothing whatever it sees
    assert rows_occ[1]["weight"] == 0.0 and rows_clear[1]["weight"] == 0.0


# ----------------------------------------------------------------------------- the loop


class _StubUpdater:
    """A belief whose Verdicts are scripted per number of Views used."""

    def __init__(self, verdicts_by_n: dict[int, list[str]]) -> None:
        self.verdicts_by_n = verdicts_by_n
        self.calls: list[list[int]] = []

    def update(self, image_ids, project_into):
        self.calls.append(list(image_ids))
        v = self.verdicts_by_n.get(len(image_ids), ["accept"])
        fused = pd.DataFrame({"track_id": range(len(v)), "verdict": v})
        return Belief(list(image_ids), pd.DataFrame(), fused, [], 0.0), []


class _StubScorer:
    def score_candidates(self, scene_id, candidates, beliefs):
        raise AssertionError("the fixed / random policies never score")


def test_strided_order_matches_gamma_groups():
    pool = list(range(0, 51, 3))  # 17 images, as the XYZ-IBD val targets
    assert strided_order(pool, 0, 4)[:4] == [0, 12, 24, 36]
    assert strided_order(pool, 1, 4)[:4] == [3, 15, 27, 39]
    assert sorted(strided_order(pool, 2, 4)) == pool
    assert strided_order([0, 1], 0, 4) == [0, 1]  # a pool smaller than the group


def test_random_prefixes_nest_and_fixed_follows_the_strided_order():
    pool = list(range(0, 51, 3))
    updater = _StubUpdater({})
    base = dict(policy="random", stop="budget", start_group=1, seed=3)
    e3 = run_episode(0, None, pool, updater, _StubScorer(), LoopParams(budget=3, **base))
    e4 = run_episode(0, None, pool, updater, _StubScorer(), LoopParams(budget=4, **base))
    assert e3.start == 3 and e3.reference == [3, 15, 27, 39]
    assert e4.views_used[:3] == e3.views_used and len(e4.views_used) == 4
    assert e3.steps[-1].reason == "stop:budget" and len(set(e4.views_used)) == 4
    other = run_episode(
        0,
        None,
        pool,
        updater,
        _StubScorer(),
        LoopParams(budget=4, policy="random", seed=4, start_group=1),
    )
    assert other.views_used != e4.views_used
    fixed = run_episode(
        0, None, pool, updater, _StubScorer(), LoopParams(budget=4, policy="fixed", start_group=1)
    )
    assert fixed.views_used == [3, 15, 27, 39]
    everything = run_episode(
        0, None, pool, updater, _StubScorer(), LoopParams(budget=0, policy="fixed")
    )
    assert sorted(everything.views_used) == pool and everything.steps[-1].reason == "stop:budget"


def test_verdict_stop_unlocks_views_only_while_a_track_requests_one():
    pool = [0, 1, 2, 3]
    scripted = {1: ["accept", "request_view"], 2: ["request_view"], 3: ["accept", "reject"]}
    e = run_episode(
        0,
        None,
        pool,
        _StubUpdater(scripted),
        _StubScorer(),
        LoopParams(policy="fixed", budget=0, stop="verdict"),
    )
    assert e.views_used == [0, 1, 2] and e.steps[-1].reason == "stop:verdict"
    quiet = run_episode(
        0,
        None,
        pool,
        _StubUpdater({1: ["accept", "reject"]}),
        _StubScorer(),
        LoopParams(policy="fixed", budget=0, stop="verdict"),
    )
    assert quiet.views_used == [0] and quiet.steps[-1].reason == "stop:verdict"
    wider = run_episode(
        0,
        None,
        pool,
        _StubUpdater({1: ["accept", "reject"]}),
        _StubScorer(),
        LoopParams(policy="fixed", budget=0, stop="verdict", uncertain=("request_view", "reject")),
    )
    assert wider.views_used == [0, 1] and wider.steps[-1].reason == "stop:verdict"


# ----------------------------------------------------------------------------- the stage


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name="config", overrides=list(overrides))


def test_nbv_stage_on_the_fixture(tmp_path):
    base = (
        "experiment=smoke",
        f"outputs_root={tmp_path}",
        "estimator.params.t_sigma_mm=4",
        "estimator.params.r_sigma_deg=6",
        "evaluate.with_vsd=false",
        "stages=[segment,coarse_pose,nbv,evaluate]",
        "nbv.n_workers=1",
    )
    nbv = run(_compose(*base), repo_root=REPO)
    d = nbv.refs["nbv"].dir
    summary = json.loads((d / "nbv_summary.json").read_text())
    assert summary["n_episodes"] == 2 and summary["views_used"] == {"mean": 2.0, "min": 2, "max": 2}
    assert summary["choices"] == {"nbv": 2}  # the one candidate scored above zero both times
    episodes = json.loads((d / "episodes.json").read_text())
    assert [e["reference"] for e in episodes] == [[0, 1], [0, 1]]
    assert all(len(s["candidates"]) == 1 for e in episodes for s in e["steps"][:1])
    fused = pd.read_parquet(d / "fused.parquet")
    assert len(fused) == 6 and (fused.n_members == 2).all()
    assert (
        fused.confidence.between(0, 1).all()
        and fused.verdict.isin(["accept", "reject", "request_view"]).all()
    )
    hyps = pd.read_parquet(d / "hypotheses.parquet")
    assert len(hyps) == 12 and hyps.confidence.notna().all()
    assert json.loads((d / "groups.json").read_text()) == {"1": [[0, 1]], "2": [[0, 1]]}
    report = json.loads((nbv.refs["evaluate"].dir / "report.json").read_text())
    assert report["n_images"] == 4 and report["n_gt"] == 12

    # a single-View budget is still projected into both reference Views, and the policy /
    # budget are part of the stage hash
    one = run(_compose(*base, "nbv.params.budget=1", "nbv.params.policy=fixed"), repo_root=REPO)
    assert one.refs["nbv"].dir != d
    f1 = pd.read_parquet(one.refs["nbv"].dir / "fused.parquet")
    h1 = pd.read_parquet(one.refs["nbv"].dir / "hypotheses.parquet")
    assert (f1.n_members == 1).all() and len(h1) == 2 * len(f1)
    e1 = json.loads((one.refs["nbv"].dir / "episodes.json").read_text())
    assert all(e["stop"] == "stop:budget" and e["views_used"] == [0] for e in e1)
    # a refit of the confidence model must miss the cache: the fingerprints are in the hash
    again = run(_compose(*base), repo_root=REPO)
    assert again.manifest.stages["nbv"]["cached"]
    # the oracle reads the ground truth and records its gain per candidate
    oracle = run(_compose(*base, "nbv.params.policy=oracle", "nbv.params.budget=2"), repo_root=REPO)
    eo = json.loads((oracle.refs["nbv"].dir / "episodes.json").read_text())
    assert all(e["views_used"] == [0, 1] and e["steps"][0]["reason"] == "oracle" for e in eo)
    assert all(e["steps"][0]["candidates"][0]["score"] >= 1 for e in eo)
