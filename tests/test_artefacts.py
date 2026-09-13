"""Frozen artefact schema (D12): writer -> reader round trips, validation, BOP RLE."""

import numpy as np
import pandas as pd
import pytest

from binposert import artefacts
from binposert import transforms as tf
from binposert.pose import CachedPoseEstimator
from binposert.segment import CachedSegmenter, GroundTruthSegmenter
from binposert.types import (
    DETECTION_COLUMNS,
    POSE_HYPOTHESIS_COLUMNS,
    PoseHypothesis,
    QualitySignals,
    Stage,
)


def test_detections_round_trip_through_cached_segmenter(mini_bop, tmp_path):
    view, gts = mini_bop.load_view(1, 0, load_rgb=False, load_depth=False)
    dets = GroundTruthSegmenter(mini_bop).segment(view)
    assert [d.detection_id for d in dets] == [g.gt_index for g in gts]

    writer = artefacts.DetectionsWriter(tmp_path, source="gt")
    for d in dets:
        writer.add(1, 0, d, time_s=0.01)
    table = writer.finish(stage="segment")
    assert list(table.columns) == list(DETECTION_COLUMNS)
    with pytest.raises(artefacts.ArtefactError):  # no _SUCCESS yet
        CachedSegmenter(tmp_path)
    artefacts.mark_success(tmp_path)

    seg = CachedSegmenter(tmp_path)
    assert seg.name == "gt"
    back = seg.segment(view)
    assert len(back) == len(dets)
    for a, b in zip(dets, back, strict=True):
        assert (a.object_id, a.detection_id, a.score) == (b.object_id, b.detection_id, b.score)
        np.testing.assert_array_equal(a.mask, b.mask)
    assert seg.segment(view, object_ids=[5])[0].object_id == 5
    other, _ = mini_bop.load_view(2, 1, load_rgb=False, load_depth=False)
    assert seg.segment(other) == []


def test_pose_hypotheses_round_trip_through_cached_estimator(mini_bop, tmp_path):
    rng = np.random.default_rng(0)
    view, gts = mini_bop.load_view(2, 1, load_rgb=False, load_depth=False)
    writer = artefacts.PoseHypothesesWriter(tmp_path, source="fake")
    hyps = []
    for g in gts:
        for h in range(2):
            hyp = PoseHypothesis(
                camera_id=view.camera_id,
                object_id=g.object_id,
                detection_id=g.gt_index,
                hypothesis_id=h,
                T_camera_object=tf.random_rigid(rng),
                stage=Stage.COARSE,
                signals=QualitySignals(pose_score=0.5 - 0.1 * h, n_inliers=40 + h),
                source="fake",
                rejection_reason=None if h == 0 else "displacement",
            )
            hyps.append(hyp)
            writer.add(2, 1, hyp, time_s=1.5)
    table = writer.finish(stage="coarse_pose")
    assert list(table.columns) == list(POSE_HYPOTHESIS_COLUMNS)
    artefacts.mark_success(tmp_path)

    est = CachedPoseEstimator(tmp_path)
    dets = GroundTruthSegmenter(mini_bop).segment(view)
    for d in dets:
        back = est.estimate(view, d, mini_bop.load_model(d.object_id))
        want = [h for h in hyps if h.detection_id == d.detection_id]
        assert [b.hypothesis_id for b in back] == [0, 1]
        for a, b in zip(want, back, strict=True):
            np.testing.assert_allclose(a.T_camera_object, b.T_camera_object)
            assert b.stage is Stage.COARSE and b.source == "fake"
            assert b.signals.pose_score == pytest.approx(a.signals.pose_score)
            assert np.isnan(b.signals.icp_fitness)
            assert b.rejection_reason == a.rejection_reason


def test_reader_rejects_missing_columns(tmp_path):
    df = pd.DataFrame({"scene_id": [1], "image_id": [0]})
    df.to_parquet(tmp_path / "pose_hypotheses.parquet")
    artefacts.mark_success(tmp_path)
    with pytest.raises(artefacts.ArtefactError, match="missing columns"):
        artefacts.read_pose_hypotheses_table(tmp_path)


def test_rle_round_trip_and_coco_compressed_string():
    rng = np.random.default_rng(1)
    mask = np.zeros((24, 31), dtype=bool)
    mask[3:10, 4:20] = True
    mask[15:20, 0:5] = True
    mask[rng.random((24, 31)) > 0.95] = True
    rle = artefacts.mask_to_rle(mask)
    assert rle["size"] == [24, 31]
    np.testing.assert_array_equal(artefacts.rle_to_mask(rle), mask)
    # mask starting with a foreground pixel: encoding must begin with an empty background run
    mask[0, 0] = True
    rle = artefacts.mask_to_rle(mask)
    assert rle["counts"][0] == 0
    np.testing.assert_array_equal(artefacts.rle_to_mask(rle), mask)
    # pycocotools-compressed strings (rleToString: 5-bit groups, +48, delta-coded from the third
    # run): "032N" encodes [0, 3, 2, 1] — a 3x2 mask, column-major, pixels 0-2 and 5 set
    assert artefacts._decode_coco_counts("032N") == [0, 3, 2, 1]
    assert artefacts._decode_coco_counts("032NV1ke0mN") == [0, 3, 2, 1, 40, 700, 5]
    decoded = artefacts.rle_to_mask({"size": [3, 2], "counts": "032N"})
    assert decoded.ravel(order="F").tolist() == [True, True, True, False, False, True]


def test_symmetric_rle_empty_mask():
    empty = np.zeros((5, 7), dtype=bool)
    rle = artefacts.mask_to_rle(empty)
    assert sum(rle["counts"]) == 35
    assert not artefacts.rle_to_mask(rle).any()
