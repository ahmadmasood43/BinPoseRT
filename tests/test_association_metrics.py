"""Association purity / completeness against ground truth on the mini fixture."""

import pandas as pd

from binposert.evaluate.association import association_metrics, label_hypotheses
from binposert.pipeline.artefacts import HYPOTHESIS_COLUMNS, HypothesisRecord, hypothesis_to_row
from binposert.types import PoseHypothesis, QualitySignals, Stage


def _tracks(mini_bop, assignment):
    """One hypothesis per GT of scene 1 in both Views; ``assignment[(image_id, gt_index)]`` is
    the track_id it is put in."""
    rows = []
    for image_id in (0, 1):
        for g in mini_bop.ground_truth(1, image_id):
            h = PoseHypothesis(
                f"{image_id:06d}",
                g.object_id,
                g.gt_index,
                0,
                g.T_camera_object,
                Stage.REFINED,
                QualitySignals(seg_score=1.0),
            )
            row = hypothesis_to_row(HypothesisRecord(1, image_id, h, 0.0))
            row.update(
                {"group_id": 0, "track_id": assignment[(image_id, g.gt_index)], "weight": 1.0}
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=[*HYPOTHESIS_COLUMNS, "group_id", "track_id", "weight"])


def test_perfect_association_scores_one(mini_bop):
    tracks = _tracks(mini_bop, {(i, g): g for i in (0, 1) for g in range(3)})
    labels = label_hypotheses(tracks, mini_bop)
    assert (labels.to_numpy() == tracks.detection_id.to_numpy()).all()  # detection_id = gt_index
    m = association_metrics(tracks, labels)
    assert m.n_tracks == 3 and m.n_multi_tracks == 3 and m.n_labelled == 6
    assert m.track_purity == 1.0 and m.member_purity == 1.0 and m.completeness == 1.0
    assert m.n_mixed_tracks == 0 and m.n_split_instances == 0 and m.n_instances_multi_view == 3


def test_swapped_members_are_mixed_and_split(mini_bop):
    # the two copies of object 1 (gt 0 and 1) are swapped between the Views
    assignment = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 1, (1, 1): 0, (1, 2): 2}
    tracks = _tracks(mini_bop, assignment)
    m = association_metrics(tracks, label_hypotheses(tracks, mini_bop))
    assert m.n_mixed_tracks == 2 and m.track_purity == 1 / 3
    assert m.member_purity == 4 / 6
    assert m.n_split_instances == 2 and m.completeness == 4 / 6
    assert {tuple(sorted(x["instances"])) for x in m.mixed} == {(0, 1)}


def test_unlabelled_hypotheses_are_ignored(mini_bop):
    tracks = _tracks(mini_bop, {(i, g): g for i in (0, 1) for g in range(3)})
    far = tracks.copy()
    far.loc[far.index[0], "T_03"] += 500.0  # 500 mm off: matches no GT
    labels = label_hypotheses(far, mini_bop)
    assert labels.iloc[0] == -1 and (labels.iloc[1:] >= 0).all()
    m = association_metrics(far, labels)
    assert m.n_labelled == 5 and m.n_instances_multi_view == 2 and m.completeness == 1.0
