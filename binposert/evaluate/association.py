"""Association correctness against ground truth (Gamma exit criterion).

BOP annotates the same physical object with the same ``gt_index`` in every image of a scene (the
scene is static; checked on T-LESS in Gamma), so a ground-truth *instance* is ``(scene_id,
gt_index)``. Every hypothesis of a view group is labelled with the instance whose annotated pose in
that View is nearest (translation, within ``match_factor × diameter``) — the physical object the
hypothesis refers to, whether or not its pose is accurate. From those labels:

* **purity** — share of multi-member tracks whose labelled members all refer to one instance
  (member-level: share of labelled members agreeing with their track's majority instance);
* **completeness** — for every instance labelled in ≥ 2 Views of a group, the share of those Views
  whose labelled hypothesis sits in the instance's largest track (1.0 = one track per object);
* **mixed / split** counts — tracks joining two instances, instances spread over several tracks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.pipeline.artefacts import columns_to_transform


@dataclass
class AssociationMetrics:
    n_groups: int
    n_tracks: int
    n_multi_tracks: int
    n_labelled: int
    track_purity: float  # tracks (≥ 2 labelled members) with a single instance
    member_purity: float
    completeness: float
    n_mixed_tracks: int
    n_split_instances: int
    n_instances_multi_view: int
    mixed: list[dict[str, Any]]  # for the gallery: scene_id, group_id, track_id, instances

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "mixed"}
        d["n_mixed_examples"] = len(self.mixed)
        return d


def label_hypotheses(
    tracks: pd.DataFrame, dataset: BopDataset, match_factor: float = 0.5
) -> pd.Series:
    """``gt_index`` (or -1) of the nearest same-object ground-truth pose in the hypothesis' View."""
    labels = np.full(len(tracks), -1, dtype=np.int64)
    diam = {oid: dataset.load_model(oid).diameter for oid in tracks.object_id.unique()}
    for key_raw, grp in tracks.groupby(["scene_id", "image_id"]):
        sid, iid = (int(x) for x in cast(tuple[Any, Any], key_raw))
        gts = dataset.ground_truth(sid, iid)
        by_obj: dict[int, list[Any]] = defaultdict(list)
        for g in gts:
            by_obj[g.object_id].append(g)
        for idx, row in grp.iterrows():
            oid = int(row.object_id)
            cands = by_obj.get(oid, [])
            if not cands:
                continue
            t = columns_to_transform(row)[:3, 3]
            d = np.asarray([np.linalg.norm(g.T_camera_object[:3, 3] - t) for g in cands])
            j = int(np.argmin(d))
            if d[j] <= match_factor * diam[oid]:
                labels[tracks.index.get_loc(idx)] = cands[j].gt_index
    return pd.Series(labels, index=tracks.index, name="gt_index")


def association_metrics(tracks: pd.DataFrame, labels: pd.Series) -> AssociationMetrics:
    t = tracks.assign(gt_index=labels.to_numpy())
    n_groups = int(t.groupby(["scene_id", "group_id"]).ngroups)
    keys = ["scene_id", "group_id", "track_id"]
    sizes = t.groupby(keys).size()
    n_tracks = int(len(sizes))
    n_multi = int((sizes > 1).sum())

    labelled = t[t.gt_index >= 0]
    pure_tracks = agree = total_members = 0
    n_tracks_scored = 0
    mixed: list[dict[str, Any]] = []
    for key, grp in labelled.groupby(keys):
        if len(grp) < 2:
            continue
        n_tracks_scored += 1
        counts = Counter(grp.gt_index.tolist())
        majority = counts.most_common(1)[0][1]
        agree += majority
        total_members += len(grp)
        if len(counts) == 1:
            pure_tracks += 1
        else:
            sid, gid, tid = (int(x) for x in cast(tuple[Any, Any, Any], key))
            mixed.append(
                {
                    "scene_id": sid,
                    "group_id": gid,
                    "track_id": tid,
                    "object_id": int(grp.object_id.iloc[0]),
                    "instances": {int(k): int(v) for k, v in counts.items()},
                    "image_ids": [int(i) for i in grp.image_id],
                }
            )

    # completeness: per (scene, group, gt_index) seen in >= 2 Views, share of Views in the largest
    # track. Several hypotheses of one View may carry the same label (duplicate Detections); a View
    # counts as covered when any of them is in the largest track.
    found = covered = 0
    n_inst = split = 0
    for _, grp in labelled.groupby(["scene_id", "group_id", "gt_index"]):
        views = grp.image_id.nunique()
        if views < 2:
            continue
        n_inst += 1
        per_track = grp.groupby("track_id").image_id.nunique()
        best = int(per_track.max())
        found += views
        covered += best
        if len(per_track) > 1:
            split += 1

    return AssociationMetrics(
        n_groups=n_groups,
        n_tracks=n_tracks,
        n_multi_tracks=n_multi,
        n_labelled=int(len(labelled)),
        track_purity=pure_tracks / n_tracks_scored if n_tracks_scored else float("nan"),
        member_purity=agree / total_members if total_members else float("nan"),
        completeness=covered / found if found else float("nan"),
        n_mixed_tracks=len(mixed),
        n_split_instances=split,
        n_instances_multi_view=n_inst,
        mixed=mixed,
    )
