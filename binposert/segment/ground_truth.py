"""Upper-bound Segmenter: BOP ``mask_visib`` for every sufficiently visible ground-truth object."""

from __future__ import annotations

from binposert.data import BopDataset
from binposert.types import Detection, View

DEFAULT_MIN_VISIBLE_FRACTION = 0.1  # BOP ignores less visible GT, so a Detection would never score


class GroundTruthSegmenter:
    name = "gt"

    def __init__(
        self, dataset: BopDataset, min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION
    ) -> None:
        self.dataset = dataset
        self.min_visible_fraction = min_visible_fraction

    def segment(self, view: View, object_ids: list[int]) -> list[Detection]:
        wanted = set(object_ids)
        out: list[Detection] = []
        for gt in self.dataset.ground_truth(view.scene_id, view.image_id):
            if gt.object_id not in wanted or gt.visible_fraction < self.min_visible_fraction:
                continue
            mask = self.dataset.gt_mask(
                view.scene_id, view.image_id, gt.gt_index, visible_only=True
            )
            if not mask.any():
                continue
            out.append(
                Detection(
                    camera_id=view.camera_id,
                    object_id=gt.object_id,
                    mask=mask,
                    score=1.0,
                    detection_id=gt.gt_index,
                )
            )
        return out
