"""Ground-truth masks as Detections: the A0 upper bound for segmentation (D4)."""

from __future__ import annotations

from collections.abc import Sequence

from binposert.data import BopDataset
from binposert.types import Detection, View


class GroundTruthSegmenter:
    """One Detection per annotated object with a non-empty visible mask.

    ``detection_id`` equals the BOP ``gt_index`` so masks and GT poses stay linkable. GT with
    ``visible_fraction`` below ``min_visible_fraction`` is dropped (BOP evaluation ignores < 0.1
    anyway, and the estimators cannot do anything with an empty mask).
    """

    name = "gt"

    def __init__(self, dataset: BopDataset, min_visible_fraction: float = 0.0) -> None:
        self.dataset = dataset
        self.min_visible_fraction = min_visible_fraction

    def segment(self, view: View, object_ids: Sequence[int] | None = None) -> list[Detection]:
        _, gts = self.dataset.load_view(
            view.scene_id, view.image_id, load_rgb=False, load_depth=False
        )
        wanted = None if object_ids is None else set(object_ids)
        out: list[Detection] = []
        for g in gts:
            if wanted is not None and g.object_id not in wanted:
                continue
            if g.visible_fraction < self.min_visible_fraction:
                continue
            mask = self.dataset.gt_mask(view.scene_id, view.image_id, g.gt_index, visible_only=True)
            if not mask.any():
                continue
            out.append(
                Detection(
                    camera_id=view.camera_id,
                    object_id=g.object_id,
                    mask=mask,
                    score=1.0,
                    detection_id=g.gt_index,
                )
            )
        return out
