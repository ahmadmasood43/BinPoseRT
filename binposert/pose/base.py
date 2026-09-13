"""``PoseEstimator``: ``(View, Detection, ObjectModel) → list[PoseHypothesis]`` (D3)."""

from __future__ import annotations

from typing import Protocol

from binposert.types import Detection, ObjectModel, PoseHypothesis, View


class PoseEstimator(Protocol):
    name: str

    def estimate(
        self, view: View, detection: Detection, model: ObjectModel
    ) -> list[PoseHypothesis]:
        """Coarse hypotheses for one Detection, best first. May be empty."""
        ...
