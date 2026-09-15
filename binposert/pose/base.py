from __future__ import annotations

from typing import Protocol

from binposert.types import Detection, PoseHypothesis, View


class PoseEstimator(Protocol):
    """``(View, Detection) -> list[PoseHypothesis]`` tagged ``stage=coarse``. Neural estimators run
    remotely (adapters/) and are consumed through :class:`CachedPoseEstimator`."""

    name: str

    def estimate(self, view: View, detections: list[Detection]) -> list[PoseHypothesis]: ...
