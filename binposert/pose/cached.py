"""PoseHypotheses read back from a stage artefact directory (FoundPose / MegaPose adapters, D3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from binposert import artefacts
from binposert.types import Detection, ObjectModel, PoseHypothesis, View


class CachedPoseEstimator:
    """Serves the hypotheses stored under ``directory`` (``pose_hypotheses.parquet``)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.table: pd.DataFrame = artefacts.read_pose_hypotheses_table(
            self.directory, validate=True
        )
        meta = artefacts.read_meta(self.directory)
        self.name = str(meta.get("source", self.table["source"].iloc[0] if len(self.table) else ""))
        self._by_detection: dict[tuple[int, int, int], pd.DataFrame] = {}
        for key, g in self.table.groupby(["scene_id", "image_id", "detection_id"], sort=True):
            s, i, d = (int(k) for k in key)  # type: ignore[call-overload]
            self._by_detection[(s, i, d)] = g.sort_values("hypothesis_id")

    def estimate(
        self, view: View, detection: Detection, model: ObjectModel
    ) -> list[PoseHypothesis]:
        rows = self._by_detection.get((view.scene_id, view.image_id, detection.detection_id))
        if rows is None:
            return []
        rows = rows[rows["object_id"] == detection.object_id]
        return artefacts.pose_hypotheses_from_rows(rows)
