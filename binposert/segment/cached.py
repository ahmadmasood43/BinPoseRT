"""Detections read back from a stage artefact directory (CNOS or any GPU segmenter, D4/D12)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from binposert import artefacts
from binposert.types import Detection, View


class CachedSegmenter:
    """Serves the Detections stored under ``directory`` (``detections.parquet`` + ``masks/``).

    The directory must carry a ``_SUCCESS`` marker and the frozen column schema; both are checked
    once at construction so a partially rsync'd cache fails loudly.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.table: pd.DataFrame = artefacts.read_detections_table(self.directory, validate=True)
        meta = artefacts.read_meta(self.directory)
        self.name = str(meta.get("source", self.table["source"].iloc[0] if len(self.table) else ""))
        self._by_image: dict[tuple[int, int], pd.DataFrame] = {}
        for key, g in self.table.groupby(["scene_id", "image_id"], sort=True):
            s, i = (int(k) for k in key)  # type: ignore[call-overload]
            self._by_image[(s, i)] = g.sort_values(["object_id", "detection_id"])

    def segment(self, view: View, object_ids: Sequence[int] | None = None) -> list[Detection]:
        rows = self._by_image.get((view.scene_id, view.image_id))
        if rows is None:
            return []
        if object_ids is not None:
            rows = rows[rows["object_id"].isin(list(object_ids))]
        return artefacts.detections_from_rows(self.directory, rows)

    @property
    def image_keys(self) -> list[tuple[int, int]]:
        return sorted(self._by_image)
