"""Segmenter that replays a ``segment`` stage directory written by an adapter (e.g. CNOS)."""

from __future__ import annotations

from pathlib import Path

from binposert.pipeline.artefacts import read_detections, read_detections_table, read_stage_info
from binposert.types import Detection, View


class CachedSegmenter:
    def __init__(self, stage_dir: str | Path) -> None:
        self.stage_dir = Path(stage_dir)
        info = read_stage_info(self.stage_dir)
        self.name = str(info.get("config", {}).get("name", "cached"))
        self.table = read_detections_table(self.stage_dir)

    def segment(self, view: View, object_ids: list[int]) -> list[Detection]:
        wanted = set(object_ids)
        dets = read_detections(self.stage_dir, view.scene_id, view.image_id)
        return [d for d in dets if d.object_id in wanted]
