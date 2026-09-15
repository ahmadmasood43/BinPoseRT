"""PoseEstimator that replays a ``coarse_pose`` stage directory written by an adapter."""

from __future__ import annotations

from pathlib import Path

from binposert.pipeline.artefacts import read_hypotheses, read_hypotheses_table, read_stage_info
from binposert.types import Detection, PoseHypothesis, View


class CachedPoseEstimator:
    def __init__(self, stage_dir: str | Path) -> None:
        self.stage_dir = Path(stage_dir)
        info = read_stage_info(self.stage_dir)
        self.name = str(info.get("config", {}).get("name", "cached"))
        self.table = read_hypotheses_table(self.stage_dir)

    def estimate(self, view: View, detections: list[Detection]) -> list[PoseHypothesis]:
        wanted = {d.detection_id for d in detections}
        hyps = read_hypotheses(self.stage_dir, view.scene_id, view.image_id)
        return [h for h in hyps if h.detection_id in wanted]
