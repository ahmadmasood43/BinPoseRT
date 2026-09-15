"""Stage DAG, content-addressed cache and run manifest (D12, ADR-0002).

Only ``artefacts``, ``cache`` and ``manifest`` are imported here so that estimator adapters can use
the artefact writers from their own environments. Stage implementations live in
:mod:`binposert.pipeline.stages` and the Hydra entry in :mod:`binposert.pipeline.run`.
"""

from binposert.pipeline.artefacts import (
    DetectionRecord,
    DetectionWriter,
    HypothesisRecord,
    read_detections,
    read_hypotheses,
    write_hypotheses,
)
from binposert.pipeline.cache import StageCache, StageRef, stage_hash
from binposert.pipeline.manifest import RunManifest

__all__ = [
    "DetectionRecord",
    "DetectionWriter",
    "HypothesisRecord",
    "RunManifest",
    "StageCache",
    "StageRef",
    "read_detections",
    "read_hypotheses",
    "stage_hash",
    "write_hypotheses",
]
