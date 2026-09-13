"""Stage DAG with a content-addressed disk cache (D12, ADR-0002).

``segment → coarse_pose → [refine → associate → fuse → confidence → nbv] → evaluate``. Each stage is
a pure function of (upstream artefacts, its config, its version string) and writes to
``outputs/<dataset>/<split>/<stage>/<hash>/``; a ``_SUCCESS`` marker means "done, skip".
GPU stages are never executed here: the runner computes their directory, and if it is missing it
prints the adapter command that fills it.
"""

from binposert.pipeline.cache import StagePlan, stage_hash
from binposert.pipeline.runner import MissingGpuArtefact, Pipeline, RunResult
from binposert.pipeline.selection import Selection

__all__ = [
    "MissingGpuArtefact",
    "Pipeline",
    "RunResult",
    "Selection",
    "StagePlan",
    "stage_hash",
]
