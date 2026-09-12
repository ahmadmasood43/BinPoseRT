---
status: accepted
date: 2026-09-12
---
# Pipeline as a stage DAG with a content-addressed on-disk cache

Neural stages (CNOS segmentation, FoundPose / MegaPose coarse pose) run only on remote GPU machines,
while all geometry, fusion, confidence and evaluation work runs on a CPU laptop, and the ablation matrix
re-uses the same expensive upstream outputs under many downstream settings. We therefore model the
pipeline as `segment → coarse_pose → refine → associate → fuse → confidence → nbv → evaluate`, where each
stage is a pure function of its upstream artefacts, its config and a version string, writing to
`outputs/<dataset>/<split>/<stage>/<hash>/` and skipped when a `_SUCCESS` marker exists. GPU outputs are
rsync'd once; everything downstream is recomputed locally.

**Considered:** a monolithic `run_pose.py` per experiment (re-runs GPU inference for every ICP variant);
Snakemake / DVC (extra tooling, and the remote GPU steps would still be manual).

**Consequences:** stage interfaces are file formats (Parquet/JSON/PNG) as much as Python types; a stage
that reads global state or mutates inputs breaks caching and is a bug; the hash must include a stage
version string bumped on behaviour changes.
