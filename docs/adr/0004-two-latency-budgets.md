---
status: accepted
date: 2026-09-12
---
# "Real-time" means two explicit, separately reported latency budgets

The zero-shot components we depend on (CNOS 0.1–1 s, FoundPose 0.1–0.6 s) cannot deliver a sub-200 ms
full pipeline, and a low average with one-second spikes is not real-time for a robot controller. We
therefore commit to two numbers: the **full pipeline** (`segment` → `confidence`, one Scene) is only
*measured* — stage-level median/p90/p95 and peak memory on one documented GPU machine, no target — while
the **update path** (`refine` → `associate` → `fuse` → `confidence` for one ObjectTrack given cached
detections and coarse hypotheses) has a **target p95 < 200 ms** in Python and < 50 ms after C++ ports.
The README states both in one sentence and never uses "real-time" without the number.

**Consequences:** the timing protocol (50 warm-up, 1000 timed, batch 1, CUDA synchronised) is frozen in
`configs/benchmark.yaml`; the accuracy–latency Pareto plot is a required final figure.
