---
status: accepted
date: 2026-09-12
---
# Python first; C++ only for profiled hot paths, after milestone Delta

The research plan recommends a C++17/20 geometry core with pybind11 bindings from the start. We decided
instead to build Foundations → Delta entirely in Python (NumPy, SciPy, Open3D, OpenCV) and to port only
the one to three geometry stages that profiling proves slow, as drop-in replacements behind the same
Python signatures. The project is a 16-week solo portfolio effort developed on a CPU laptop; a two-language
core would double build, test and cross-machine setup cost before a single experiment exists, while Open3D
already exposes ICP/GICP/raycasting in C++ under a Python API. Making the C++ work a *measured result*
(before/after latency on the update path) is also a stronger engineering signal than an up-front rewrite.

**Consequences:** `cpp/` does not exist until after Delta; every Python geometry function must have a
stable signature that a compiled implementation can satisfy; CUDA kernels are not planned.

**2026-09-23 Deployment status note.** After the Class A + ROI Python fixes (F1/F3/F4/F5/F5b/F7),
the two remaining hot paths — `registration_icp` (~66 ms) and `cast_rays` (~18 ms × 5 renders) —
are **already C++ inside Open3D**. No Python hot spot survives the fixes. The precondition of this
ADR is not met; `cpp/` is not created. This measurement *is* the finding for RQ-F (see
`docs/milestone_deployment_decision.md` §P1 and the Deployment Pareto figure).
