---
status: accepted
date: 2026-09-12
---
# Continuous symmetries are discretised into one flat SymmetryGroup per object

Fusion, hypothesis dispersion, next-best-view scoring and angular-error reporting all need to compare
poses of symmetric objects. Rather than a second analytic code path that projects out continuous
rotational degrees of freedom, every ObjectModel gets one flat list of 4×4 transforms: BOP's
`symmetries_discrete` plus each `symmetries_continuous` axis sampled at 36 evenly spaced angles. All
consumers use one rule — align a pose to a reference by right-multiplying the symmetry element that
minimises SE(3) distance — before any averaging. This mirrors the BOP toolkit's own MSSD/MSPD treatment,
and the residual axial error (< 5°) is invisible to symmetry-aware metrics.

**Consequences:** alignment must happen in the object frame (right multiplication) and before conversion
to world-frame averaging; a naive quaternion or Lie-algebra mean of unaligned symmetric poses is a bug.
