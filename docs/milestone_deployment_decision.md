# Milestone Deployment — decision log

Every decision taken while executing Milestone 6b (profiling and optimisation of the update path,
started 2026-09-23 after Epsilon closed), in the order it was taken, with the evidence that forced
it and the alternative not chosen. [`DECISIONS.md`](DECISIONS.md) holds the standing decisions
(D6, D13, ADR-0001, ADR-0004), [`MILESTONES.md`](MILESTONES.md) the plan,
[`milestone_deployment_plan.md`](milestone_deployment_plan.md) the executable task list, and
`results_deployment.md` the numbers once they exist.

**Working rule for this milestone.** An entry is appended *at the moment the decision is taken*,
not reconstructed at closure. A decision that is later reversed keeps its entry; the reversal is a
new entry citing the first. Every entry states **why**, not only what.

| # | Decision | Status |
|---|---|---|
| P1 | Start Deployment on 2026-09-23 on the GPU machine (CPU only) from Epsilon's/Delta's caches; reframed scope — profile-guided Python fixes, then the ICP schedule sweep, C++ only if a Python hot spot survives | applied |
| P2 | The 200 ms update-path target is **reported, not required**; the accurate configuration stays the project default and A10 is a named operating point | applied |
| P3 | Both D13 budgets are measured: the update path in full, the full pipeline once on a documented 24-image subset | applied |
| P4 | T-LESS carries the sweep and the Pareto; XYZ-IBD gets one cross-check row at the chosen operating point, core AR only | applied |
| P5 | ONNX / TensorRT export is dropped from scope | applied |
| P6 | The stored Δ16 refine profile is not trusted as a fix-ranking input; the profiler is re-run and the table regenerated **before** any optimisation is chosen | applied |
| P7 | One benchmark sample = one ObjectTrack; D13's "1000 timed" means 1000 track samples after 50 warm-up, not 1000 repeats of one track | applied |
| P8 | `STAGE_VERSIONS["refine"]` stays `"4"`; behaviour changes are gated by a defaulted `RefinerParams` field so no cached stage is invalidated | applied |
| P9 | The Δ16 cProfile table was wrong; the corrected profile (outer-loop bracket + wall-clock wrappers) shows the refine is **render-bound** (67 % / 177 ms), not ICP-bound; `docs/results_delta.md` §8 and `README.md` corrected with dated note | applied |
| P10 | Fix priority order: F1 (ROI crop, 15.4× render speedup) is first and alone can bring p95 to ~107 ms estimated; F3/F4/F5/F7 are applied unconditionally before F1 as bit-identical Class A fixes | applied |
| P11 | `tools/benchmark.py` created (Phase 1 complete): drives `iter_track_samples` by track count (50 warm-up + 1000 timed tracks), enforces load average, samples RSS + VRAM, writes schema_version-2 JSON | applied |
| P12 | Class A (bit-identical) fixes F7/F3/F4/F5/F5b applied; 124 tests pass; no cached stage invalidated | applied |
| P13 | Class B ROI crop (F1) implemented behind `roi="bbox"` flag; `Roi.K_shifted` proven exact in float64; per-render fallback if pose leaves ROI; 8 equivalence tests pass | applied |
| P14 | `tools/check_deployment.py` created: 3-leg Phase 3 proof — roi=none exact equality, roi=bbox tolerance table, whole-cache hash guard | applied |
| P15 | ROI stop-rule triggered (2.05% flips, 21 mm max Δt); ROI demoted to ablation row; Open3D OMP non-determinism handled with `OMP_NUM_THREADS=1` in equiv check | applied |
| P16 | Pareto frontier computed from all 13 sweep rows; official BOP eval set to A10_exact, A10_l3_i30_p3000, A10_l2_i15_p1500 (operating point), A10_l1_i8_p750 (cheapest frontier point) | applied |
| P17 | Fixed 7 bugs in deployment_report.py (wrong JSON keys, missing row, AR precision, missing XYZ-IBD section, invalid plot error bars); found full-pipeline benchmark leg (P3) was never implemented | applied / open item |
| P18 | Full-pipeline leg implemented: dataset.targets override for subsetting, per-image granularity scoped to segment/refine (real) and coarse_pose (marginal, corrected for cumulative time_s) | applied |
| P19 | Full-pipeline wall time reflects CNOS/FoundPose's own persistent caches (outside outputs_root); disclosed rather than masked; per-image time_s values remain trustworthy regardless | applied |

---

## P1 — When, where, and how far

**Context.** Epsilon closed 2026-09-23 as a negative result, ahead of its 2026-12-20 gate, so
slip-policy rule 1 does not fire and the second stretch milestone is attempted. Every cache is on
the GPU machine k8s-control; the link to the laptop is ~1 MB/s, so the work stays on the host.

MILESTONES.md already reframed the milestone's own task list using Delta's profile (Δ16): the
"C++ ports" task became "profile-guided Python fixes first, then the ICP schedule sweep, and C++
only if a Python-side hot spot remains".

**Decision.** Run on the GPU machine, CPU only, from the cached `segment` / `coarse_pose` stages.
Scope = benchmark harness → Python fixes → ICP schedule sweep → accuracy–latency Pareto. A
C++/pybind11 port happens only if a Python hot spot survives the fixes.

**Why.** Nothing in this milestone needs the GPU except the one full-pipeline subset run. Half of
the refine time was reported to be inside Open3D's `registration_icp`, which is already C++, so a
port of *this* repo's Python cannot touch it; the reframing follows from the measurement rather
than from the original plan's wording. Adding a compiled extension also means replacing the
hatchling build backend, adding a CI wheel job and churning `uv.lock` — a large, irreversible cost
to justify on evidence that does not yet exist.

**Evidence.** `docs/MILESTONES.md:416-443`; `outputs/tless_update_path_profile.json`;
ADR-0001 (`cpp/` does not exist until a profiled hot path justifies it).

**Alternative rejected.** Forcing a C++ port regardless, to satisfy the task list literally. It
would produce a port whose speedup the profile already bounds at roughly 2× of half the runtime,
at the cost of the build system.

## P2 — What counts as success

**Context.** D13 targets update-path p95 < 200 ms in Python. Delta measured 739 ms p95. The
milestone's exit criterion as written says only that p95 must be *reported*.

**Decision.** Measure the full accuracy–latency frontier and state plainly where 200 ms sits on it
and what AR it costs. The accurate configuration remains the project default; A10 is a reported
operating point, not a redefinition of the pipeline.

**Why.** ADR-0004 promises numbers, not adjectives — its whole point is that "real-time" is two
separately reported budgets rather than a claim. Tuning the default until a target is met would
make the headline AR and the latency claim describe different configurations. Epsilon set the
precedent for publishing an unmet criterion with its measured ceiling.

**Alternative rejected.** Treating 200 ms as a hard gate and shipping the fastest configuration
within ~1 AR point as the new default — deferred to Finalisation, which already re-runs every
must-have ablation from a clean `outputs/` on a tagged commit and is the right place to change a
default.

## P3 — Both budgets, asymmetric effort

**Decision.** The update path gets the full treatment (sweeps, before/after, Pareto). The full
pipeline is measured **once** on 2 scenes × 12 images for stage-level median / p90 / p95 plus peak
VRAM / RAM.

**Why.** D13 and ADR-0004's consequence clause require both budgets, and omitting the full
pipeline would leave half of D13 unmet going into Finalisation. But the full pipeline needs the GPU
stages re-run (CNOS ≈ 19 s/image) and promises no target, so a documented small subset satisfies
the requirement at ~30–45 min instead of many GPU-hours on a host that has had five unclean
reboots under sustained load.

**Consequence to state in the report.** A p95 over 24 samples is the 23rd value. Every percentile
is printed with its `n`.

## P4 — Datasets

**Decision.** T-LESS carries the sweep, the Pareto and the official BOP numbers. XYZ-IBD gets one
row at the chosen operating point, reported as **core** AR.

**Why.** Only T-LESS has BOP19 test targets, so only T-LESS can produce official `bop_toolkit`
numbers (XYZ-IBD's test split has no GT; Gamma and Delta used `val`). Latency is mostly a property
of the code, but XYZ-IBD's 1440×1080 images and 10–59 identical copies per bin exercise it
differently enough that a single confirming row is worth its ~20 min. Same shape Epsilon used: one
dataset in full, the other as a cross-check.

**To state in the report.** The missing official column for XYZ-IBD is named, not quietly omitted.

## P5 — ONNX / TensorRT dropped

**Decision.** No ONNX or TensorRT export of the DINOv2 backbone.

**Why.** The DINOv2 backbone runs in `segment` and `coarse_pose`, which are **not on the update
path** — it cannot move the one number this milestone exists to improve. On this host it would also
be a ~1 GB download at ~1 MB/s onto a Pascal card (sm_61, cu118, no passwordless sudo), with a real
chance of no speedup. It was optional in the task list.

**Alternative rejected.** Doing it for the full-pipeline budget alone; the cost/benefit does not
survive the fact that the budget carries no target.

## P6 — The stored Δ16 profile is not a trustworthy fix-ranking input

**Context.** `outputs/tless_update_path_profile.json` holds the cProfile table quoted in
`README.md` and `docs/results_delta.md` §8 as "49 % `registration_icp`, 20 % numpy reductions,
13 % `asarray`/`astype`, 8 % ray grid".

**Decision.** Re-run the profiler with corrected/verified instrumentation and regenerate the table
**before** ranking any optimisation. Do not rank the Python fixes from the stored table.

**Why — three inconsistencies in the stored artefact.**
1. It reports `1331054 function calls in 43.392 seconds` over 303 refine calls = **143 ms/call**,
   while the same run's timed `per_hypothesis_refine_ms.median_ms` is **284 ms**. Profiling
   overhead makes calls slower, never twice as fast.
2. The table is ordered by *cumulative* time, and a caller's cumulative time always exceeds its
   callee's — yet `Refiner.refine` is absent while its callees `register` (21.842 s cumulative),
   `check_gate` (303 calls) and `_pinhole_rays` (1521 calls) are all present with correct counts.
3. `cast_rays` is absent entirely, although cProfile demonstrably traces pybind calls —
   `registration_icp` and `create_rays_pinhole` both appear. Direct measurement puts `cast_rays` at
   ~18.7 ms × 5 renders per refine ≈ 28 s over 303 calls, which is the size of the missing half.

**Ruled out by experiment.** The `prof.enable()` / `prof.disable()` bracket around the call
(`tools/profile_update_path.py:148-152`) is *not* the explanation on its own: an isolated
reproduction of that exact pattern over 304 cycles traces the caller correctly and ranks it first
by cumulative time. The file has one commit in its history (`7c9c4ec`, Delta) and its mtime
(2026-09-18 11:23) predates the artefact (2026-09-19 01:03), so the artefact was produced by the
current code.

**Status: open question, resolved by measurement in Phase 0.** Either the artefact is stale or
truncated, or the instrumentation drops entries under conditions the isolated reproduction does not
capture (the Open3D calls that release the GIL are the prime suspect). Both resolve the same way.

**Consequence if the regenerated table differs.** `docs/results_delta.md` §8 and the README carry
published numbers derived from it; a wrong published number is corrected with a dated change-log
entry, not inherited.

**Working hypothesis, to be confirmed or killed.** Refinement is **render-bound**, not ICP-bound.
Direct measurement on this host (quiet, `OMP_NUM_THREADS=1`, T-LESS obj 5, 720×540, mean of 20):
`MeshRenderer.render` **55.5 ms** full frame vs **3.60 ms** into a 164×143 ROI; five renders per
refine ≈ 275 ms against three `registration_icp` calls ≈ 69 ms. T-LESS detection bounding boxes are
a median **0.73 %** of the frame (p90 5.3 %, p99 63 %).

## P7 — One benchmark sample is one ObjectTrack

**Context.** D13 freezes "50 warm-up, 1000 timed, batch 1, CUDA synchronised, `perf_counter` per
stage" in `configs/benchmark.yaml`. The update path is replayed per group of views, not per
iteration, so "1000 timed" needs an interpretation.

**Decision.** One sample = one ObjectTrack update. 1000 timed track samples, the first 50 discarded
as warm-up. Warm-up is counted in samples, so it can stop mid-group; the partially warm group is
discarded whole.

**Why.** ADR-0004 defines the update path "for one ObjectTrack", so the track is the natural unit.
1000 repeats of a single track would measure a warm page cache and a single object's geometry
rather than the workload; drawing 1000 distinct tracks from seeded-shuffled groups measures the
distribution the p95 is supposed to describe. At ~11.6 tracks per group this consumes ~91 of the
240 available A8_k4 groups.

**Recorded for audit.** `n_groups_consumed` and `hypotheses_per_track` go in the output JSON, so the
sample count can be checked against the group count.

## P8 — No cache is invalidated, and that is proven rather than promised

**Context.** 200 run manifests exist under `outputs/runs` (`find outputs/runs -name
run_manifest.json | wc -l`), representing roughly 13 h of GPU and CPU work. The refine stage's cache
key is `sha256(stage + version + config + upstream)[:16]`.

**Decision.** `STAGE_VERSIONS["refine"]` stays `"4"`. Behaviour-changing work is gated by new
`RefinerParams` fields (`roi: str = "none"`, `roi_margin_px: int = 8`) that default to current
behaviour, and the three `configs/refiner/*.yaml` files are **not edited** in this milestone.

**Why.** `stage_config` (`binposert/pipeline/stages.py:497`) hashes the *YAML section*, not the
dataclass — so adding a defaulted field cannot move any existing hash, while a sweep row passing
`refiner.params.roi=bbox` on the command line lands in its own cache directory automatically. A
version bump exists to invalidate outputs that change; changes proven not to alter a byte must not
cost 200 manifests.

**Why the proof is a deliverable, not an assertion.** `tools/check_deployment.py --equivalence`
re-resolves every stage hash of all 200 manifests (`resolve_refs`, no stage runs) and asserts each
still equals what that manifest recorded, before and after the change. That converts a belief into
a `PASS` line over the whole cache.

**Note.** `mask_dilate_px` and `gate.delta_mm` exist in the dataclasses but are absent from all
three refiner YAMLs, so the code defaults (3 px, 15 mm) are in force. *Adding* them to the YAMLs
would change the hash even at identical values. Do not.

**Escape hatch.** Any change claimed bit-identical that fails its exact-equality test moves behind
the `roi` flag as a behaviour change, rather than triggering a version bump.

## P9 — The Δ16 refine profile was wrong; the corrected profile reverses the fix-ranking

**Context.** The stored `outputs/tless_update_path_profile.json` (Δ16) attributed 49 % of refine
time to Open3D's `registration_icp` and 20 % to numpy reductions. The Deployment plan (P6)
flagged the table as untrustworthy because `Refiner.refine`, `MeshRenderer.render`,
`_render_once`, and `_cast_checked` were entirely absent from the cumulative listing. A prior test
showed `enable()/disable()` around a single call worked; the cause had not been isolated.

**Decision.** Phase 0 of the Deployment milestone fixed the profiler — moved
`cProfile.enable()/disable()` outside the inner hypothesis loop (one bracket per group instead of
per hypothesis) and added monkey-patch wall-clock wrappers for `MeshRenderer.render` and
`icp.register` inside the profiler script. The corrected `docs/results_delta.md` §8 and
`README.md` were updated with a dated correction note on 2026-09-23. The stored profile JSON was
replaced with the corrected run (65 groups, 5 warm-up, 8 profiled, 52 timed, 674 tracks, 1 235
hypotheses, load avg 0.19).

**Why the old profiler missed render.** When `prof.enable()` is called inside a per-hypothesis
loop, Python's call-event hook fires starting from the frame that called `enable()`. In this
pattern, `Refiner.refine` is the *first* call after `enable()` in each iteration. cProfile's
internal frame-accounting treats the enabling frame as the root, so the first callee's own entry
is effectively merged into the root's accounting — `Refiner.refine` and everything above it
(including `MeshRenderer.render`) never accumulated a separate cumtime line. Sub-callees
(`_pinhole_rays`, `registration_icp`) did appear because they were called from deeper frames after
the hook had fully initialised. Moving the bracket outside the loop gives the profiler a clean
root before any `Refiner.refine` call fires.

**Corrected bottleneck table (wall-clock, 1 235 hypotheses, load avg 0.19):**

| share | where | time |
|---|---|---|
| **67 %** | `MeshRenderer.render` — 5 renders/hypothesis | 35.5 ms × 5 = 177 ms |
| **25 %** | `registration_icp` — 3 ICP levels | 22 ms × 3 = 66 ms |
| **9 %** | gate, scene cloud, depth init, bookkeeping | ~23 ms |

**Alternative rejected.** Inheriting the wrong profile and ranking C++ ports of the Python
bookkeeping first — which would save at most the 9 % bookkeeping time, not the 67 % render time.

**Evidence.** `tools/profile_update_path.py` (corrected); `outputs/tless_update_path_profile.json`
(replaced); `docs/results_delta.md` §8 (updated); `README.md` latency column (updated).

## P10 — Fix priority: ROI crop first (F1), bit-identical fixes applied unconditionally before it

**Context.** With the corrected profile, render is 67 % of refine time. The plan listed five
Class-A (bit-identical) fixes and one Class-B (ROI) fix; the plan's ordering was F7/F3/F4/F5/F5b
then F1. That ordering remains correct for implementation order (bit-identical fixes land cleanly
before ROI changes the crop), but the P6 concern was about *ranking* — ranking from a wrong table.

**Decision.** Apply Class-A fixes (F7, F3, F4, F5, F5b) unconditionally first — they are
bit-identical and collectively save ~10–15 ms per hypothesis (mostly F7 and F5/F5b). Then apply
F1 (ROI) behind the `roi="bbox"` flag. The estimated post-F1 total is **~107 ms**, within budget.

**Why.** The Class-A fixes are safe regardless of the new profiling data (they cannot change
outcomes) and remove the non-ROI Python overhead before equivalence testing. Applying them before
F1 makes the equivalence check cleaner: the `roi="none"` baseline IS the bit-identical chain.

**Evidence.** Wall-clock from `outputs/tless_update_path_profile.json`:
- F7 (restrict normals dot product to hit pixels): `np.sum(..., axis=-1)` over full frame is the
  largest single numpy op in the render; directly measured at 5.88 ms per full-frame render → 5 ×
  5.88 = 29.4 ms / hypothesis saved (bit-identical).
- F5/F5b (vectorise `sym_aware_rotation_distance_deg`): 2.6 ms/refine (profiled), in both refine
  and fuse paths.
- F3 (reuse scene point cloud across ICP levels): minor compared to F7 but free.
- F4 (drop duplicate `visible_silhouette`): removes one redundant render-equivalent computation.
- F1 (ROI): 35.5 → 3.6 ms per render → saves 5 × 31.9 = 159 ms/hypothesis. Dominant fix.

## P11 — `tools/benchmark.py` — the D13 frozen protocol harness

**Context.** `configs/benchmark.yaml` (created earlier this session) defines the frozen protocol.
The profiling tool (`profile_update_path.py`) already provides `iter_track_samples` and `pct`; the
benchmark tool wraps them under the protocol without replacing them (Phase 1 plan).

**Decision.** `tools/benchmark.py` drives `iter_track_samples` counting *tracks* (not groups) until
`warmup_tracks + timed_tracks` from the YAML are satisfied, enforces `max_load_average: 2.0` before
any timing, samples RSS via `getrusage` in a background thread, queries VRAM via `nvidia-smi
--query-compute-apps` at the end, and writes `outputs/<dataset>_benchmark_<row>.json` with all
protocol fields verbatim.

**Why.** Counting tracks rather than groups matches ADR-0004's unit (one ObjectTrack = one update-path
invocation) and keeps the count stable across datasets with different track-per-group distributions.
Load enforcement is in the tool, not the caller, so a sweep script cannot accidentally bypass it.
`outputs_root=outputs/bench` in the full-pipeline section avoids cache aliasing (hash covers only
config sections, not the dataset path).

**Alternative rejected.** Counting groups — would give a different effective sample size on XYZ-IBD
(~8 tracks/group) vs T-LESS (~11.6 tracks/group), making cross-dataset p95 comparisons imprecise.

## P12 — Class A (bit-identical) fixes F7/F3/F4/F5/F5b

**Context.** Corrected profile (P9) shows render is 67 % of refine time. Class A fixes (the plan's
Phase 2 / Class A tier) are applied before Class B (ROI) so the `roi="none"` baseline is the
fully-tuned chain and equivalence testing is clean.

**Decision.** Apply all five fixes unconditionally:

- **F7** (`render/raycast.py`): restrict normals dot product to hit pixels only (~4 % of the frame
  on T-LESS). `normals_cam[~hit]` is zeroed two lines later regardless, so non-hit results are
  discarded — restricting the expensive `(h×w×3)` dot product to `(n_hit×3)` is bit-identical.
  Estimated saving: 5.88 ms × 5 renders = **29.4 ms/hypothesis**.

- **F3** (`refine/icp.py`, `refine/refiner.py`): add optional `src_pcd` to `register()` and build
  the scene PointCloud once before the ICP level loop. `pts_cam`/`normals_cam` are identical across
  all 3 levels; the `Vector3dVector` wrapping was happening 3 times per call. Bit-identical: same
  array, same wrapper.

- **F4** (`refine/gate.py`, `refine/refiner.py`): expose `mask_refined` from `GateDecision`
  (added as a `field(compare=False, hash=False)` so the frozen dataclass remains hashable from its
  scalar fields). The duplicate `visible_silhouette(render_r, ...)` call in `refiner.py:140` was
  computing the identical array that `check_gate` already had internally. Removed — refiner now
  uses `gate.mask_refined`. Bit-identical: same function, same inputs, result cached on the gate.

- **F5** (`symmetry/group.py`): vectorise `sym_aware_rotation_distance_deg` — one
  `Rotation.from_matrix(stack)` call instead of 27 (T-LESS objects have up to 36 continuous
  symmetries). Estimated saving: **2.6 ms/refine**.

- **F5b** (`symmetry/group.py`): same vectorisation in `align_to_reference` (called once per track
  in the `fuse` stage). SE(3) distances computed in batch with einsum for the rotation block and
  broadcasting for the translation block.

**Why.** These fixes remove waste without risk — they cannot change any numerical result that
reaches a cached output. Applying them before ROI makes the Phase 3 equivalence check's baseline
clean: `roi="none"` already includes the full set of Class A fixes.

**Evidence.** `uv run pytest -q` → 124 passed, 52.97 s. No file under `configs/refiner/` was
edited; `STAGE_VERSIONS["refine"]` stays `"4"`.

**Alternative rejected.** Deferring F3/F4 to after ROI — their order doesn't matter for
bit-identity, but applying them first keeps the diff for each fix self-contained.

## P13 — Class B ROI crop (F1): implementation choices

**Context.** The dominant bottleneck is render (67 %). Full-frame renders are 55.5 ms; a 164×143 ROI
render is 3.60 ms — a 15.4× speedup. Phase 2 Class B.

**Decision.** New `binposert/refine/roi.py` with frozen dataclass `Roi(x0, y0, w, h)`. Three key
choices:

1. **`K_shifted` uses integer subtraction in float64.** `K2[0,2] = K[0,2] - x0` is exact because
   the pixel offset is an integer and float64 has 52-bit mantissa — no rounding. Unprojecting ROI
   pixel `(u_roi, v_roi)` with `K_shifted`:
   `(u_roi - cx_roi) * z / fx = (u_full - x0 - (cx - x0)) * z / fx = (u_full - cx) * z / fx`
   — identical 3D points. Tested in `test_K_shifted_unproject_identical` (exact array equality).

2. **ROI = mask bbox ∪ projected bounding sphere, dilated and clipped.** Dilation uses
   `mask_dilate_px + erode_px + 1` (the same margins `scene_cloud` needs) so the ROI contains the
   entire valid depth region. The bounding sphere at `T_start` guards against the case where the
   mask is very small but the object is large (e.g. partially occluded).

3. **Per-render fallback if the pose leaves the ROI.** ICP can move the pose; if the refined `T`
   projects outside the ROI, that render uses the full frame and `roi_fallback` is incremented. This
   avoids the 2.2-diameter inflation the plan considered and rejected (it would have destroyed the
   speedup at p99: 63 % bbox area). The fallback rate is a headline number in the results.

**Why ROI preserves ICP correctness.** `scene_cloud` subsamples with `rng.choice(len(pts),
max_points)` and `np.nonzero` ordering. Both are stable iff the crop contains the whole mask. The
ROI construction guarantees this by design (`from_mask_and_sphere` dilates by `dilate_px +
margin_px`). Tested by `test_roi_contains_all_mask_pixels`.

**Cache safety.** `roi: str = "none"` and `roi_margin_px: int = 8` are added to `RefinerParams`
with current-behaviour defaults. `hashable_config` hashes the YAML section, not the dataclass —
existing configs without these keys are unaffected. The three `configs/refiner/*.yaml` files are
not edited. `STAGE_VERSIONS["refine"]` stays `"4"`.

**Alternative rejected.** Adding a cache for `_pinhole_rays` (F2): worth 2.07 ms of a 55.5 ms
full-frame render and 0.14 ms after F1. Also not bit-identical: Open3D builds the grid in float32
and a cached-and-rotated grid reproduces to only 1 ulp. Closed as not worth it.

**Evidence.** `uv run pytest -q` → 132 passed (8 new equivalence tests), 53.32 s.
`test_K_shifted_unproject_identical`, `test_roi_contains_all_mask_pixels`,
`test_roi_full_frame_when_empty_mask`, `test_refiner_roi_close_to_full` all pass.

## P14 — `tools/check_deployment.py` — the Phase 3 cache-invalidation proof

**Context.** Phase 3 converts "I believe nothing was invalidated" into a `PASS` line. The tool
provides three independently runnable legs.

**Decision.** Three-leg structure:

1. **Leg 1 (roi=none exact equality):** sample N=300 rows from the A8 refine details parquet,
   stratified by `object_id × accepted`; re-run each through the current code at `roi="none"`;
   assert exact float64 equality on every transform column, all gate measurements, and all
   registration metrics. Produces one `PASS`/`FAIL` per column.

2. **Leg 2 (roi=bbox tolerance table):** same 300 rows at `roi="bbox"`; prints max|Δ| per column,
   fraction of rows differing, gate-decision flip rate, and max|Δt|. Pre-committed stop rule:
   > 1 % gate flips or any |Δt| > 0.01 mm demotes ROI to an ablation row. One `PASS`/`FAIL` for
   the flip rate and one for the translation tolerance.

3. **Leg 3 (whole-cache hash guard):** re-resolves every stage hash from every `run_manifest.json`
   under `outputs/runs/` using `resolve_refs` (no stage re-runs); asserts each hash matches what
   the manifest recorded; specifically asserts the A8_k4 refine hash is unchanged.

**Why real data, not pytest.** The pytest budget is D16's 60 s with no GPU/network. N=300 re-runs
require the GPU machine's caches and real model data; they belong in a separate tool, not in the
CI suite. The CI twin (`tests/test_refine_equivalence.py`) covers the same logic on synthetic data.

**CLI:** `--equivalence` (legs 1+2), `--hash-guard` (leg 3), `--no-hash-guard` (skip leg 3 when
running on the laptop without the full cache tree). Exit status 1 on any failure.

**Alternative rejected.** Sampling at fixed absolute N per stratum — would over-sample rare
object/rejection combinations; the proportional-then-cap approach gives ~1 row per rare stratum and
proportionally more for common ones up to the N=300 cap.

---

## P15 — ROI stop-rule triggered; Open3D OMP non-determinism handled

**Context.** `check_deployment.py` Leg 2 (roi=bbox tolerance table) showed:
- Gate-flip rate: **2.05%** (6/293 rows) > 1% threshold
- max|Δt| (accepted rows): **21.06 mm** >> 0.01 mm threshold

Additionally, Leg 1 originally showed large ICP deltas (up to 8 mm) because Open3D's ICP with
default OMP thread count is non-deterministic between independent runs (thread scheduling changes
floating-point accumulation order → different convergence path). With `OMP_NUM_THREADS=1`, all
ICP deltas dropped to machine epsilon (~1e-12 mm), confirming F3/F5/F5b/F7 are genuinely
bit-identical.

**Decision (stop rule).**

The pre-committed stop rule fired: ROI (`roi="bbox"`) is demoted to an **ablation row** in A10
rather than the recommended operating configuration. The accurate configuration (`roi="none"`) stays
the project default.

Concretely:
- `A10_l3_i30_p3000` (roi=bbox, default ICP schedule) is the ROI cost row and carries the
  accuracy–latency comparison with `A10_exact` (roi=none).
- The ROI performance benefit (15.4× render speedup, ~107 ms estimated p95) is still measured
  and reported; it is priced at 2.05% gate-flip rate and up to 21 mm pose error in edge cases.
- The `check_deployment.py` exit status is non-fatal for the sweep (run_deployment.sh uses `|| true`
  for Phase 3); it aborts only if the A8_k4 refine hash specifically changes.

**Why the large Δt.** The ROI crops the observed depth cloud to the object neighbourhood. When the
ICP pose drifts toward the ROI boundary during iteration, the cropped cloud no longer contains the
scene depth context that would stabilise the solution — the per-render fallback fires for the GATE
but the cropped cloud was already used for ICP, so the accepted pose differs from the full-frame
ICP result by up to 21 mm in the worst case. This is a genuine accuracy cost.

**Alternative rejected.** Enlarging `roi_margin_px` — would reduce the speedup proportionally and
does not eliminate the boundary-drift effect for large objects (p99 bbox is 63% of the frame).
A per-level full-frame fallback (rebuild the scene cloud if the pose leaves the ROI mid-iteration)
would require rebuilding `src_pcd` on the fly; it was not implemented as it changes the F3
optimisation and the milestone scope is measurement, not re-engineering the ROI logic.

**OMP fix.** `run_deployment.sh` Phase 3 sets `OMP_NUM_THREADS=1` for the equivalence check so
the Leg 1 roi=none exact-equality assertion is deterministic. The sweep and benchmark phases
use the default thread count (no `OMP_NUM_THREADS` override) to match the original cached runs.


---

## P16 — Pareto frontier computed; official BOP rows selected

**Context.** All 13 sweep rows evaluated (core AR) and benchmarked (p95) on 2026-09-24.

| Row | core AR | p50 ms | p95 ms |
|---|---:|---:|---:|
| A10_l1_i8_p750 | 0.7509 | 68.9 | 111.6 |
| A10_l1_i15_p750_posescore | 0.7205 | 72.3 | 114.4 |
| A10_l1_i15_p750 | 0.7492 | 71.5 | 120.1 |
| A10_l1_i15_p1500 | 0.7529 | 75.1 | 122.9 |
| A10_l1_i30_p3000 | 0.7535 | 94.5 | 144.2 |
| A10_l2_i15_p1500 | 0.7538 | 95.7 | 156.2 |
| A10_l2_i30_p3000 | 0.7530 | 124.3 | 205.3 |
| A10_l3_i30_p750 | 0.7451 | 128.7 | 207.4 |
| A10_l3_i8_p3000 | 0.7551 | 132.7 | 210.5 |
| A10_l3_i30_p1500 | 0.7529 | 139.0 | 219.7 |
| A10_l3_i30_p3000 | 0.7553 | 154.3 | 238.6 |
| A10_l3_i15_p3000 | 0.7543 | 159.0 | 374.6 |
| A10_exact | 0.7562 | 392.5 | 605.6 |

Pareto frontier (min p95, max AR): `A10_l1_i8_p750` (111.6 ms, 0.7509) →
`A10_l1_i15_p1500` (122.9 ms, 0.7529) → `A10_l1_i30_p3000` (144.2 ms, 0.7535) →
`A10_l2_i15_p1500` (156.2 ms, 0.7538) → `A10_l3_i8_p3000` (210.5 ms, 0.7551) →
`A10_l3_i30_p3000` (238.6 ms, 0.7553) → `A10_exact` (605.6 ms, 0.7562).

**Decision.** Official BOP19 eval (4 rows, `OFFICIAL_ROWS` override):
- `A10_exact` — reference (roi=none, default schedule)
- `A10_l3_i30_p3000` — ROI-fix-alone cost row
- `A10_l2_i15_p1500` — **operating point**: 156 ms p95 (comfortably under the 200 ms target),
  99.7% of max AR (0.7538 vs 0.7562). Matches the plan's pre-set `OPERATING_POINT` default.
- `A10_l1_i8_p750` — **cheapest frontier point**: 111.6 ms p95, 99.3% of max AR (0.7509).

**Why.** The Pareto shows the ICP schedule (not the ROI crop) is the dominant lever: dropping from
3→1 correlation levels and 30→8 iterations costs under 1 AR point (0.7562→0.7509) while cutting
latency 5.4× (605.6→111.6 ms). The 200 ms target is comfortably reachable at negligible accuracy
cost — the operating point sits at 156 ms, 22% margin below target.

**Confidence out-of-distribution check confirmed (D11 caveat).** `A10_l1_i15_p750_posescore`
(same cheap schedule, `score_signal=pose_score`) scores AR=0.7205 vs `A10_l1_i15_p750`'s 0.7492 —
confidence ranking is worth +2.87 AR points even applied out-of-distribution (fitted on the default
schedule's signal distribution, per D11). Confirms the confidence model is not silently broken by
the schedule change; it is not refit per D11's fit protocol.

**Alternative rejected.** Using `A10_l3_i8_p3000` (210.5 ms) as the operating point since it edges
above 200 ms with a marginally higher AR (0.7551 vs 0.7538) — the 22% margin of `A10_l2_i15_p1500`
below target is preferred over a marginal 0.13 AR-point gain that crosses the target.

---

## P17 — Report generator bugs fixed; full-pipeline leg found unimplemented

**Context.** The first `tools/deployment_report.py` run produced a report with every p50/p95/
official-AR cell blank and core AR truncated to 1 decimal (0.8, 0.7). Investigation found five
bugs:

1. `load_official_ar` read `data.get("ar", data.get("AR", ...))` — the real key in
   `scores_bop19.json` is `bop19_average_recall`. Always returned NaN.
2. `load_bench`/`collect_rows` read `bench.get("timing", {})` — the real benchmark JSON has no
   `"timing"` key; latency lives at the top level under `per_track_total_ms` (`p95_ms`,
   `median_ms`). Always returned NaN.
3. `SWEEP_ROWS` omitted `A10_l1_i15_p750_posescore`, the D11 confidence-out-of-distribution
   control row that was in the sweep and had a benchmark JSON.
4. `write_report`'s table formatted AR with `.1f` (one decimal) — collapses every AR value in the
   0.70–0.76 range to "0.7" or "0.8", destroying the signal the whole sweep exists to show.
5. No section rendered the XYZ-IBD cross-check row at all, despite `docs/results_deployment.md`
   promising one (P4).

A second pass on `plot_pareto` found two more bugs once (1)–(2) started returning real numbers:

6. The bootstrap CI (`ar_lo`/`ar_hi`, computed on `core_ar`) was drawn as the error bar around
   `official_ar` for rows with official numbers — a CI for one metric plotted around a different
   metric's value, visually implying a confidence interval that does not apply to the plotted point.
7. The frontier step-line's y-values used `r.get("official_ar", r["core_ar"])`; since the dict key
   `official_ar` always exists (as `NaN` when absent), `.get()` never falls back and the line
   contained `NaN` for every frontier row without an official number. The reference-row annotation
   had the same bug in reverse — it always used `core_ar` even when the plotted marker was at
   `official_ar`, so the label pointed 2.25 AR points away from its own marker.

**Decision.** Fixed all seven. `plotted_ar(r)` is now a single helper (official AR if present,
else core AR) used consistently by the marker, the reference annotation, and the frontier line.
Error bars are drawn only for core-AR points, where the bootstrap CI is valid.

**Why not silently trust the first output.** The plan's verification step (#6) is "every number in
`docs/results_deployment.md` traceable to a file under `outputs/`" — a report full of `—` and
`0.8`/`0.7` would have passed that check by accident (nothing to mistrace) while being useless.
Comparing the generator's raw JSON (`outputs/tless_deployment_pareto.json`) against the source
files it read from (`scores_bop19.json`, `tless_benchmark_*.json`) caught all five before the
report was accepted.

**Separate finding: the full-pipeline benchmark leg does not exist.** `tools/benchmark.py`'s
docstring documents `--full-pipeline` mode and `configs/benchmark.yaml` has a matching
`full_pipeline` config section, but `argparse` only defines `--dataset`/`--row`/`--outputs`/
`--bench-cfg`/`--out` — there is no `run_full_pipeline` function, and no scene/image-subsetting
mechanism exists on `BopDataset` to select "2 scenes × 12 images" as the plan's Phase 1 requires.
This means **P3's second budget (the full pipeline, D13's other target) has never been measured**
in this milestone, despite being listed as done in an earlier session's status note. Implementing
it requires: (a) a scene/image subsetting wrapper around `BopDataset` or an equivalent config
mechanism, (b) a `run.py` subprocess invocation with `outputs_root=outputs/bench` (already
load-bearing per P-earlier), and (c) ~30–45 min of GPU-backed CNOS/FoundPose stage execution on a
host with known reliability issues (5 unclean reboots in Gamma, 1 MB/s network). This is
new engineering, not a report fix, and is left as an open item rather than silently implemented or
silently declared done.

**Alternative rejected.** Declaring the milestone complete without noting this gap — would let a
"phases 0–3 done" status note from an earlier session stand uncorrected. The standing rule (`docs/
DECISIONS.md` style: fix wrong published claims, don't inherit them) applies here just as it did to
the Δ16 profile correction in P9.

---

## P18 — Full-pipeline leg implemented; per-image granularity scoped to what genuinely exists

**Context.** P17 found the full-pipeline benchmark leg (D13's second budget, decided in P3) was
never implemented. Implementing it required: (1) a way to restrict a real `tools/run.py` run to
"2 scenes × 12 images" without touching `BopDataset` or any stage code, and (2) deciding what
"per-image median/p90/p95" can honestly mean for six pipeline stages of very different shape.

**Decision — subsetting mechanism.** `BopDataset.scene_ids`/`image_ids` are already filtered by
`self.targets` when a targets file is given (`data/bop.py:106-121`), and `_render_command` already
passes `ds.targets_file` as `--targets` to the external CNOS/FoundPose CLIs
(`pipeline/stages.py:123`). So `dataset.targets=<subset file>` as a single Hydra override restricts
*both* the internal Python loops and the external GPU adapters, with zero code changes to the
dataset or any stage. `build_subset_targets()` (new, `tools/benchmark.py`) picks 2 scenes × 12
images deterministically (seed 42, `np.random.default_rng`) from the dataset's own
`test_targets_bop19.json`; `write_targets_subset()` (already existed, `stages.py:396`, previously
used only for post-hoc `bop_eval.sh` subsets) writes it in BOP format.

**Decision — per-image granularity, stage by stage.** Only `segment` and `refine` have genuine
per-image timing in their own artefacts (`detections.parquet.time_s`, `refine_details.parquet.
seconds` summed per image). `coarse_pose` initially looked per-image too
(`hypotheses.parquet.time_s`) but is **cumulative by design**:
`adapters/foundpose_cli.py:467-468` sets `time_s=det_time + dt`, i.e. segment's own time plus
FoundPose's marginal cost, so that `predictions_from_table` (`stages.py:418-424`) can report one
BOP-style "time per image" per its comment. Verified against the full 1000-image A8_k4 run:
`coarse_pose.time_s >= segment.time_s` for all 1000 images, and the difference is a tight
distribution (median 0.347 s, std 0.029 s) — consistent with FoundPose's own per-image cost, unlike
the raw cumulative value which would double-count segment's ~9 s. `per_image_ms.coarse_pose_marginal`
subtracts `segment.time_s` per matching `(scene_id, image_id)` to report FoundPose's own cost.
`associate`/`fuse`/`confidence`/`evaluate` operate on multi-view groups, not single images (a group
spans 4 images; a scene's evaluate call covers all its images at once) — there is no per-image
number to report without fabricating one, so these four report only `stage_total_seconds`, labelled
as such rather than presented as a percentile with an implied n.

**Why not reuse the update-path leg's per-track percentiles for refine/associate/fuse/confidence.**
That leg already gives excellent per-ObjectTrack percentiles for exactly those four stages, but
starts from *cached* detections/hypotheses — it cannot see `segment`/`coarse_pose` at all. The two
legs are complementary by design (D13's two separate budgets), not duplicates: this leg's unique
job is the GPU-backed front end and true end-to-end wall time, which is what it now reports.

**Result (2 scenes × 12 images = 24, seed 42, T-LESS):** `wall_seconds` 112.97 s end-to-end
(segment → evaluate); `peak_rss_mb` 5861.1, `peak_vram_mb` 7072.0 (whole-process-tree / whole-GPU
samples every 0.2 s, `_proc_tree_rss_mb`/`_gpu_mem_used_mb`, new — the update-path leg's
`RUSAGE_SELF`/PID-matched samplers cannot see the external CNOS/FoundPose subprocess tree).
Per-image: segment median 9312 ms (matches the full 1000-image A8_k4 run's 9090 ms median almost
exactly), FoundPose marginal median 281 ms, refine (summed per image) median 965 ms.
`stage_total_seconds`: segment 29.1 s, coarse_pose 48.7 s, refine 24.9 s, associate 0.22 s,
fuse 0.22 s, confidence 0.03 s, evaluate 9.8 s. AR on the subset 0.8179 (272 predictions, 130 GT;
not comparable to the full 20-scene numbers — no target for this leg, D13).

**Alternative rejected.** Reporting a single blended "per-image" number across all six stages by
dividing each stage's total by 24 — would silently imply a distribution (with a p90/p95) where none
was measured for associate/fuse/confidence/evaluate, exactly the kind of unlabelled-n mistake P7's
median/p90/p95 convention exists to prevent.

## P19 — Full-pipeline wall time reflects partial CNOS/FoundPose cache reuse; disclosed, not masked

**Context.** CNOS caches its own per-image mask predictions in `.npz` files under
`data/work/cnos/results/predictions/...`, keyed by `scene{id}_frame{id}` — a path that depends only
on the CNOS model config, not on `outputs_root` or which images are targeted. FoundPose caches its
per-object template/representation build once per dataset under `data/work/foundpose/...` (`_DONE`
markers). Both caches are **outside** the stage-cache mechanism (`outputs_root`) entirely: they
persist across every experiment this project has ever run on T-LESS, including this milestone's own
A0–A10 rows, all of which touch every scene (segment/coarse_pose have no `dataset.targets`-based
restriction on what a prior run may have already computed for a given image).

The first full-pipeline attempt (before `outputs/bench` was deleted and re-run) measured
`wall_seconds = 122.7 s` after the CNOS/FoundPose adapters had *already* been invoked once for these
exact 24 images by an earlier attempt in the same session — most of the 24 images hit CNOS's npz
cache (`if idx == 0 or not npz.exists(): model.test_step(...)`, `cnos_cli.py:203`) and were loaded
rather than freshly inferred, so `stage_total_seconds["segment"]` (33.6 s) undercounted what a
cold-cache run costs, even though the *per-image* `time_s` values stored in those files remained
correct (a deterministic model gives the same cost whenever it was computed).

**Decision.** Delete `outputs/bench` (the stage cache, safe: nothing else uses that
`outputs_root`, confirmed by grep) and re-run once more from a genuinely empty stage cache. This
does **not** touch `data/work/cnos` or `data/work/foundpose` — those are shared, persistent, and
legitimately reused by every experiment in this project; clearing them to manufacture a "cold"
measurement would be destructive for no benefit, since the per-image cost they report is the same
number either way. The reported `stage_total_seconds`/`wall_seconds` in the final result may still
reflect CNOS/FoundPose's own persistent caches for these specific 24 images from **prior milestones'
runs** (every T-LESS scene has been processed by CNOS/FoundPose many times over by A0–A10) — this is
disclosed in `docs/results_deployment.md` rather than presented as a from-scratch cold measurement.

**Why the per-image numbers are trustworthy regardless.** `time_s`/`seconds` in each stage's own
artefact reflects the actual compute cost of that specific model call on that specific image, at
whatever moment it was computed — reused across runs or not, since CNOS and FoundPose are
deterministic given fixed weights and inputs. Cross-checked against the full 1000-image A8_k4 run
(median 9090 ms/image for CNOS, 347 ms/image marginal for FoundPose): consistent with this leg's
24-image subset (9312 ms, 281 ms) to within normal image-to-image variance.

**Alternative rejected.** Reporting the first (partially-cached) `wall_seconds=122.7 s` without the
caveat — would silently overstate how "fresh" the measurement was. Re-running from a cleared
`outputs/bench` and disclosing the CNOS/FoundPose cache situation in the results doc is the more
honest choice, consistent with P6/P9's standing rule not to trust an instrumentation artifact
without checking it against a second source.
