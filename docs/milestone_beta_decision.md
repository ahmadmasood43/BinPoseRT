# Milestone Beta — decision log

Every decision taken while executing Milestone 3 (depth-based refinement, 2026-09-15 → 2026-09-16), in
the order it was taken, with the evidence that forced it and the alternative that was not chosen.
[`DECISIONS.md`](DECISIONS.md) holds the *standing* decisions (D8 was revised from this log);
[`MILESTONES.md`](MILESTONES.md) the plan; [`results_beta_tless.md`](results_beta_tless.md) the numbers.
Numbers below are T-LESS BOP19 unless stated; *core* = `binposert.evaluate`, otherwise official
`bop_toolkit`.

| # | Decision | Status |
|---|---|---|
| B1 | Tune on val scenes {1, 6, 11, 16}, report on all 20, check on the other 16 | applied |
| B2 | Run Beta on the GPU machine's CPU cores, not the laptop | applied |
| B3 | Raycast single-threaded; one BLAS/OpenMP thread per worker process | applied (D8) |
| B4 | Bump the evaluate stage version; re-evaluate A1 (and A5 for A5r); leave A0's official number | applied |
| B5 | Split the `no_depth` rejection into `no_depth` / `depth_window` | applied |
| B6 | Add a depth initialisation of the translation before ICP | applied (D8) |
| B7 | Apply the depth initialisation always (no shift threshold), ≥ 50 overlapping pixels | applied |
| B8 | Displacement caps measured from the ICP start pose; IoU-drop baseline stays the coarse render | applied (D8) |
| B9 | Judge the gate per hypothesis with MSSD + MSPD recall fractions; two precision definitions | applied |
| B10 | Sweep silhouette and fitness thresholds too, not only α and β; keep the default unless ≥ 0.1 pt better | applied |
| B11 | One gate for all variants: α 0.6 d, β 90°, silhouette checks off (signals only), fitness ≥ 0.3 | applied (D8) |
| B12 | Point-to-plane is the default registration variant | applied |
| B13 | `GateParams` code defaults stay the D8 design values; tuned values live in configs | applied |
| B14 | Read "accepted worse" as wrong-instance detections; `broke` is the regression count | applied |
| B15 | Keep the D8-default rows and report on disk as provenance | applied |
| B16 | Deliver the stratified results as tables *and* a figure; commit downscaled figures | applied |
| B17 | A5 with refinement is a new row `A5r`; A5 stays the Alpha comparator | applied |
| B18 | Wait on files a job writes, never on `pgrep -f` of the job's own name | applied (process) |
| B19 | Commit Beta as four themed commits on `main`; do not push | applied |
| B20 | Do not correct the ~2.5 mm camera-y offset ICP converges to; record it, calibrate GT-free in Gamma | applied |

---

## B1 — Validation split for the gate thresholds

**Context.** D8 says α and β are "tuned on T-LESS val". BOP's T-LESS has no val split; the 20 test
scenes are all there is, and Alpha's numbers are on all 20.

**Decision.** Scenes {1, 6, 11, 16} (943 hypotheses, ~17 %) are the val set for every threshold choice;
the choice is then re-scored on the other 16 scenes and both numbers are reported; the headline AR is
on all 20 scenes, as in Alpha, and the doc says so.

**Not chosen.** Reporting only on the 16 held-out scenes (cleaner, but not comparable with Alpha's
A0/A1/A5 numbers or with the BOP leaderboard); tuning on all 20 (leaks). The held-out check is what
makes the all-20 headline defensible.

## B2 — Where Beta runs

**Context.** The plan says "laptop"; the Alpha caches (CNOS, FoundPose, MegaPose outputs, 18 GB of
templates) were never rsync'd off the GPU machine (~1 MB/s link).

**Decision.** Run every Beta stage on the GPU machine's 16 CPU cores from the caches in place; nothing
in Beta needs the GPU. Tests still run CPU-only (< 60 s) so the laptop rule holds in substance.

## B3 — Rendering is single-threaded; worker pools pin one thread each

**Context.** The first A2 run had a load average of 205 on 16 cores (15 workers × ~90 OpenMP/Embree
threads). After pinning `OMP_NUM_THREADS=1` in the workers, the stage crashed inside numpy on
`normals_cam[~hit] = 0.0` with impossible indices (a valid column index with bit 17 or 18 set).
Isolated with a stress test (15 processes, each render checked against a re-render):

| `cast_rays` | renders | exceptions | silently different renders |
|---|---|---|---|
| default (all threads) | 1751 | 126 | 5 |
| `np.where` instead of boolean indexing | 1897 | 0 | 6 |
| `nthreads=1` | ~1800 | 0 | 0 |

Pure numpy under the same load: 26 k boolean assignments, 0 errors. Single process: 0 errors either
way. So Open3D 0.19's parallel raycaster corrupts its output only under multi-process load, and the
`np.where` rewrite would have hidden it.

**Decision.** `MeshRenderer` and `render_scene` cast with `nthreads=1` (13 ms vs 5 ms per 720×540
render); `binposert/pipeline/pool.py` creates spawn pools with `OMP_NUM_THREADS=1` and the BLAS
equivalents. Parallelism is over scenes, one process each. Documented under D8's renderer bullet.

**Not chosen.** Pinning the Open3D version or filing upstream first (no time in the milestone; the
finding is recorded with numbers so it can be revisited); keeping parallel raycasting for single-
process use (a silent-corruption path that only shows under load is not worth 8 ms).

## B4 — Evaluate stage version bump; which rows to re-evaluate

**Context.** The evaluate stage renders for VSD; B3 changes render output in rare cases, and D12 says a
stage's output is a function of its version.

**Decision.** `evaluate` 6 → 7. A1 re-evaluated (35.18 core vs 35.2 before: unchanged); A5 re-evaluated
when it became the *before* row of A5r. A0's official 59.2 stays as reported by Alpha — the official
number does not depend on our renderer, and the core number moved by < 0.1 on the rows that were
re-run.

## B5 — Rejection reason taxonomy

**Context.** 1138 of 5452 A2 hypotheses (21 %) were rejected as `no_depth` while their masks had full
depth coverage: nothing survived the ±1.5 d depth window around a coarse z that was up to 5 m off.

**Decision.** `no_depth` = the mask has no valid depth; `depth_window` = depth exists but none within
the window of the start pose. Both are *pre-gate* reasons (`PRE_GATE_REASONS`, with `no_overlap` and
`icp_failed`) that a threshold sweep cannot re-decide; the analysis keeps them apart from the gate's
own reasons.

## B6 — Depth initialisation of the translation

**Context.** A2 with the design as planned: 58 % of hypotheses rejected (21 % `depth_window`, 13 %
`icp_failed` with zero correspondences, 17 % translation cap with fitness 0.9). Against GT on a 1/4
sample (1371 hypotheses): coarse |Δz| median 25 mm, 75th 99 mm, 95th 767 mm; 52 % beyond 0.25 d.
Local ICP with a 0.15 d search radius cannot start from there, and when it can, the move exceeds the
cap. This is Alpha's "right in the image, wrong in depth" (MSPD 82 vs VSD/MSSD ≈ 50) made concrete.

**Decision.** Before ICP, scale the coarse translation along the ray through the object origin so the
median rendered depth matches the median observed depth on the pixels both cover
(`refine/depth_init.py`, `z_init: median_depth`). Offline on the sample: success (MSSD < 0.1 d)
11.5 % → 46.5 %, 89 % of hypotheses closer to GT, 6.7 % farther by > 1 mm. The shift is stored per
hypothesis (`z_shift_mm`, `z_init_overlap`), the refine stage version is bumped 3 → 4, and D8 records
the step.

**Not chosen.** Widening the depth window / search radius (does not fix the geometry: the coarse
pose is not *near* the right one, it is on the wrong ray depth); a full re-estimation (out of scope,
and unnecessary given the numbers).

## B7 — Apply the initialisation always

**Context.** A small overlap on an occluded fixture instance let the initialisation move a
near-correct pose by 8 mm; the gate caught the resulting ICP drift. Question: apply only when the
shift is large, or the overlap is large?

**Decision.** Always apply when ≥ 50 pixels overlap. On the sample:

| rule | success | improved | worsened > 1 mm | broke a success |
|---|---|---|---|---|
| always | 0.465 | 0.893 | 0.067 | 3 / 1371 |
| only if |dz| > 0.05 d | 0.457 | 0.813 | 0.058 | 1 |
| only if |dz| > 0.1 d | 0.406 | 0.702 | 0.051 | 0 |
| overlap ≥ 200 px | 0.465 | 0.892 | 0.067 | 3 |

Every threshold loses more than it saves. The fixture test now expects ≥ 11 of 12 accepted.

## B8 — What the gate compares against

**Context.** With the initialisation, "displacement from the coarse pose" would include the depth
shift — often > 0.25 d by design — and reject the very corrections the step exists for.

**Decision.** Displacement (α, β) is ICP's move from the pose it started at (after the initialisation);
the IoU-drop check keeps the coarse render as its baseline, because it asks whether the whole
refinement made the silhouette agreement worse. `check_gate(T_start, …, render_coarse, …)`.

## B9 — How the gate is judged

**Context.** The plan asks for "precision of rejection (how often a rejection was the right call)".
AR alone cannot attribute a change to the gate; the refine stage stores the coarse pose, the
candidate and the gate measurements per hypothesis, so each decision can be judged against GT.

**Decision.** `evaluate/refinement.py` scores every hypothesis against its closest GT (by coarse MSSD)
as coarse / init / candidate / final, using MSSD + MSPD recall fractions (no VSD, no top-n matching —
a proxy used only to judge decisions; all AR numbers come from the evaluate stage). Two precisions are
reported: *strict* (the rejected candidate was no closer to GT than the coarse pose) and *success*
(it was not within 0.1 d, so rejecting it cost nothing), plus the raw counts *rejected good* and
*avoided break*. Only hypotheses matched to a valid (BOP-target) GT count.

## B10 — Sweep beyond α and β; the no-change rule

**Context.** The α/β-only sweep on A2 moved the val objective by +0.96 pt; per-check precision showed
the silhouette checks were the larger loss (strict precision 13–24 %, 124 good candidates rejected).

**Decision.** `gate_sweep` crosses α × β × min IoU × max IoU drop × min fitness (720 combinations)
by re-deciding the stored measurements with the *same* `gate_reason()` the stage uses. The default
is kept unless the best combination beats it by ≥ 0.1 pt on val (`choose_gate_caps`).

**Not chosen.** Re-running the stage per combination (720 × 8 min); tuning on the per-GT protocol
(needs re-evaluation per combination). The proxy's ranking was confirmed by the real re-run: core AR
46.22 → 47.55 for A2.

## B11 — The chosen gate

**Context.** Sweep on the D8-default rows, val → held-out:

| row | best on val | val gain | held-out rf | held-out success | held-out net successes | rejected |
|---|---|---|---|---|---|---|
| A2 | (0.6, 90, 0, 1, 0.3) | +1.70 | 58.7 → 60.2 | 53.1 → 55.9 | −132 → 0 | 29.9 % → 7.1 % |
| A3 | (0.6, ∞, 0, 1, 0) | +1.24 | 58.9 → 60.4 | 53.0 → 55.9 | −138 → 0 | 28.7 % → 2.2 % |
| A4 | (0.4, 90, 0, 0.2, 0) | +1.41 | 58.2 → 59.8 | 50.6 → 54.2 | −177 → −6 | 31.3 % → 7.5 % |

The top is flat: with the silhouette checks off, anything from (0.4 d, 90°) to a fully open gate is
within 0.1 pt on val; the fitness check never rejects a success (success precision 100 %).

**Decision.** One gate for all three variants: α = 0.6 d, β = 90°, min IoU 0, max IoU drop 1.0 (both
checks off), min fitness 0.3. Silhouette IoU, displacement and fitness are still measured and stored
as QualitySignals for D11's confidence model. Re-run confirmed: official AR A2 46.5 → 47.8,
A3 46.5 → 47.9, A4 45.9 → 47.3; 7.5 % rejected, success precision 99.7 %, 1 good candidate lost,
1 break avoided. Written into D8's change log with the reasoning: the CNOS mask is the evidence the
coarse pose was fitted to, so silhouette agreement is not independent evidence of a correct pose.

**Not chosen.** A fully open gate (same AR, but the fitness signal is free and its rejections are all
failures — useful downstream); per-variant gates (differences within noise, one config is simpler);
an IoU-drop check referenced to the initialised pose (untested; the absolute IoU check was harmful
too, so the silhouette is unlikely to be rescued by a different baseline).

## B12 — Registration variant

**Context.** Official AR with the tuned gate: point-to-plane 47.8, Tukey-robust 47.9, GICP 47.3;
median cost 0.79 / 0.85 / 0.99 s per hypothesis.

**Decision.** Point-to-plane stays the default (`configs/config.yaml`, `refiner: pt2plane`): tied best
within noise, fastest, no extra parameters. RQ-B's answer is "the variant is worth < 1 point".

## B13 — Code defaults vs. tuned configs

**Decision.** `GateParams` in `refine/gate.py` keeps the D8 design values (0.25, 30°, 0.5, 0.1, 0.3):
the tests exercise the mechanism with them, and the dataclass documents the design. The tuned values
live only in `configs/refiner/*.yaml`, with the reasoning in a comment. The docstring says so.

## B14 — Reading the "accepted worse" gallery

**Context.** 578 accepted candidates ended farther from the target GT than the coarse pose. The gallery
shows nearly all are wrong-instance detections: CNOS masked another copy of the same object, the coarse
pose sits on it, and ICP correctly registers to the object under the mask; both poses fail for the
target (typically 100 → 150 mm).

**Decision.** Treat *broke* (a coarse success turned failure: 65 for A2) as the refinement's real
regression count and say so in the doc; do not try to gate wrong-instance cases in Beta — that is a
segmentation/association problem (Gamma's association, Delta's confidence).

## B15 — Provenance of the untuned rows

**Decision.** The D8-default runs stay on disk (refine `cbbca7cfd5be174c` / `cdb555f0e60f6df7` /
`cd0910fa3e187f6f`, evaluate `b17e103d4ae5d086` / `520654e4616ccd29` / `40f8786265cf26cc`) and their
report is kept as `outputs/tless_beta_report_gate_d8_default.md`; the results doc quotes both, so the
"gate as designed" numbers are reproducible, not just remembered.

## B16 — Plots, and what gets committed

**Context.** The deliverables line says "stratified tables + plots"; the first close-out had tables and
galleries only (found during re-verification).

**Decision.** `tools/refine_report.py` now also writes `outputs/<dataset>_beta_strata.png` (before/after
AR by initial-error bin and by visibility bin; per-hypothesis success ladder coarse → init → ICP →
gate). Three figures are committed under `docs/figures/` (the strata PNG, 96 KB; the two A2 galleries
as 3/4-scale JPEGs, ~210 KB each) so the results doc renders on GitHub; everything else stays under
the git-ignored `outputs/`.

## B17 — A5 with refinement

**Context.** The ablation matrix lists A5's refine column as "best"; Alpha ran A5 un-refined as the
estimator comparator (48.2), and that row is quoted in the Alpha results and README.

**Decision.** A new experiment `A5r` (`configs/experiment/A5r.yaml`: CNOS → MegaPose → depth init +
point-to-plane + tuned gate) answers "does depth refinement add on top of MegaPose's own refiner?";
A5 is left as it was so Alpha's row keeps its meaning (matrix: A5 refine "—", A5r "best").
`tools/run_beta.sh` takes `ONLY=A5r BEFORE=A5` and writes its own report file (`*_beta_report_A5r.md`),
so the milestone report is not overwritten.

**Result.** 48.2 → 46.9 official AR: the stage *hurts* MegaPose. Per-hypothesis success 42.0 % →
61.0 % after the depth initialisation → 55.2 % after ICP; the gate is neutral. MegaPose has 2495 of
5818 instances already within 10 mm, and those are what ICP damages (see B20). Conclusion for the
matrix's "best refine" column: the stage is for RGB-only coarse poses; on a depth-aware estimator only
the initialisation is worth keeping.

## B18 — Waiting on long jobs (process)

**Context.** Two completion waiters written as `until ! pgrep -f "tools/run_beta.sh"` matched their own
command line and never exited; the finished pipeline sat unreported overnight. Same root cause as the
`pkill -f` note already in the machine's memory file.

**Decision.** Wait on a file the job writes last (the report, a `_SUCCESS` marker) or on a recorded
PID, never on a process-name pattern. Recorded in the machine notes.

## B19 — Commits

**Decision.** Four themed commits on `main`, in dependency order — (1) renderer + pool fix,
(2) depth initialisation + gate reference + tuned configs + stage version bumps, (3) analysis module,
galleries, report tooling and their tests, (4) results, milestone and decision records — so each is
reviewable on its own. Not pushed; the caches and reports under `outputs/` stay on the GPU machine.

## B20 — The constant offset ICP converges to

**Context.** Both A2 and A5r lose AR in the [0, 5) mm bin. For hypotheses within 0.05 d *after* the depth
initialisation, ICP moves the pose by ~3 mm and worsens MSSD in 72 % (FoundPose) / 91 % (MegaPose) of
cases — 95–98 % for objects under 90 mm. The move is the same for both estimators: +0.4–0.6 mm in
camera x, **+2.2–2.5 mm in camera y**, 0 in z; fitness 0.96, RMSE 1.5 mm, i.e. the registration is as
good as the data allows. A constant camera-frame offset between depth-fitted poses and GT is a
property of the sensor data (depth↔RGB alignment, ≈ 4 px at 700 mm), not of the registration.

**Decision.** Record it; do not correct it in Beta. Subtracting an offset estimated against the test
GT — even on the val scenes — would be tuning the method on the benchmark's ground truth rather than on
image evidence. The legitimate fix is a GT-free calibration (depth-edge to RGB-edge alignment on the val
scenes) applied as a sensor constant; that belongs with Gamma, where registration is revisited for joint
multi-view ICP. Until then the [0, 5) mm regression and the A5r result are reported as they are.

**Not chosen.** A "keep the initialised pose if ICP moves it less than X" rule (would also freeze
genuine small corrections, and X would again be tuned on GT); dropping ICP for depth-aware estimators
(A5r shows the initialisation alone would beat both, but that is one dataset and one sensor).
