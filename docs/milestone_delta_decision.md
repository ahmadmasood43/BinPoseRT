# Milestone Delta — decision log

Every decision taken while executing Milestone 5 (confidence and Verdict, started 2026-09-17
straight after Gamma, closed 2026-09-19), in the order it was taken, with the evidence that forced it and the
alternative not chosen. [`DECISIONS.md`](DECISIONS.md) holds the standing decision (D11),
[`MILESTONES.md`](MILESTONES.md) the plan, [`results_delta.md`](results_delta.md) the numbers.
Numbers are on the held-out (eval) scenes unless stated.

| # | Decision | Status |
|---|---|---|
| Δ1 | Start Delta on 2026-09-17 on the GPU machine (CPU only), from Gamma's caches | applied |
| Δ2 | Split: fit on T-LESS scenes {1, 6, 11, 16} (B1) + XYZ-IBD scenes {0, 20, 40, 60}; report on the other 16 + 11 scenes, per dataset and pooled; cross-dataset rows as the out-of-distribution check | applied |
| Δ3 | Label = MSSD against the *nearest* same-object ground truth (all annotations, `gt_valid` recorded); a hypothesis of an absent object is a failure; a FusedPose is labelled in the world frame against the ground truth of every View of its group | applied |
| Δ4 | Feature schema v1: the 15 QualitySignals + a missing indicator each, mm distances divided by the diameter, plus `rejected`, `log_diameter`; Model F adds the track shape and the members' Model H probabilities | applied |
| Δ5 | Fitted models are JSON files under `models/confidence/<tag>/`, content-fingerprinted into the stage hash | applied (D18 amended) |
| Δ6 | Model H as the fusion weight is an opt-in `+multiview.params.weights.source=model_h` so Gamma's association caches stay valid | applied |
| Δ7 | The `confidence` stage sits between `fuse` and `evaluate`, rewrites the fuse tables with `confidence` / `verdict`, and A8 ranks BOP predictions by Confidence | applied |
| Δ8 | Verdict thresholds: τ_acc = the lowest threshold whose accept band is ≥ 95 % successes, τ_rej = the highest whose reject band is ≥ 90 % failures (≥ 20 val rows each); on overlap the accept band wins and the pushed-down reject band is re-checked | applied |
| Δ9 | Regularisation C by scene-grouped 5-fold CV (log-loss) on the fit rows | applied |
| Δ10 | MLP comparator kept only if it gains ≥ 2 pt ROC-AUC *and* a lower Brier on eval rows | applied |
| Δ11 | The FusedPose table pools the k = 1…4 mean rows (the same objects in several groupings; scene-level split keeps fit and eval disjoint) | applied |
| Δ12 | Recalibrate on out-of-fold logits (Platt), fit Model F on out-of-fold Model H aggregates, choose thresholds on out-of-fold probabilities — in-sample fits on 8 scenes are over-confident | applied (Δ8 amended) |
| Δ13 | The pre-registered val scenes are the easiest of both datasets; the split is kept and the shortfall reported, with a leave-one-scene-out diagnostic as the achievable reference | applied |
| Δ14 | No post-hoc feature selection: `pose_score` / `seg_score` hurt held out, but dropping them after seeing the held-out numbers would be tuning on the test set; recorded for Finalisation | applied |
| Δ15 | A8 ranks BOP predictions by Confidence (+1.1 / +2.6 / +3.3 AR on T-LESS with 2 / 3 / 4 views); Model H as the fusion weight (A8w) changes AR by ±0.2 and de-calibrates Model F — product weights stay the default | applied |
| Δ16 | The update path is 0.51 s median / 0.74 s p95 per track in Python on a quiet machine (refine ≈ 95 % of it, Open3D ICP ≈ half of refine); the 200 ms target is out of reach of C++ ports alone — algorithmic candidates recorded for Deployment | applied (D6 / D13 note) |
| Δ17 | Closure review (three lenses, adversarially verified) before the commit: the MLP comparator's penalty was mis-mapped (`alpha = 1/(C·n)` → `1/C`) and unrecalibrated, the Model F ablation left the signal in through Model H, the LOSO band precisions were in-sample for the threshold choice, `check_delta` mis-specified A8w, `run_delta.sh` skipped rows regardless of the model tag — all fixed; the published v1 model files are byte-identical before and after | applied |
| Δ18 | A job whose worker reports corrupted render output (`RenderCorrupted`) is resubmitted to a fresh process like a dead worker (G20); the XYZ-IBD A8_k3 evaluate died twice on a flipped index bit (2^17 / 2^18) before that | applied |

---

## Δ1 — When and where

**Context.** The plan put Delta in weeks 10–12 (from 2026-11-16) on the laptop; Gamma closed on
2026-09-17 with every cache on the GPU machine and the link still ~1 MB/s.

**Decision.** Start immediately, on the GPU machine, from the cached refine / associate / fuse
stages of the Gamma rows. Nothing in Delta needs the GPU; the labelled tables and the fits are
CPU work of minutes, the A8 rows are new `associate → fuse → confidence → evaluate` runs of the
cached chain (evaluate ≈ 10 min per row).

## Δ2 — Split discipline

**Context.** D11 says "fitted on held-out val splits; evaluated on test". T-LESS has no val split
(B1 made scenes {1, 6, 11, 16} the val set for every threshold choice); XYZ-IBD's test split has no
ground truth, so Gamma ran on `val` (15 scenes, ids 0, 5, …, 70, one object each).

**Decision.** The *fit* rows are T-LESS scenes {1, 6, 11, 16} and XYZ-IBD scenes {0, 20, 40, 60}
(every fifth of the 15, 4 objects of 15); Model H, Model F, C and the Verdict thresholds are chosen
on them and on nothing else. Every reported number is on the other scenes (16 T-LESS, 11 XYZ-IBD),
per dataset and pooled. Two extra rows fit on one dataset's fit scenes only and evaluate on every
row of the other dataset (the out-of-distribution check the risk table asks for). The published
models are the pooled fit; the A8 pipeline rows apply them to all scenes, so the four fit scenes of
each dataset are in-sample there — the Delta report separates `held-out` from `all`, as B1 did.

**Not chosen.** Scene-grouped cross-validation over all scenes (more eval rows, but then no single
published model corresponds to the reported numbers); fitting per dataset (the point of one
schema is one model).

## Δ3 — What "success" is for a row

**Decision.** `success = MSSD(T_pred, T_gt) < 0.1 · diameter` with `T_gt` the annotation of the
same object *nearest by MSSD* among all annotations in the View — not only the BOP-valid ones.
A pose that is right on a heavily occluded copy is a right pose and the model must learn it;
`gt_valid` and the annotation's visibility are recorded for stratification. A hypothesis whose
object is not annotated in the View (a segmentation false positive) is a failure with an infinite
error. A FusedPose is `T_world_object`; MSSD is invariant under a common rigid motion, so it is
scored against the annotations of every View of its group lifted through `T_world_camera`, nearest
wins — equivalent to scoring its projection in the View where it fits best. Candidates farther
than two diameters in translation are skipped unless nothing is nearer (the MSSD of a 17 796-vertex
T-LESS model against 36 symmetry elements costs 15 ms; XYZ-IBD images hold up to 59 copies).

## Δ4 — Feature schema

**Decision.** One versioned `FeatureSchema` (`binposert/confidence/schema.py`, v1) shared by
single-view and multi-view rows: every QualitySignals field is a feature, NaN → 0 plus a
`<signal>_missing` indicator (D11), millimetre distances (`icp_rmse_mm`, `displacement_mm`,
`multiview_residual_mm`, `dispersion_mm`) divided by the object diameter so a 3 mm residual means
the same thing on a 60 mm T-LESS part and a 300 mm XYZ-IBD bar, plus `rejected` (the Refinement
declined the candidate) and `log_diameter`. Model F's schema adds the track shape (`n_members`,
`n_aligned`, `weight_sum`) and the members' Model H probabilities (mean / min / max) and rejected
share. A fitted model stores its schema and refuses another version.

**Not chosen.** Log transforms and pairwise products (an MLP is the sanctioned way to get
non-linearity, Δ10); a dataset indicator (not available for a new dataset); dropping the
always-missing multi-view fields from Model H (the shared schema is the point).

## Δ5 — Where fitted models live

**Decision.** `models/confidence/<tag>/{model_h,model_f,thresholds}.json` + `card.md`, committed
(a few kB of JSON: schema, standardisation, weights, provenance — no pickles, so a model fitted
today loads after a scikit-learn upgrade). A stage config names the files; their content hash
enters the stage hash (D12), so a refit invalidates the confidence stage and the Model-H-weighted
association without touching Gamma's caches. D18's layout gains `models/`.

## Δ6 — Model H as the fusion weight

**Decision.** `WeightParams.source = "model_h"` replaces the D10 product by Model H's
probability (floored at 1e-3; the model has seen `rejected`, so no extra ×0.2). It is switched on
with `+multiview.params.weights.source=model_h` on the command line rather than by new keys in
`configs/multiview/strided.yaml`: the associate hash covers the whole section, and new default
keys would have invalidated every Gamma association, fuse and evaluate cache (≈ 100 rows × 10 min).

## Δ7 — The confidence stage

**Decision.** `associate → fuse → confidence → evaluate`. The stage scores every member of every
track with Model H (batched per scene), aggregates per track, scores the FusedPose with Model F,
assigns the Verdict, and rewrites `fused.parquet` and the projected `hypotheses.parquet` with
`confidence` / `verdict` columns (the artefact schema gains optional columns; every reader
validates required columns only). A8 sets `evaluate.score_signal = confidence`, so the BOP
top-n selection ranks by Confidence instead of FoundPose's `pose_score`. A `fusion=none` row
has no FusedPoses; its hypotheses are scored by Model H directly.

## Δ8 — Verdict thresholds

**Decision.** On the fit rows of Model F: τ_acc is the *lowest* Confidence at which the poses at or
above it are ≥ 95 % successes (the widest acceptance the target allows), τ_rej the *highest* at
which the poses at or below it are ≥ 90 % failures; a band must hold ≥ 20 val rows; if the bands
overlap the accept band wins, the reject band is pushed below it and whether the pushed band still
meets its target is re-checked and recorded (`reject_satisfiable`); an unsatisfiable target leaves
its band empty and says so. `accept ≥ τ_acc`, `reject ≤ τ_rej`, `request_view` between.

## Δ9 — Regularisation

**Decision.** L2 logistic regression on standardised features; C ∈ {0.03, 0.1, 0.3, 1, 3, 10}
chosen by scene-grouped 5-fold cross-validation (log-loss) on the fit rows, separately for H and F.

## Δ10 — MLP comparator

**Decision.** A one-hidden-layer (16 units) MLP on the same features is fitted on the same rows and
reported; it replaces the logistic model only if it gains ≥ 2 pt ROC-AUC *and* has a lower Brier
score on the eval rows (D11: "only if logistic clearly under-fits").

## Δ11 — Which FusedPoses are fitted on

**Decision.** The k = 1, 2, 3, 4 mean rows of Gamma (`A6_k<k>_mean`; the k = 1 mean row was added
for Delta — identical to `none` in pose, but it writes FusedPoses). The same physical objects
appear in several groupings, so rows are correlated; the split is by scene, so fit and eval never
share an object. The per-track base rate falls with k (T-LESS 58.7 % at k = 1, 52.9 % at k = 2):
good hypotheses merge into one track, wrong ones fail the association gate and stay separate —
which is why Model F sees `n_members` and the members' probabilities.

## Δ12 — Out-of-fold recalibration, features and thresholds

**Context.** The first T-LESS-only fit had in-sample ROC-AUC 98.6 / 99.1 (H / F) against 91.2 / 93.0
on the held-out scenes; thresholds chosen on the in-sample probabilities for 95 % accept precision
gave τ_acc = 0.526 next to τ_rej = 0.523 (no `request_view` band at all) and 82 % precision held out.
Model F was also fitted on Model H probabilities that were in-sample for its own fit rows, so it
learnt to trust `p_h_*` more than the eval rows deserve.

**Decision.** Every quantity chosen on the fit rows is chosen on scene-grouped *out-of-fold*
predictions of those rows: (1) each model's logit is recalibrated by Platt scaling `σ(a·z + b)`
fitted on its OOF logits (`ConfidenceModel.calibration`, stored in the JSON); (2) the Model H
aggregates of the fit rows' members are OOF probabilities, the final model's elsewhere; (3) the
thresholds are chosen on the OOF recalibrated Model F probabilities. Only fit rows are ever used.

**Evidence** (pooled v1): Model F's recalibration is `0.926·z − 0.501` and takes the held-out ECE
from 7.2 % to 4.5 % (Brier 0.117 → 0.112); Model H's is `1.045·z − 0.069` (nothing to correct — see
Δ13). Thresholds became τ_acc = 0.823 / τ_rej = 0.675 with a 6 % `request_view` band on the fit rows.

## Δ13 — The val scenes are the easiest ones

**Evidence.** Leave-one-scene-out over all scenes (each scene scored by Model H fitted on the other
34) ranks the fit scenes by AUC: T-LESS {1, 6, 11, 16} at ranks 4, 2, 1, 10 of 20 (mean 0.976 against
0.924 for the other 16), XYZ-IBD {0, 20, 40, 60} at ranks 2, 8, 4, 3 of 15 (0.989 against 0.927). The
out-of-fold AUC *within* the fit rows is 97.1 while the held-out AUC is 92.4; the same thresholds
give 95 / 90 % band precision on the fit rows and 89 / 84 % on the held-out ones (T-LESS 90 / 85,
XYZ-IBD 88 / 82). Per-scene LOSO AUC spans 0.85–0.99 on T-LESS and 0.74–1.00 on XYZ-IBD (the
per-scene table is in `outputs/confidence/v1/report.md`): scene difficulty, not model variance, is
the dominant source of spread.

**Decision.** The split stays as pre-registered (B1's rule, applied to XYZ-IBD before any Delta
number existed) and the published models and thresholds are the ones fitted on it; the shortfall is
the reported result. The leave-one-scene-out analysis (every prediction out of fold, Platt on the
pooled OOF logits — two parameters on 25 516 rows; thresholds chosen on one half of the scenes and
scored on the other, both ways, so their precisions are out of sample too) is reported as the
*achievable* calibration: ECE 2.6 / 2.2 %; the accept band keeps its 95.0 % precision on the
half it was not chosen on while covering 18–26 % of the poses, 24–35 % are deferred (72–73 %
right), and the reject band's precision swings between 85 and 97 % because τ_rej itself moves from
0.60 to 0.28 with the half — the reject threshold is the fragile one.

**Not chosen.** Re-selecting fit scenes by difficulty or enlarging the fit set after seeing the
held-out numbers (tuning on the test set); scene-grouped cross-validation as the *published* protocol
(no single model would correspond to the reported numbers, and the A8 rows need one model file).
A representative val split is noted for Finalisation: with a dataset that has one, or by fixing the
fit scenes with a stratification rule *before* fitting.

## Δ14 — No post-hoc feature selection

**Evidence.** Ablate-one-signal on the held-out rows (the signal removed from Model H *and* Model F,
because Model F reads its members through Model H's probabilities — an earlier version of the table
dropped it from Model F's own features only and showed −0.11 for the ICP residual instead of −1.00):
without `pose_score` Model H gains 1.07 pt and Model F 0.73 pt ROC-AUC; without `seg_score` +0.10 /
+0.31; without `icp_rmse_mm` −1.13 / −1.00; without `silhouette_iou` −0.37 / −0.35; without
`dispersion_mm` (F) −0.23. FoundPose's and CNOS's own scores are the least portable evidence: large
positive coefficients on the fit scenes, a cost on the others.

**Decision.** The published models keep the full schema. Dropping a signal because the *held-out*
numbers improve would make the held-out numbers a fit set. The MLP comparator (Δ10; same rows, same
C with sklearn's `alpha = 1 / C` — its penalty is already per sample — and the same Platt-on-OOF
recalibration) gains 0.4 / 0.5 pt ROC-AUC with the same Brier (H 0.125 vs 0.127, F 0.115 vs 0.112);
below the 2 pt rule, logistic stays.

## Δ15 — What the confidence stage does to the pipeline rows

**Evidence** (T-LESS, core AR, `outputs/tless_delta_report.md`): ranking the BOP predictions by
Model F's Confidence instead of FoundPose's `pose_score` leaves k = 1 unchanged (50.9) and lifts
k = 2 / 3 / 4 from 63.9 / 69.2 / 72.3 to 65.0 / 71.8 / 75.6 (official 64.2 / 69.5 / 72.5 → 65.3 /
72.1 / 75.9; on the held-out scenes alone, paired scene-bootstrap: +1.1 [+0.5, +1.8], +2.6
[+1.5, +3.6], +3.6 [+2.3, +4.8]); on XYZ-IBD +0.3 / +1.0 / +1.8 (held-out paired +0.3 [−0.1, +0.9],
+1.4 [+0.1, +3.6], +2.3 [+0.3, +4.8]). A fused
pose is predicted in every View of its group (G6), so once look-alikes or the wrong copy have been
detected an image holds several projections per object, and the protocol keeps the `inst_count`
best-scored; `pose_score` ranks them by one View's template similarity, the Confidence by what the
whole track knows. Model H as the fusion weight (A8w, Δ6) moves AR by −0.1 … +0.2 on T-LESS and
+0.1 … +0.6 on XYZ-IBD: the D10 product
and the learnt probability order the members nearly alike and the mean of members that agree to
millimetres is insensitive to the weights (as G17 found for the joint polish). It shifts Model F's
inputs (`weight_sum`, reference member, dispersion) and the A8w Confidence is worse calibrated
(held-out ECE 6–9 % vs 3–6 % on T-LESS, 10–11 % vs 7–10 % on XYZ-IBD).

**Decision.** A8 = product weights + Confidence ranking is the default row; A8w stays as the
ablation (its XYZ-IBD gain of up to +0.6 is inside the paired interval's width and comes with
the worse calibration). A Model F deployed with Model-H weights must be refitted on Model-H-weighted tracks.

**Also seen.** Inside the accept band the view count separates right from wrong: accepted tracks
with 1 / 2 / 3 / 4 members are 71 / 87 / 95 / 98 % correct at k = 4 (195 / 251 / 306 / 257 held-out
tracks); a rule "accept only if ≥ 3 Views confirm" meets the 95 % target on T-LESS with the
published thresholds. Accepted failures on T-LESS are 42–57 % wrong identity / copy (the pose fits
a look-alike carrying the wrong label — invisible to every pose signal and to multi-view agreement,
which confirms the same wrong label), 26–44 % near misses at 0.10–0.20 d, 15–20 % orientation flips
of undeclared symmetries; on XYZ-IBD (one object per scene) 52–63 % are orientation flips — a
cup-shaped housing seen from above with four Views agreeing on the wrong way up, and the 296 mm bar
sliding along its axis — and only 8–16 % wrong copies.

## Δ16 — Update-path profile (D13, precondition of D6)

**Evidence** (`tools/profile_update_path.py`, T-LESS A8_k4, 4-view groups, single-threaded,
`outputs/tless_update_path_profile.json`, quiet machine, 50 timed groups / 580 tracks): per
track the update path is **505 ms median / 739 ms p95** against the 200 ms target; `refine` is
285 ms median / 331 ms p95 per hypothesis (1.8 per track), `associate` 1.6 ms, `fuse` 2.5 ms
(29 ms p95, symmetry alignment of 36-element groups), `confidence` 1.7 ms. Inside `refine`:
Open3D `registration_icp` 49 %, numpy reductions over full-resolution masks 20 %, rebuilding the
pinhole ray grid per render 8 %, tensor ↔ numpy conversions 13 %, the gate 5 %. Under the load of
the Delta rows (8 evaluate workers) the same step measured 1.1 s median.

**Decision.** No C++ port is started in Delta (D6 says "after Delta" and only for measured hot
paths). The measured hot path is Open3D's `registration_icp` — already C++ — plus the Python-side
conversions around it; a port of the Python parts could at best halve the refine time, nowhere near
the 200 ms p95. The candidates for Deployment, in order of expected return: (1) a coarser ICP
schedule (two levels, fewer iterations, fewer points — an accuracy/latency Pareto to measure, the
figure D13 requires); (2) the numpy reductions over full-resolution masks and depth (20 % of
refine) and the tensor ↔ numpy conversions (13 %); (3) caching the pinhole ray grid per (K, image
size) in the renderer — it is rebuilt on every render, 8 %; (4) a vectorised symmetry-aware
rotation distance (7.9 k `Rotation.from_matrix` calls in 10 groups). `associate`, `fuse` and
`confidence` together are 6 ms median / 36 ms p95 per track and already inside the budget.

## Δ17 — Closure review

**Context.** Before the Delta commit the diff was reviewed along three lenses (code correctness,
statistical methodology, docs-vs-numbers) by independent readers, every finding then attacked by
two refuters; 18 findings survived.

**What changed.** (1) The MLP comparator: scikit-learn's `alpha` already divides the L2 term by
the sample count, so `alpha = 1 / (C·n)` regularised it ~2 000× less than the logistic model; it is
now `1 / C` (pinned by a test that an MLP without hidden layer reproduces `LogisticRegression(C)`)
and the comparator is recalibrated on out-of-fold logits like the published models — its numbers
moved from "+1.3 / +1.1 pt AUC, ECE 10 %" to "+0.4 / +0.5 pt, same Brier"; the verdict (logistic
kept) is unchanged. (2) The ablate-one-signal table removed the signal from Model F's own features
only, while Model F kept reading it through the members' Model H probabilities; the signal is now
removed from both models — the ICP residual moved from −0.11 to −1.00 pt for Model F, the rest
within 0.1 pt. (3) The LOSO diagnostic chose and scored its thresholds on the same rows (the
precision equals the target by construction); it now chooses on one half of the scenes and scores
on the other, both ways. (4) The overlap branch of `choose_thresholds` re-checks the pushed-down
reject band and records whether it still meets its target. (5) `check_delta.py` no longer expects
the A8w rows (a different association) to hold the Gamma row's prediction count; `run_delta.sh`
skips a row only if its manifest was written with the current tag's model files, gates the fit on
its last output and runs the bop_toolkit evaluations one at a time (they race on the shared
checkout); `confidence_report.py` refuses a row run with another tag's models and reports the
held-out, paired AR delta with its bootstrap interval; the confidence stage logs a warning when the
association's weighting differs from the one Model F was fitted on. (6) Docs: T-LESS Model F's
3.0 % ECE is carried by the extreme bins while the 0.5–0.9 bins are 5–9 pt over-confident — which
is where the thresholds sit and why the accept band lands at 90 %; a handful of stale numbers.

**Not changed.** The published `models/confidence/v1/*.json` files (byte-identical after every
re-run of the fit tool; the A8 caches stay valid); the split; the schema.

## Δ18 — Corrupted render output is a process failure

**Evidence.** The XYZ-IBD `A8_k3` evaluate failed twice, hours apart, with
`RenderCorrupted: index 263101 / 132151 is out of bounds` — 2^18 + 957 and 2^17 + 1079, single
bits flipped in an index into a 1080-row image after the cast itself had validated (G15's
in-render retry ran three times on the same corrupted memory). G21's hardware suspicion stands.

**Decision.** `map_scenes` treats a job that raises `RenderCorrupted` like a job whose worker
died (G20): the job is resubmitted to a fresh process, up to the same retry count; any other
exception still propagates. Matched by exception name so the pool module imports nothing heavy.
