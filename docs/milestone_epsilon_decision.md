# Milestone Epsilon — decision log

Every decision taken while executing Milestone 6a (uncertainty-driven next-best-view, started
2026-09-19 straight after Delta), in the order it was taken, with the evidence that forced it and
the alternative not chosen. [`DECISIONS.md`](DECISIONS.md) holds the standing decision (D14),
[`MILESTONES.md`](MILESTONES.md) the plan, [`results_epsilon.md`](results_epsilon.md) the numbers.

| # | Decision | Status |
|---|---|---|
| E1 | Start Epsilon on 2026-09-19 on the GPU machine (CPU only), from Delta's caches; Epsilon before Deployment, as the plan's default | applied |
| E2 | An episode is one scene and one start View; the Candidate Viewpoints are the scene's other target Views (17 per XYZ-IBD val scene, 50 per T-LESS scene); every policy is evaluated on the *same reference Views* — Gamma's strided 4-view group of the start — so policies are paired by GT instance and `fixed` with budget 4 is exactly the A8 k = 4 group | applied |
| E3 | The hypothesis set of a track is its symmetry-aligned members topped up with posterior samples of the FusedPose (translation noise 0.05 d along the observing camera's ray, 0.02 d across it, 10° rotation): the caches hold one refined hypothesis per Detection, so D14's "top-K hypotheses" does not exist for a single-view track | applied (D14 amended) |
| E4 | Score `S(v) = Σ_t (1 − Confidence_t) · U_t(v) · V_t(v)` over *every* track, `U` the mean pairwise silhouette disagreement of the set rendered into `v`, `V` the predicted visible fraction against the other tracks' fused poses; the Verdict governs stopping only — `request_view` is 3 % of XYZ-IBD FusedPoses (Delta), too few to drive a score | applied |
| E5 | Silhouettes at a quarter of the View's resolution; ties or an all-zero score fall back to the fixed order and are recorded as `fallback` | applied |
| E6 | One `nbv` stage stands in for `associate → fuse → confidence` (the same functions on in-memory tables after every unlocked View); its cache hash carries the three configs and the model fingerprints; the joint ICP polish is not offered (dropped in Gamma) | applied |
| E7 | Baselines: `fixed` (the strided order, budgets 1–4 and every View), `random` (a permutation drawn once per episode from (seed, scene, start), so budget rows are nested prefixes), NBV budgets 2–4; four start groups per scene = 60 episodes per policy and budget | applied |
| E8 | The D14 loop proper (`stop = verdict`) runs with the published thresholds (τ_acc 0.823 / τ_rej 0.675) and the whole pool as budget; the `uncertain` set is a parameter (default `request_view`) | applied |
| E9 | Grasp Poses are one JSON file per dataset (`models/grasps/<dataset>.json`), seeded from the bounding box (jaws across the smallest extent, approach along the middle one) and editable by hand; `T_robot_world` is a parameter of the demo (no dataset value); Open3D's offscreen renderer draws the pick (EGL on the GPU machine; the test skips without a context) | applied |
| E10 | Track weight ablation: the binary entropy of the Confidence (`nbv.params.score.weight=entropy`, rows `A9_nbve_*`) next to the E4 weight — with 64 % of tracks rejected, `1 − p` spends the score on spurious Detections; run before the other start groups so the choice rests on evidence, not on the first curve | applied |
| E11 | An `oracle` policy (the next View scored on the ground truth of the reference Views: distinct annotated instances matched by a successful FusedPose, 500 sampled points) as the ceiling of view choice — RQ-E needs to say how much any choice can gain before it says what NBV gains | applied |
| E12 | All four XYZ-IBD start groups (60 episodes) plus T-LESS group 0 (20 episodes) confirm the group-0 reading: NBV, random next and the strided order lie within one paired interval at every budget on both datasets; the oracle's +7–17 pt ceiling follows the detector's per-image success (ρ 0.71 / 0.62), not the geometric score (ρ 0.04 / 0.10) — closed as a negative result with its ceiling; no re-tuning of the score on the val scenes | applied |

---

## E1 — When and where

**Context.** Delta closed on 2026-09-19, ahead of its 2026-12-06 gate, so slip-policy rule 1 does
not fire and a stretch milestone is attempted. The plan prefers Epsilon (laptop-only, produces a
figure) over Deployment; every cache is on the GPU machine, the link to the laptop is ~1 MB/s.

**Decision.** Start immediately on the GPU machine, CPU only, from the cached refine stage of the
A8 rows. Nothing in Epsilon needs the GPU: the active loop re-runs association, fusion and the
confidence models (seconds per scene), the NBV score renders silhouettes with the CPU raycaster,
and the evaluate stage on 60 reference images takes ~6 min per row.

## E2 — What an episode is and where it is scored

**Context.** D14 says NBV unlocks a real View of a multi-view Scene. The XYZ-IBD val scenes have
50 calibrated Views of one camera on a ring above the bin; Gamma's targets keep every third (17
per scene) and its k-view rows group them by striding (`0, 12, 24, 36`, `3, 15, 27, 39`, …),
each group scored on its own images. A policy that chooses its own Views cannot be scored on
"its own images": the sets differ per policy and a policy that picks hard Views would be
penalised for it.

**Decision.** An episode is (scene, start View); the start is the first image of strided group
`g`, the pool is the scene's other target Views, and the episode's FusedPoses are projected into
the *reference Views* = the four images of strided group `g`, whatever the policy used. Every
policy of one start group is therefore scored on the same GT instances in the same images, the
differences are paired, and `fixed` with budget 4 uses exactly the Views of the A8 k = 4 group
`g` — `tools/check_epsilon.py` asserts that its per-GT errors reproduce A8_k4's. Four start
groups per scene give 60 episodes per (policy, budget) on XYZ-IBD.

**Not chosen.** Scoring on every View of the pool (17 images per scene, 30 min per evaluate row
instead of 6, and the same fairness) and scoring on the start View only (a 1-view test of a
world-frame pose; MSPD would not see a depth error).

## E3 — The hypothesis set of a track

**Context.** D14 renders "the top-K aligned hypotheses of an uncertain ObjectTrack" into each
candidate. FoundPose leaves one refined PoseHypothesis per Detection in the caches (MegaPose's
multi-hypothesis output was not refined past Alpha), so a track seen from one View owns one
pose and no pairwise disagreement exists.

**Decision.** The set is the track's members, symmetry-aligned to the fused pose (the real
samples), topped up to `n_samples = 6` with posterior samples of the fused pose. A sample
perturbs the pose in a member camera's frame with σ = 0.05 d along the optical axis, 0.02 d
across it and 10° about the object origin — the anisotropy Beta measured ("right in the image,
wrong in depth": median 25 mm depth error of the coarse poses, the dominant residual after
ICP). Samples are deterministic in (seed, scene, track) so every candidate is scored on the same
set. A View that looks across the observing ray sees the depth samples as a lateral spread; one
that looks along it sees a change of scale (synthetic test: the across-ray View's disagreement
is > 3× the along-ray one's).

**Not chosen.** Coarse + refined pose as a 2-set (real, but measures the refinement's
correction, not the posterior); the members alone (no signal for single-view tracks, which are
70 % of XYZ-IBD tracks after one View).

## E4 — Which tracks the score counts

**Context.** D14 scores "an uncertain ObjectTrack". With the published thresholds 3.5 % of
XYZ-IBD FusedPoses (Delta, held out) and 9 of 193 after one View on the val scenes are
`request_view`; 64 % are `reject`.

**Decision.** Every track contributes, weighted by `1 − Confidence`: the expected benefit of a
View is the sum over tracks of the probability the track is wrong times what the View can tell
about it. Accepted tracks (Confidence ≥ 0.82) contribute at most 0.18 of their disagreement;
rejected ones drive the score. The Verdict enters the loop only as the stopping rule (E8).

## E5 — Rendering cost

Silhouettes at a quarter of the View's resolution (360 × 270 for XYZ-IBD): ~1 s per candidate
View for 17 tracks × 7 renders plus one scene render, 16 candidates per step, ~80 s per 15-scene
step with 8 workers. A step is a robot decision, so this is also the loop's cost figure.

## E6 — One stage for the loop

The loop re-runs association, fusion and the confidence models after every unlocked View. Rather
than a stage per step, the `nbv` stage runs `associate_scene`, `fuse_tracks` (a new entry point
of the fuse stage that projects into given Views) and the confidence stage's scoring functions
on in-memory tables, and writes what `evaluate` reads from a confidence stage plus
`episodes.json`. Its hash carries the multiview, fusion and confidence configs and the model
fingerprints, so a refit or a changed gate misses the cache exactly as it would for A8.

## E7 — Baselines

`fixed` follows the strided order, so its budget-k row is Gamma's k-view group (budget 1 = the
single View projected into four; budget 0 = every View of the pool, the information ceiling).
`random` draws one permutation per episode from (seed, scene, start), so the budget-2 Views are a
prefix of the budget-3 Views and the curves are paired within a policy too. NBV rows re-run the
loop per budget (the choices are deterministic, so they nest as well; the checker asserts it).

## E8 — The Verdict-stopped loop

`stop = verdict`: unlock Views while any track's Verdict is in `uncertain`; stop at accept /
reject everywhere, exhaustion or the budget. With `request_view` alone the loop is short (E4);
the set is a parameter so a wider one (`request_view, reject`) can be run without code.

## E9 — Simulated pick

`T_robot_gripper = T_robot_world @ T_world_object @ T_object_gripper` (D14), spelled out in
`binposert.viz.grasp.pick_transform` and printed matrix by matrix by `tools/grasp_demo.py`. The
Grasp Poses are committed JSON (`models/grasps/xyzibd.json`, 15 objects), seeded by the
bounding-box rule and meant to be edited by hand; `T_robot_world` places a robot base 600 mm from
the world origin because the dataset has none. The drawing (FusedPoses coloured by Verdict, the
used camera frames, the gripper at the three most confident accepted picks) uses Open3D's
offscreen renderer, which has an EGL context on the GPU machine; the test skips where it has
none.

## E10 — What the score weights

**Context.** The first start group's curve (XYZ-IBD, 15 episodes): NBV 33.7 / 36.2 / 39.5 AR at
2 / 3 / 4 Views against random 33.5 / 38.0 / 39.5 and the strided order 33.9 / 38.5 / 40.1 —
inside the paired interval of random at every budget and behind the strided order at three
Views. After one View 64 % of the tracks are `reject` (Delta: 82 % of them rightly), so the E4
weight `1 − p` puts most of the score on Detections that no View will rescue, and the argmax
becomes "the View in which the most spurious tracks would look different".

**Decision.** Add the binary entropy of the Confidence as an alternative weight — largest at
p = 0.5, ~0.14 at p = 0.02 or 0.98 — and run it as its own rows (`A9_nbve_b<k>_g<g>`) on every
start group, before the remaining groups run, so the report can compare both against random with
the same episodes. The E4 default stays the D14 reading; the ablation is a finding, not a
re-selection after seeing the test set (there is no held-out set here: every row is on the val
scenes, which is what the exit criterion asks).

**Not chosen.** Restricting the score to `request_view` tracks (E4: 3 % of tracks) or dropping
rejected tracks outright (the reject band is only 82–84 % precise).

## E11 — The ceiling of view choice

**Context.** With the first start group in, NBV, random and the strided order lie within one
paired interval of each other on both datasets, while every extra View is worth 3–13 points. The
exit criterion compares NBV with random; a robot engineer needs the other comparison too: how
much would the *best possible* next View gain? If that is also small, view choice is the wrong
lever on these rigs and no score will change it.

**Decision.** A fourth policy, `oracle`, ranks the candidates by the belief they would produce:
each candidate's belief update is labelled against the reference Views' annotations
(`label_fused`, 500 sampled surface points — a ranking proxy, not the published metric) and the
View that matches the most distinct annotated instances with a successful FusedPose wins. It reads
the ground truth and is reported as an upper bound, never as a policy. Cost: ~45 s per step per
XYZ-IBD scene (16 candidate belief updates plus labels), ~5 min per row with 8 workers.

**First observation.** On scene 0 the oracle's second View is image 21 — 3° from the start View,
the *closest* candidate — where the D14 score picked image 9 at 19°: the association gains more
from a View that sees the same Detections than from a wide baseline.

## E12 — What the full run says, and what is not done about it

**Evidence** (`results_epsilon.md`, closed on all four XYZ-IBD start groups + T-LESS group 0).
XYZ-IBD, 60 episodes: NBV 31.6 / 35.8 / 39.4 AR at 2 / 3 / 4 Views, random 32.5 / 36.5 / 39.1,
strided 32.2 / 36.4 / 39.1, entropy-weighted NBV 31.7 / 36.4 / 39.0; paired NBV − random
−1.14 [−2.68, +0.53] / −0.85 [−2.40, +0.61] / +0.93 [−0.51, +2.56] — spans zero at every budget,
the same reading as group 0 alone but with a tighter interval (60 vs 15 episodes). Oracle
39.5 / 45.1 / 47.9 (+7.4 / +9.4 / +10.0, every interval clear of zero and growing with more
data). T-LESS, 20 episodes (unchanged from group 0, run as a cross-check, not extended): NBV
67.1 / 70.5 / 74.4, random 66.4 / 68.2 / 74.6, strided 65.4 / 69.9 / 72.6, oracle 79.5 / 85.1 /
85.3 (+13.1 / +17.0 / +10.7). The oracle's per-candidate gain correlates 0.71 (XYZ-IBD) / 0.62
(T-LESS) with the number of correct single-view hypotheses the candidate image contributes and
0.04 / 0.10 with the D14 score — the same ordering on both datasets, now on 60 XYZ-IBD episodes
instead of 15.

**Decision.** Report it as it is: the D14 score answers a question these bins do not ask. The
mechanism it measures (which View sees the residual ambiguity of a tracked pose) is real and
recovered by the synthetic test, but after depth-initialised ICP and fusion the residual pose
error is not what limits AR — the detector's recall is, and a View is worth the copies it adds,
which no pre-observation geometric score can see. All four XYZ-IBD start groups (E7) ran to
completion; the correlation held at 4× the data (0.09 → 0.04 on XYZ-IBD, if anything weaker),
so the milestone is closed rather than left open for a result that a fixed conclusion would not
change.

**Not done.** Tuning the score's σ / K / render scale or its weight against the val curves (there
is no held-out set in this milestone: the rows *are* the val scenes), or adding an image-based
term (a detector run on the candidate image is the observation NBV is meant to avoid). A
learned "expected detection gain" from the first View's image is the natural follow-up and is
recorded for Finalisation's limitations section, not attempted.

**The Verdict loop.** With the published thresholds `request_view` marks 3–6 % of tracks and not
the scenes that gain. Averaged over all four XYZ-IBD start groups the NBV-driven and
random-driven loops both settle near 4.0 Views (matching the fixed budget-4 row's Views used) but
reach only 31.2 / 32.2 AR against the fixed row's 39.1 — the loop is a *worse* allocator than
simply picking a four-View budget, because a scene it should keep exploring (a track stuck at
`accept`/`reject` on the wrong pose) looks the same as one it has genuinely resolved. On T-LESS
the NBV-driven loop stops all 20 episodes on `stop:verdict` at a mean of 3.05 Views for 61.4 AR,
11 points under the random row at the same mean budget. A budget-driven loop with the strided
order is the better allocator on both datasets, and that is what Finalisation will say.
