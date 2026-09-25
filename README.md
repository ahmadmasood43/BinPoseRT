# BinPoseRT

Reliability-aware multi-view 6D pose estimation for robotic bin picking.

Given one or more calibrated RGB-D views of a cluttered bin and CAD models of the parts, BinPoseRT
returns, for every visible physical object, a world-frame pose `T_world_object`, a **calibrated
confidence** that the pose is correct, and a **verdict** (`accept` / `reject` / `request_view`).

> Status: **Epsilon closed** (increment 6a of 7, 2026-09-23, stretch): uncertainty-driven
> next-best-view over real camera views does not beat random-next on AR-vs-views — reported as a
> negative result. On the full run (XYZ-IBD 60 episodes across four start groups, T-LESS 20) the
> D14 score, random next and the strided view order land within one paired 95 % interval at every
> budget, while a ground-truth oracle shows +7–17 AR is available: a View's worth on these bins is
> the copies the detector finds in it (ρ = 0.71 / 0.62 with the oracle's gain), not its geometry
> (ρ = 0.04 / 0.10 with the D14 score). The grasp transform chain is implemented and visualised.
> Delta (increment 5, 2026-09-19): two logistic models predict pose failure with 0.90–0.93
> held-out ROC-AUC on T-LESS and XYZ-IBD (0.96 for four-view fused poses); calibration is 3 % ECE
> on T-LESS, 8 % on XYZ-IBD, and the verdict bands chosen for 95 / 90 % precision deliver 89 / 84 %
> held out because the pre-registered val scenes are the easiest ones. Next: Finalisation
> (stretch Deployment if time allows).
> See [docs/MILESTONES.md](docs/MILESTONES.md). Numbers in this README only ever come from `outputs/`.

## What it does

- zero-shot segmentation and initial pose (CNOS · FoundPose · MegaPose, as plugins)
- depth-based refinement: translation from depth, ICP variants, a measured acceptance gate
- symmetry-aware evaluation and fusion
- calibrated multi-view SE(3) fusion
- confidence / failure prediction with calibration analysis
- *(stretch)* uncertainty-driven next-best-view over real camera views, with a GT oracle as the ceiling
  and a simulated pick (`T_robot_gripper` chain, Open3D)

"RT" refers to one explicit budget: the **update path** (refine → associate → fuse → confidence for one
object, given cached detections) targets p95 < 200 ms; the full pipeline latency is measured and reported,
not promised ([ADR-0004](docs/adr/0004-two-latency-budgets.md)).

## Key results

BOP19 average recall on the T-LESS test targets (1,000 images, 6,423 instances), official
`bop_toolkit` numbers, single view ([Alpha](docs/results_alpha_tless.md) · [Beta](docs/results_beta_tless.md)).

| Row | Masks | Coarse pose | Refinement | AR | VSD | MSSD | MSPD |
|---|---|---|---|---|---|---|---|
| A0 upper bound | ground truth | FoundPose | — | **59.2** | 45.7 | 50.1 | 81.9 |
| A1 baseline | CNOS | FoundPose | — | **35.4** | 27.4 | 30.1 | 48.6 |
| A2 | CNOS | FoundPose | depth init + point-to-plane ICP + gate | **47.8** | 41.4 | 50.6 | 51.4 |
| A3 | CNOS | FoundPose | depth init + Tukey-robust ICP + gate | **47.9** | 41.4 | 50.7 | 51.5 |
| A4 | CNOS | FoundPose | depth init + GICP + gate | **47.3** | 40.6 | 50.3 | 51.0 |
| A5 comparator | CNOS | MegaPose (multi-hypothesis, RGB) | MegaPose's own | **48.2** | 45.6 | 45.9 | 53.0 |
| A5r | CNOS | MegaPose | MegaPose's own + depth init + point-to-plane ICP + gate | **46.9** | 40.8 | 49.5 | 50.3 |

Reading: FoundPose-coarse is right in the image plane (MSPD 82 with GT masks) but off in depth
(VSD/MSSD ≈ 50; median depth error 25 mm, half the hypotheses beyond 0.25 d). Depth refinement recovers
+12 AR points, nearly all in the depth-sensitive metrics, and most of that comes from re-initialising
the translation along the viewing ray before ICP; the registration variant is worth < 1 point. The
silhouette-vs-mask acceptance gate rejected six good candidates per bad one (the mask is the evidence
the coarse pose was fitted to) and is kept only as a signal. On top of MegaPose's own refiner the
stage *costs* 1.3 points (A5r): ICP moves already-good poses by a constant ~2.5 mm along camera y — a
depth↔RGB offset of the sensor data, to be calibrated without GT in Gamma. CNOS still costs the most:
15 % of targets get no prediction and AR stays ~1 % below 30 % visibility.

| Dataset | AR (VSD/MSSD/MSPD), single view | Δ from refinement | Δ from depth↔RGB calibration | Δ from fusion (2 / 3 / 4 views) | Δ from Confidence ranking (2 / 3 / 4 views, official) | Update-path p95 (Python) |
|---|---|---|---|---|---|---|
| T-LESS | 51.1 (48.0 / 51.4 / 53.2) | +12.4 (35.4 → 47.8) | +3.3 (47.8 → 51.1, GT-free estimate) | +13.1 / +18.4 / **+21.4** (→ 64.2 / 69.5 / 72.5) | +1.1 / +2.6 / **+3.4** (→ 65.3 / 72.1 / 75.9) | **0.75 s** (refine 0.27 s / hyp, render-bound 67%; associate + fuse + confidence 28 ms; single thread, quiet machine) |
| XYZ-IBD (val, core) | 22.6 (20.4 / 23.4 / 23.9) | in the chain | none needed (0 px) | +8.3 / +13.9 / **+15.7** (→ 30.9 / 36.5 / 38.3, best) | +0.3 / +1.0 / **+1.8** (→ 30.6 / 36.8 / 39.1, core) | — |

Gamma (multi-view, [`docs/results_gamma_tless.md`](docs/results_gamma_tless.md)): the depth map of the
T-LESS sensor sits 3.3 px below the RGB image, measured from edge alignment on val scenes with no pose
annotation, and correcting it is worth +3.3 AR by itself. Fusing calibrated views — one-to-one
association per View with a symmetry-aware gate (purity 100 % vs GT), symmetry alignment, weighted SE(3)
mean, the fused pose predicted in every View — lifts AR from 51 to 72.5 with four views; occluded objects
gain most (visibility 10–30 %: 1 → 40). A joint multi-view ICP polish *lowers* the mean at every view
count (the union cloud inherits each view's residual calibration error) and is kept only as an ablation.
Calibration errors of 2 mm / 0.25° cost under 2 points. On XYZ-IBD
([`docs/results_gamma_xyzibd.md`](docs/results_gamma_xyzibd.md)) the detector finds fewer than half of the
10–59 identical copies per bin, so the single-view row is 22.6; four views lift it to 38.3, association
stays 89–93 % pure on stacked copies, and picking the best-weighted view edges out averaging there
because members already agree to under a millimetre.

Delta (confidence, [`docs/results_delta.md`](docs/results_delta.md)): every FusedPose now carries a
Confidence from a logistic model on the versioned signal schema (ICP residual in diameters, silhouette
IoU, ICP displacement, the members' dispersion and view count) and a Verdict. Held out, Model H (per
hypothesis) reaches 91.1 ROC-AUC and Model F (per fused pose) 92.4, rising from 90 to 96 as the view count
goes 1 → 4; Brier 0.11–0.13. Ranking BOP predictions by Confidence instead of the estimator's score
is worth +1.1 / +2.6 / +3.3 AR on T-LESS with 2 / 3 / 4 views (official 72.5 → 75.9 at four) and
+0.3 / +1.0 / +1.8 on XYZ-IBD, nothing with one; Model H as the fusion weight changes AR by under
0.6 and is not the default. The
estimator's and detector's own scores are the least portable signals (dropping `pose_score` *gains*
1 pt held out); discrimination transfers across datasets but calibration does not (ECE 12–19 %).
The verdict bands miss their targets because every fifth scene — the val rule fixed in Beta — turned
out to be the easiest fifth; leave-one-scene-out shows 95 / 90 % precision is reachable with
representative fit scenes, accepting 22 % of poses outright and deferring 28 %.

## Project documents

- [`docs/DECISIONS.md`](docs/DECISIONS.md) — the single decision record: scope, architecture, milestones, ablations, risks
- [`docs/MILESTONES.md`](docs/MILESTONES.md) — the dated milestone plan: weekly tasks, exit criteria, slip policy
- [`docs/milestone_beta_decision.md`](docs/milestone_beta_decision.md) — the Beta decision log: what was decided, on what evidence, and what was not chosen
- [`docs/milestone_gamma_decision.md`](docs/milestone_gamma_decision.md) — the Gamma decision log (G1–G21)
- [`docs/milestone_delta_decision.md`](docs/milestone_delta_decision.md) — the Delta decision log (Δ1–Δ18)
- [`CONTEXT.md`](CONTEXT.md) — the project vocabulary (Detection → PoseHypothesis → ObjectTrack → FusedPose)
- [`docs/adr/`](docs/adr/) — the four hard-to-reverse decisions
- [`docs/source/`](docs/source/) — the original research plan this project was derived from

## Installation (CPU, development)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest -q
uv run python tools/run.py experiment=smoke      # mini fixture end to end, no GPU
```

No GPU, CUDA or Docker is needed to develop and test the core. Neural estimators run in their own
environments on a GPU machine (`docker/`, or `tools/setup_estimator_env.sh <name>` where Docker cannot
see the GPU) and their outputs are cached under `outputs/<dataset>/<split>/<stage>/<hash>/`.

## Running an experiment row

```bash
uv run python tools/download_bop.py tless --parts models test_primesense_bop19
uv run python tools/run.py experiment=A0 dataset=tless                  # from cached GPU outputs
uv run python tools/run.py experiment=A1 dataset=tless run_external=true  # on the GPU machine
tools/bop_eval.sh outputs/tless/test_primesense/evaluate/<hash>/binposert-A1_tless-test.csv
uv run python tools/gallery.py outputs/tless/test_primesense/evaluate/<hash>
uv run python tools/report.py A0 A1 A5
tools/run_alpha.sh                                                       # all of the above
tools/run_beta.sh                                                        # A2 A3 A4 + refinement analysis (CPU only)
uv run python tools/refine_report.py --before A1 A2 A3 A4                # stratified before/after, gate stats, sweep, galleries
tools/run_gamma.sh                                                       # A6/A7 AR-vs-views rows + extrinsic sweep (CPU only)
tools/run_delta.sh                                                       # labelled tables, confidence models, A8 rows, report
uv run python tools/fit_confidence.py --tag v2                           # refit Model H / F + calibration analysis
```

Every stage is a pure function of its inputs, config and version, cached by content hash and skipped
when its `_SUCCESS` marker exists; a run writes `run_manifest.json` with the commit, config hash,
per-stage timings and GPU. The core evaluator reimplements VSD / MSSD / MSPD and agrees with the
official `bop_toolkit` within 0.15 AR points; the official numbers are the ones reported.

## Datasets

BOP format throughout. `tools/download_bop.py` fetches T-LESS models; large datasets (XYZ-IBD, IPD) are
downloaded on the GPU machines only.

## Project structure

See [D18 in DECISIONS.md](docs/DECISIONS.md#d18--repository-layout).

## Acknowledgements

BOP benchmark and toolkit · FoundPose · MegaPose · CNOS · T-LESS · XYZ-IBD · Open3D · OpenCV.

## License

MIT — see [LICENSE](LICENSE).
