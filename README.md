# BinPoseRT

Reliability-aware multi-view 6D pose estimation for robotic bin picking.

Given one or more calibrated RGB-D views of a cluttered bin and CAD models of the parts, BinPoseRT
returns, for every visible physical object, a world-frame pose `T_world_object`, a **calibrated
confidence** that the pose is correct, and a **verdict** (`accept` / `reject` / `request_view`).

> Status: **Beta complete** (increment 3 of 6, 2026-09-16): depth-based refinement — translation
> initialised from depth, ICP behind one signature (point-to-plane / robust / GICP), an acceptance gate
> that is measured rather than assumed — lifts the T-LESS baseline from 35.4 to 47.8 AR. Next: Gamma
> (multi-view association and fusion). See [docs/MILESTONES.md](docs/MILESTONES.md).
> Numbers in this README only ever come from `outputs/` run manifests.

## What it does

- zero-shot segmentation and initial pose (CNOS · FoundPose · MegaPose, as plugins)
- depth-based refinement: translation from depth, ICP variants, a measured acceptance gate
- symmetry-aware evaluation and fusion
- calibrated multi-view SE(3) fusion
- confidence / failure prediction with calibration analysis
- *(stretch)* uncertainty-driven next-best-view over real camera views

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

| Dataset | AR (VSD/MSSD/MSPD) | Δ from refinement | Δ from 3-view fusion | Update-path p95 |
|---|---|---|---|---|
| T-LESS | 47.8 (41.4 / 50.6 / 51.4) | +12.4 (from 35.4) | n/a | TBD (refine 0.79 s / hyp on CPU) |
| XYZ-IBD | TBD | TBD | TBD | TBD |

## Project documents

- [`docs/DECISIONS.md`](docs/DECISIONS.md) — the single decision record: scope, architecture, milestones, ablations, risks
- [`docs/MILESTONES.md`](docs/MILESTONES.md) — the dated milestone plan: weekly tasks, exit criteria, slip policy
- [`docs/milestone_beta_decision.md`](docs/milestone_beta_decision.md) — the Beta decision log: what was decided, on what evidence, and what was not chosen
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
