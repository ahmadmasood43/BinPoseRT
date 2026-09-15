# BinPoseRT

Reliability-aware multi-view 6D pose estimation for robotic bin picking.

Given one or more calibrated RGB-D views of a cluttered bin and CAD models of the parts, BinPoseRT
returns, for every visible physical object, a world-frame pose `T_world_object`, a **calibrated
confidence** that the pose is correct, and a **verdict** (`accept` / `reject` / `request_view`).

> Status: **Alpha complete** (increment 2 of 6, 2026-09-15): the single-view baseline — CNOS masks,
> FoundPose / MegaPose coarse poses — runs end to end on T-LESS from cached GPU outputs with official
> BOP numbers. Next: Beta (depth refinement). See [docs/MILESTONES.md](docs/MILESTONES.md).
> Numbers in this README only ever come from `outputs/` run manifests.

## What it does

- zero-shot segmentation and initial pose (CNOS · FoundPose · MegaPose, as plugins)
- depth-based refinement with an independent acceptance gate
- symmetry-aware evaluation and fusion
- calibrated multi-view SE(3) fusion
- confidence / failure prediction with calibration analysis
- *(stretch)* uncertainty-driven next-best-view over real camera views

"RT" refers to one explicit budget: the **update path** (refine → associate → fuse → confidence for one
object, given cached detections) targets p95 < 200 ms; the full pipeline latency is measured and reported,
not promised ([ADR-0004](docs/adr/0004-two-latency-budgets.md)).

## Key results

BOP19 average recall on the T-LESS test targets (1,000 images, 6,423 instances), official
`bop_toolkit` numbers; single view, no refinement yet ([details](docs/results_alpha_tless.md)).

| Row | Masks | Coarse pose | AR | VSD | MSSD | MSPD |
|---|---|---|---|---|---|---|
| A0 upper bound | ground truth | FoundPose | **59.2** | 45.7 | 50.1 | 81.9 |
| A1 baseline | CNOS | FoundPose | **35.4** | 27.4 | 30.1 | 48.6 |
| A5 comparator | CNOS | MegaPose (multi-hypothesis, RGB) | **48.2** | 45.6 | 45.9 | 53.0 |

Reading: FoundPose-coarse is right in the image plane (MSPD 82 with GT masks) but off in depth
(VSD/MSSD ≈ 50), which is what depth refinement targets next; CNOS costs 24 points, mostly by
missing or mis-ranking the target instance (15 % of targets get no prediction, AR 1 % below 30 %
visibility); MegaPose's built-in refiner closes most of the depth gap on the same masks.

| Dataset | AR (VSD/MSSD/MSPD) | Δ from refinement | Δ from 3-view fusion | Update-path p95 |
|---|---|---|---|---|
| T-LESS | 35.4 (27.4 / 30.1 / 48.6) | TBD (Beta) | n/a | TBD |
| XYZ-IBD | TBD | TBD | TBD | TBD |

## Project documents

- [`docs/DECISIONS.md`](docs/DECISIONS.md) — the single decision record: scope, architecture, milestones, ablations, risks
- [`docs/MILESTONES.md`](docs/MILESTONES.md) — the dated milestone plan: weekly tasks, exit criteria, slip policy
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
