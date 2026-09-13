# BinPoseRT

Reliability-aware multi-view 6D pose estimation for robotic bin picking.

Given one or more calibrated RGB-D views of a cluttered bin and CAD models of the parts, BinPoseRT
returns, for every visible physical object, a world-frame pose `T_world_object`, a **calibrated
confidence** that the pose is correct, and a **verdict** (`accept` / `reject` / `request_view`).

> Status: **Alpha in progress** (increment 2 of 6): the laptop-side pipeline, adapters and configs are done;
> the GPU runs that produce the first numbers are not. See [docs/MILESTONES.md](docs/MILESTONES.md).
> Numbers in this README will only ever come from `outputs/`.

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

| Dataset | AR (VSD/MSSD/MSPD) | Δ from refinement | Δ from 3-view fusion | Update-path p95 |
|---|---|---|---|---|
| T-LESS | TBD | TBD | n/a | TBD |
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
```

No GPU, CUDA or Docker is needed to develop and test the core. Neural estimators run in their own Docker
images on a GPU machine (`docker/`) and their outputs are cached under `outputs/`.

## Running an experiment

One ablation row is one command; every run writes a BOP CSV, `run_manifest.json`, `report.{json,md}`
(AR, per-object AR, AR by visibility bin) and a failure gallery under `outputs/`:

```bash
uv run python tools/run.py experiment=smoke                 # CPU end-to-end on the committed fixture
uv run python tools/run.py experiment=A0 dataset=tless      # GT masks · FoundPose
uv run python tools/run.py experiment=A1 dataset=tless      # CNOS masks · FoundPose
uv run python tools/run.py experiment=A5 dataset=tless      # CNOS masks · MegaPose
```

Stages are cached by content hash (`outputs/<dataset>/<split>/<stage>/<hash>/`). When a GPU stage is not
cached the run stops, leaves an `adapter_request.json` and prints the adapter command to run inside the
matching container on the GPU machine ([docker/README.md](docker/README.md)); `rsync outputs/` back and
re-run. `tools/bop_eval.sh <csv>` produces the official `bop_toolkit` numbers.

## Datasets

BOP format throughout. `uv run python tools/download_bop.py tless --parts base models test_primesense_bop19`
fetches the T-LESS BOP'19 test subset (1000 images, 0.9 GB) used by every T-LESS experiment; large datasets
(XYZ-IBD, IPD) are downloaded on the GPU machines only. `tools/sync.sh` moves `data/` and `outputs/`
between the laptop and a GPU machine (code travels through git).

## Project structure

See [D18 in DECISIONS.md](docs/DECISIONS.md#d18--repository-layout).

## Acknowledgements

BOP benchmark and toolkit · FoundPose · MegaPose · CNOS · T-LESS · XYZ-IBD · Open3D · OpenCV.

## License

MIT — see [LICENSE](LICENSE).
