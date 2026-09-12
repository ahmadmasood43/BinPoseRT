# BinPoseRT

Reliability-aware multi-view 6D pose estimation for robotic bin picking.

Given one or more calibrated RGB-D views of a cluttered bin and CAD models of the parts, BinPoseRT
returns, for every visible physical object, a world-frame pose `T_world_object`, a **calibrated
confidence** that the pose is correct, and a **verdict** (`accept` / `reject` / `request_view`).

> Status: **Foundations complete** (increment 1 of 6). No results yet — see [docs/DECISIONS.md](docs/DECISIONS.md)
> for the milestone plan. Numbers in this README will only ever come from `outputs/`.

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
images on a GPU machine (`docker/`, from increment 2) and their outputs are cached under `outputs/`.

## Datasets

BOP format throughout. `tools/download_bop.py` fetches T-LESS models; large datasets (XYZ-IBD, IPD) are
downloaded on the GPU machines only.

## Project structure

See [D18 in DECISIONS.md](docs/DECISIONS.md#d18--repository-layout).

## Acknowledgements

BOP benchmark and toolkit · FoundPose · MegaPose · CNOS · T-LESS · XYZ-IBD · Open3D · OpenCV.

## License

MIT — see [LICENSE](LICENSE).
