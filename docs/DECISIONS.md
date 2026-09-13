# BinPoseRT — Decision Record

This is the single authoritative record of what BinPoseRT is, what it is not, and why.
It was produced on 2026-09-12 by interrogating the research plan in
[`docs/source/BinPoseRT-research-plan.pdf`](source/BinPoseRT-research-plan.pdf) decision by decision.
The PDF is a *recommendation* and is cited as a source; this file is the *decision*.

- Vocabulary lives in [`CONTEXT.md`](../CONTEXT.md). Code and docs use those terms and no others.
- The four decisions that are hard to reverse also have ADRs in [`docs/adr/`](adr/).
- When a decision changes, edit it here in place, note the date, and update any affected ADR.

## 1. Project definition

**BinPoseRT** is a reliability-aware RGB-D perception system for model-based 6DoF pose estimation of
textureless industrial objects in cluttered bins. Given one or more calibrated RGB-D Views of a bin and
CAD models of the target objects, it produces, for every visible physical object, a world-frame pose
`T_world_object`, a calibrated Confidence that the pose is correct, and a Verdict
(`accept` / `reject` / `request_view`).

Research questions (from the PDF, kept verbatim in spirit):

| ID | Question |
|---|---|
| RQ-A | Does depth-based refinement materially improve zero-shot RGB pose estimates, and in which initial-error / visibility / symmetry regimes? |
| RQ-B | Which local registration strategy (point-to-plane ICP, robust ICP, GICP) is most robust in clutter? |
| RQ-C | How much does fusing 2–4 calibrated views improve accuracy, stratified by visibility, and where does it saturate? |
| RQ-D | Can a small calibrated model predict pose failure reliably (ROC-AUC, Brier, ECE, risk–coverage)? |
| RQ-E | *(stretch)* Can uncertainty-driven next-best-view selection match fixed multi-view acquisition with fewer views? |
| RQ-F | *(stretch)* What accuracy is retained after C++/TensorRT optimisation, measured as (AR, median, p95, VRAM)? |

## 2. Decisions

### D1 — Compute strategy: develop and test on a CPU laptop, run heavy compute on remote CUDA machines
- The development laptop has **no NVIDIA GPU** (Intel iGPU + AMD Radeon 540X, 15 GB RAM, 8 cores).
  Several remote machines with strong NVIDIA GPUs are available over SSH.
- Workflow: write and unit/integration-test every component locally, then move the repo to a GPU machine
  for full inference, model fitting and benchmarking.
- **Consequence:** every module must be testable without a GPU. Neural estimators sit behind plugin
  interfaces and their outputs are cached to disk in a defined format (D12), so the
  geometry / fusion / confidence / NBV stack runs from fixtures locally. Nothing in `binposert/` may
  import `torch` at module top level.

### D2 — Purpose and timeline: portfolio for job / MSc applications, ~16 weeks
- Target: polished public repo, 6–10 page paper-style report, 60–90 s demo video by early January 2027
  (~14–20 h/week from mid-September 2026).
- Must-have milestones: Foundations → Alpha → Beta → Gamma → Delta (§3). Stretch: Epsilon, Deployment.
- Audience: a technical reviewer with two minutes, then thirty. Honest ablations and failure analysis are
  worth more than SOTA claims.

### D3 — Initial-pose back ends: FoundPose (primary) + MegaPose (second)
- Both implement one `PoseEstimator` plugin interface: `(rgb, depth?, K, Detection) → list[PoseHypothesis]`.
- FoundPose: explicit 2D–3D correspondences → reprojection error and inlier counts become QualitySignals;
  training-free. MegaPose: render-and-compare comparator and alternative refiner (the FoundPose paper
  reports the two are complementary).
- No third estimator (FoundationPose, SAM-6D) before milestone Delta is complete; if added later it is a
  comparator row only.
- Each estimator lives in its own container on the GPU machine (D15); the core consumes cached outputs.
- Found during Alpha (2026-09-13): the public FoundPose release (`3103473b`) is the *coarse* pipeline only
  (no featuremetric refinement), which is exactly the `stage=coarse` input Beta consumes; its `score` is a
  many-to-many inlier ratio and is cached as `pose_score`. The MegaPose adapter caches *all* scored
  hypotheses of `megapose-1.0-RGB-multi-hypothesis` (coarse classifier → MegaPose's own RGB refiner →
  scorer) as `stage=coarse`; depth is never given to it, so Beta's depth Refinement is the only depth user.
  Adapters run the upstream scripts unchanged and only convert formats (`adapters/*_cli.py`).

### D4 — Segmentation: ground-truth masks + CNOS; trained fast segmenter deferred
- `Segmenter` plugin interface: `(rgb, K, object_ids) → list[Detection]`.
- In scope: `GroundTruthSegmenter` (BOP `mask_visib`) and `CNOSSegmenter` (SAM + DINOv2 template
  matching; GPU only; outputs cached).
- A trained lightweight instance segmenter is stretch — needed only for the deployment / latency branch
  (ablations A10, A11).
- Alpha (2026-09-13): the `Segmenter` signature takes the `View` (rgb, `K` and the BOP ids together) —
  `segment(view, object_ids) → list[Detection]`. "CNOS" means the **official BOP'23 default detections**
  (`cnos-fastsam_<dataset>-test.json`, the same file FoundPose and most zero-shot entries use) imported by
  `adapters/cnos_cli.py` without a GPU (`segmenter.variant=bop23_default`); running the pinned CNOS repo
  (`variant=run`, `docker/cnos/`) produces the same JSON format and is kept for datasets without official
  detections (XYZ-IBD). Detection ids are the rank by score within an image; GT detections use `gt_index`.

### D5 — Datasets: T-LESS + XYZ-IBD must-have; ITODD-MV and IPD stretch
- **T-LESS** — development/debug set: single-view, 30 symmetric textureless objects, public GT, small.
- **XYZ-IBD** — centrepiece: multi-view, calibrated extrinsics, reflective bin clutter, public train/val GT.
  Lives only on GPU machines (100+ GB). Used for Gamma and Delta.
- ITODD-MV (hidden test GT → BOP server) and IPD (13-camera, very large) are breadth rows if time allows.
- All datasets are read through one BOP-format loader; a new dataset is a config entry, not code.
- Found during Foundations: every image in a BOP scene directory is a calibrated View of the *same static*
  scene (`cam_R_w2c`/`cam_t_w2c`), so T-LESS is usable for multi-view experiments too by selecting
  several image_ids of one scene. XYZ-IBD remains the multi-view centrepiece; T-LESS becomes the
  multi-view *debug* set.
- The laptop holds only T-LESS subsets and the committed mini fixture. Alpha: the HF mirror ships
  `tless_test_primesense_bop19.zip` (0.83 GB) — exactly the 1000 BOP'19 target images — so the laptop
  carries the full evaluation subset and only estimator inference needs the GPU machine.

### D6 — Language split: Python first; C++ only for *measured* hot paths, after Delta  → [ADR-0001](adr/0001-python-first-cpp-after-profiling.md)
- Foundations → Delta entirely in Python (NumPy, SciPy, Open3D, OpenCV, scikit-learn).
- After Delta: profile; port the 1–3 slowest geometry stages (candidates: visibility-aware model
  cropping, ICP inner loop, depth → cloud) to C++17 via pybind11 as drop-in replacements behind the same
  Python signatures; report before/after latency. `cpp/` does not exist until then.
- CUDA kernels are not planned.

### D7 — Core vocabulary
Defined in [`CONTEXT.md`](../CONTEXT.md). Non-negotiable rules:
- "instance" is banned as a type or variable name (use Detection or ObjectTrack).
- Raw estimator scores are QualitySignals, never "confidence".
- `T_a_b` maps points expressed in frame `b` into frame `a`; `T_world_object = T_world_camera @ T_camera_object`.

### D8 — Refinement acceptance gate: rendered-silhouette IoU vs mask + displacement cap
- After registration, render the ObjectModel at the refined pose and compare its silhouette with the
  Detection mask (IoU, boundary error). ICP never optimises this quantity, so the check is independent.
- Hard cap: reject when the refined pose moved more than `α · diameter` in translation or `β°`
  (symmetry-aware) in rotation from the coarse pose. Defaults `α = 0.25`, `β = 30°`, tuned on T-LESS val.
- A rejected refinement returns the coarse hypothesis unchanged with a `rejection_reason`; IoU and
  displacement are stored as QualitySignals either way.
- Renderer: Open3D `RaycastingScene` (CPU, Embree) for depth/silhouette rendering and visibility-aware
  cropping — headless, no EGL/OSMesa. Measured 22 ms for 640×480 on the laptop. A GPU rasteriser may
  replace it later behind the same `render_depth(mesh, T_camera_object, K, size)` signature.
- Pixel convention: integer pixel coordinates are pixel centres (OpenCV/BOP). Open3D shoots rays through
  `(u+0.5, v+0.5)`, so the renderer shifts the principal point by half a pixel; rendered depth then
  unprojects exactly with `unproject_depth`. Depth is z along the optical axis, not ray length.

### D9 — Symmetry: one flat SymmetryGroup per ObjectModel; continuous axes discretised  → [ADR-0003](adr/0003-discretised-continuous-symmetries.md)
- Load `symmetries_discrete` and `symmetries_continuous` from BOP `models_info.json`. Expand each
  continuous axis into N evenly spaced rotations (default N = 36) and append to the discrete list. The
  identity is always element 0.
- One rule everywhere (fusion, dispersion, NBV, angular error): given a reference pose, replace every
  other pose `T` by `T @ S*` with `S* = argmin_S d_SE(3)(T_ref, T @ S)`.
- Alignment is a right-multiplication in the **object** frame and happens before any averaging.

### D10 — Multi-view fusion: weighted SE(3) mean + joint multi-view ICP polish
1. `T_world_object_i = T_world_camera_i @ T_camera_object_i` for every associated hypothesis.
2. Reference = highest-weight hypothesis; symmetry-align the rest to it (D9).
3. Weighted Lie-algebra mean: `ξ_i = log(T_ref⁻¹ T_i)`, `T_fused = T_ref · exp(Σ wᵢ ξᵢ / Σ wᵢ)`.
   Initial weights: hand-set product of segmentation score, ICP fitness, depth coverage, visible fraction;
   later optionally Model H's probability (D11).
4. Joint polish: one ICP run of the visibility-cropped model against the union of every view's masked
   world-frame point cloud, gated exactly as in D8. Reuses the Refinement module unchanged.
- Ablation rows: best single view / mean only / mean + joint ICP.
- CosyPose-style joint render-based optimisation is out of scope.

### D11 — ConfidenceModel: two calibrated logistic models, one success criterion
- **Success criterion (project-wide):** `MSSD(T_pred, T_gt) < 0.1 · diameter` — BOP-standard and
  symmetry-aware. ADD/ADI thresholds are reported as secondary numbers only.
- **Feature schema** = QualitySignals, a fixed, versioned field list. A missing signal is NaN, imputed with
  a constant plus an indicator flag, so single-view and multi-view rows share one schema.
- **Model H** (per refined PoseHypothesis, camera frame): logistic regression on QualitySignals. Feeds
  fusion weights and NBV uncertainty.
- **Model F** (per FusedPose): logistic regression on aggregated track signals (weighted-mean signals,
  view count, hypothesis dispersion, joint-ICP fitness, multi-view residual). Its output is the published
  Confidence and drives the Verdict.
- Fitted on held-out *val* splits; evaluated with ROC-AUC, PR-AUC, Brier, ECE, reliability diagram and
  risk–coverage curve. MLP only if logistic clearly under-fits.
- Verdict thresholds `τ_acc`, `τ_rej` chosen on val for a target precision and reported.

### D12 — Execution model: stage DAG with a content-addressed disk cache  → [ADR-0002](adr/0002-stage-dag-content-addressed-cache.md)
- Stages: `segment → coarse_pose → refine → associate → fuse → confidence → nbv → evaluate`.
- Each stage is a pure function of (upstream artefacts, stage config, stage version string) and writes to
  `outputs/<dataset>/<split>/<stage>/<hash>/`, `hash = sha256(upstream hashes + config + version)`.
  A stage is skipped when its directory contains a `_SUCCESS` marker.
- Artefacts: Parquet/JSON tables for Detections, PoseHypotheses, ObjectTracks, FusedPoses (4×4 as 16
  floats, QualitySignals as columns); masks as PNG; BOP CSV from `evaluate`.
- GPU stages (`segment` = CNOS, `coarse_pose` = FoundPose / MegaPose) run remotely; their output
  directories are rsync'd to the laptop and every downstream stage runs locally.
- Configuration: Hydra. One `configs/experiment/A<n>.yaml` per ablation row.
- Every run writes `run_manifest.json`: git commit, config hash, dataset version, object/camera subset,
  checkpoints, GPU/CPU, CUDA version, seed, per-stage timings, peak memory.
- Implemented in Alpha (2026-09-13), `binposert/pipeline/` + `binposert/artefacts.py`:
  - The table schemas are frozen as column lists in `types.py` (`DETECTION_COLUMNS`,
    `POSE_HYPOTHESIS_COLUMNS`, `ARTEFACT_SCHEMA_VERSION = 1`); `artefacts.py` holds the writers / readers
    and imports only numpy · pandas · imageio so the estimator containers install the core with
    `pip install --no-deps`. Readers refuse a directory without `_SUCCESS` or with missing columns.
  - `hash = sha256({stage, version, config, dataset identity, upstream hashes})[:16]`. Dataset identity
    is `{name, split, models_dir, targets, scene_ids, image_ids, max_images_per_scene}`; keys named `root`
    or starting with `_` never enter a hash, so a cache produced on the GPU machine keys identically on the
    laptop. A stage directory without `_SUCCESS` is wiped and re-run.
  - GPU stages (`impl: gpu`) are never executed by the runner: it writes `adapter_request.json` into the
    planned directory, prints the adapter command and exits with code 2. The adapter fills the directory
    from that request (dataset identity, image keys, object targets, upstream dirs) and the run resumes
    after `rsync`. `evaluate` consumes `pose_hypotheses` from the *latest* stage producing them, so
    `refine` slots in without touching the evaluator.
  - Every run also writes `outputs/<dataset>/<split>/runs/<experiment>/<stamp>/` with
    `run_manifest.json`, `report.{json,md}` and the BOP CSV under bop_toolkit's naming rule
    `<method>_<dataset>-<split>.csv` (method = `<experiment>-<segmenter>-<estimator>`).
  - Datasets are covered through BOP `test_targets_bop19.json` when the dataset config names one
    (T-LESS: 1000 images), optionally narrowed by `scene_ids` / `image_ids` / `max_images_per_scene`.

### D13 — Latency: two explicit budgets, numbers over adjectives  → [ADR-0004](adr/0004-two-latency-budgets.md)
- **Full pipeline** (`segment` → `confidence`, one Scene, all objects): measured on one documented GPU
  machine; stage-level median / p90 / p95 and peak VRAM/RAM reported. No target promised.
- **Update path** (`refine` → `associate` → `fuse` → `confidence` for one ObjectTrack, given cached
  Detections and coarse hypotheses): **target p95 < 200 ms** in Python, < 50 ms after C++ ports. This is
  what "RT" refers to and the README says so in one sentence.
- Protocol frozen in `configs/benchmark.yaml`: 50 warm-up, 1000 timed, batch 1, CUDA synchronised,
  `perf_counter` per stage. The accuracy–latency Pareto plot (x = p95, y = BOP AR) is a required figure.

### D14 — Active view (stretch): NBV over the Scene's real, not-yet-used Views; pick = transform chain
- Candidate Viewpoints are the remaining real Views of a multi-view Scene (XYZ-IBD / IPD). NBV *unlocks*
  an existing View; it never renders an unseen camera. Synthetic free-camera NBV is out of scope.
- Score: render the top-K aligned hypotheses of an uncertain ObjectTrack into each candidate View;
  `U(v)` = mean pairwise silhouette disagreement, `V(v)` = predicted visible fraction; motion and
  collision costs are zero. Choose the argmax.
- Loop: 1 View → fuse → Verdict; on `request_view` unlock the NBV choice; repeat until `accept`,
  `reject` or exhaustion. Baselines: fixed-1, fixed-2, fixed-all, random-next.
- **Simulated pick** = `T_robot_gripper = T_robot_world @ T_world_object @ T_object_gripper` with a
  hand-authored grasp pose per ObjectModel, visualised in Open3D. No physics simulator unless everything
  else is finished.
- Consequently BlenderProc synthetic data is not a dependency of any milestone; it is an optional
  sim-to-real experiment (A11) at the very end.

### D15 — Environments: uv for the core; one Docker image per external estimator
- Core: `pyproject.toml`, Python ≥ 3.10, managed with uv, `uv.lock` committed. `torch` is an optional
  extra (`binposert[gpu]`) and is never imported by core modules.
- `docker/cnos/`, `docker/foundpose/`, `docker/megapose/`: one Dockerfile each, pinning the upstream
  repo commit, plus a thin `adapters/<name>_cli.py` that reads BOP data and writes D12 artefacts.
- `bop_toolkit` is **not** a core dependency: it pins `numpy<2`, `opencv-python` (conflicts with the
  headless wheel) and `python<3.13`. The core implements ADD / ADI / MSSD / MSPD / VSD itself in
  `binposert/evaluate/metrics.py` (unit-tested against hand-computed cases) and writes BOP-format CSVs;
  the *official* numbers are produced by running `bop_toolkit` in its own environment
  (`tools/bop_eval.sh`, GPU machine) on those CSVs and are the ones reported.
- Laptop bootstrap: install uv → `uv sync` → `uv run pytest`. CI runs lint, type-check and tests on CPU.

### D16 — Testing: synthetic-truth tests for every geometry stage + one tiny BOP fixture
- `tests/synth/` builds Scenes from primitives / tiny meshes, renders depth and masks at known poses, adds
  controlled noise. Every geometry stage has a "recovers known truth" test:
  refinement (small perturbation recovered, large perturbation rejected), symmetry (137° about a cylinder
  axis → error ≈ 0), fusion (3 noisy views → better than best single view), association (3 copies, 2
  cameras, one occluded → exactly 3 tracks), NBV (disagreement in one view → that view wins),
  confidence (synthetic pass/fail → AUC > 0.9).
- `tests/fixtures/mini_bop/`: committed BOP-format dataset (0.2 MB) built from two decimated T-LESS
  models (obj 1: continuous symmetry, obj 5: discrete): 2 scenes × 2 calibrated Views × 3 objects,
  scene 1 with real occlusion (37–48 % visible). Regenerated byte-identically by `tools/make_mini_bop.py`.
  Exercises loader → evaluate → BOP CSV and the metrics path.
- Budget: `pytest` < 60 s on the laptop, no network, no GPU. GPU tests are marked `gpu` and skipped.
- A frame-convention test runs first.

### D17 — Where decisions live
- `docs/DECISIONS.md` (this file): the single owned decision record.
- `CONTEXT.md`: glossary only, no implementation details.
- `docs/adr/`: ADRs 0001–0004 only, for the decisions that are hard to reverse.
- README links to all three; the PDF stays in `docs/source/` as a cited source.

### D18 — Repository layout
```
BinPoseRT/
├── README.md  CONTEXT.md  LICENSE  pyproject.toml  uv.lock
├── docs/            DECISIONS.md  adr/  source/  report/ (later)
├── configs/         config.yaml  dataset/ segmenter/ estimator/ refiner/ fusion/ confidence/ nbv/
│                    experiment/A0..A9.yaml smoke.yaml  benchmark.yaml   (singular group names so the
│                    CLI reads `experiment=A0 dataset=tless`)
├── binposert/       core package (CPU-only imports)
│   ├── types.py         View, Scene, ObjectModel, SymmetryGroup, Detection, PoseHypothesis,
│   │                    QualitySignals, ObjectTrack, FusedPose, Verdict + artefact schema  (D7, D12)
│   ├── artefacts.py     Parquet/PNG writers and validating readers, BOP RLE codec        (D12)
│   ├── transforms.py    SE(3)/so(3) helpers, T_a_b convention, log/exp, distances
│   ├── data/            BOP loader, ObjectModel onboarding, BOP writer
│   ├── render/          Open3D RaycastingScene depth/silhouette renderer          (D8)
│   ├── segment/         Segmenter interface, GroundTruthSegmenter, CNOS cache reader (D4)
│   ├── pose/            PoseEstimator interface, FoundPose/MegaPose cache readers  (D3)
│   ├── refine/          depth→cloud, visibility-aware crop, ICP variants, gate     (D8)
│   ├── symmetry/        SymmetryGroup expansion + alignment                        (D9)
│   ├── multiview/       association, SE(3) fusion, joint ICP                       (D10)
│   ├── confidence/      QualitySignals schema, Model H / F, calibration metrics    (D11)
│   ├── active/          NBV scoring + loop                                          (D14)
│   ├── pipeline/        stages, cache, run manifest, Hydra entry                    (D12)
│   ├── evaluate/        BOP CSV writer, localisation protocol, stratified reports
│   └── viz/             overlays, failure galleries, grasp-pose visualisation
├── adapters/        _common.py  cnos_cli.py  foundpose_cli.py  megapose_cli.py  (docker/, GPU)
├── docker/          README.md  cnos/  foundpose/  megapose/                          (D15)
├── tools/           download_bop.py  make_mini_bop.py  run.py  bop_eval.sh  benchmark.py  plot.py
├── tests/           synth/  fixtures/mini_bop/  test_*.py                           (D16)
└── outputs/         (git-ignored) <dataset>/<split>/<stage>/<hash>/
```
Directories are created when their milestone starts. `cpp/` appears only after Delta profiling (D6);
`synthetic/` only if A11 is attempted (D14).

### D19 — Increment plan
Work proceeds in increments, each ending with green tests and a touch to this file:
1. **Foundations** — docs, packaging, `types`, `transforms`, `symmetry`, `render`, BOP loader, mini fixture,
   `evaluate` on GT.
2. **Alpha** — estimator adapters + Docker, stage cache, Hydra run, A0/A1/A5 on T-LESS.
3. **Beta** — `refine` module with gate, A2–A4, stratified before/after analysis.
4. **Gamma** — `multiview` association + fusion, XYZ-IBD loader quirks, A6/A7, extrinsic perturbation sweep.
5. **Delta** — `confidence` models, calibration plots, Verdict, A8.
6. Stretch — Epsilon (`active`), Deployment (profiling, C++ ports, optional TensorRT), Finalisation.

## 3. Milestones

| Gate | Exit criterion | Runs on | Ablations |
|---|---|---|---|
| **Foundations** | `uv sync && uv run pytest` green on the laptop; mini fixture loads → evaluate → BOP CSV; renderer, symmetry, transforms tested | laptop | — |
| **Alpha** — baseline | `tools/run.py experiment=A0 dataset=tless` yields a BOP CSV and deterministic AR from cached FoundPose outputs; A1 with CNOS | GPU adapters once, then laptop | A0, A1, A5 |
| **Beta** — refinement | Before/after refinement AR on T-LESS, stratified by initial-error bin and visibility; gate rejection rate reported | laptop | A2, A3, A4 |
| **Gamma** — multi-view | XYZ-IBD 1/2/3/4-view AR curve with error bars; association correctness on repeated objects | laptop (cached GPU outputs) | A6, A7 |
| **Delta** — reliability | Models H/F fitted on val; ROC/PR/Brier/ECE, reliability diagram, risk–coverage; every FusedPose carries Confidence + Verdict | laptop | A8 |
| **Epsilon** (stretch) | NBV over real views beats random-next on AR-vs-views; grasp transform chain visualised | laptop | A9 |
| **Deployment** (stretch) | Profile → 1–3 C++ ports → update-path p95 reported; optional ONNX/TensorRT of the DINOv2 backbone | GPU machine | A10 |
| **Finalisation** | README with measured numbers, report, video, failure galleries, model card | — | — |

## 4. Ablation matrix

| ID | Seg | Coarse pose | Refine | Sym | Views | Conf | NBV | Question |
|---|---|---|---|---|---|---|---|---|
| A0 | GT | FoundPose | — | eval | 1 | — | — | upper-bound baseline |
| A1 | CNOS | FoundPose | — | eval | 1 | — | — | cost of automatic segmentation |
| A2 | CNOS | FoundPose | pt-plane ICP + gate | ✓ | 1 | — | — | does depth help? (RQ-A) |
| A3 | CNOS | FoundPose | robust ICP + gate | ✓ | 1 | — | — | robust loss in clutter (RQ-B) |
| A4 | CNOS | FoundPose | GICP + gate | ✓ | 1 | — | — | registration comparison (RQ-B) |
| A5 | CNOS | MegaPose | best | ✓ | 1 | — | — | estimator dependency |
| A6 | CNOS | FoundPose | best | ✓ | 2 (mean / mean + joint) | — | — | two-view gain (RQ-C) |
| A7 | CNOS | FoundPose | best | ✓ | 3–4 | — | — | saturation (RQ-C) |
| A8 | CNOS | FoundPose | best | ✓ | best | ✓ | — | failure detection (RQ-D) |
| A9 | CNOS | FoundPose | best | ✓ | dynamic | ✓ | ✓ | active vs fixed (RQ-E) |
| A10 | fast / GT | best | best (C++) | ✓ | best | ✓ | opt | accuracy / latency (RQ-F) |
| A11 | synthetic-trained | best | best | ✓ | best | ✓ | opt | sim-to-real (optional) |

Additional cheap, high-value experiment: perturb XYZ-IBD extrinsics by δt ∈ {0, 1, 2, 5, 10} mm and
δθ ∈ {0, 0.1, 0.25, 0.5, 1}° and plot multi-view AR.

Analysis always stratifies by visibility bins [0.10, 0.30), [0.30, 0.60), [0.60, 1.00] and, for refinement,
by initial error bins [0, 5), [5, 10), [10, 20), [20, ∞) mm.

## 5. Risks

| Risk | Mitigation baked into a decision |
|---|---|
| Estimator environment hell | D15 one Docker per estimator; D12 cached outputs; core never imports them |
| Depth refinement hurts | D8 visibility-aware crop + independent silhouette gate + displacement cap |
| Symmetry corrupts fusion | D9 align in object frame before averaging; cylinder test in D16 |
| Repeated objects mismatched across cameras | D10 gated Hungarian association with unassigned allowed; 3-copy test |
| Calibration error dominates multi-view | extrinsic perturbation sweep; dataset extrinsics trusted as given |
| "Real-time" claim fails | D13 two budgets, numbers only |
| Scope balloons (NBV, synthetic, robot) | D14 NBV over real views, no simulator, synthetic optional-last |
| Frame-convention bugs | D7 naming rule; first test in the suite |
| Laptop cannot run anything real | D1 / D16 every stage runs from cached artefacts and synthetic-truth fixtures |
| Mediocre benchmark score | D2 rigorous ablations and honest failure galleries instead of SOTA claims |

## 6. Deliverables (end of project)

Public repository · Docker images for the three estimators · dataset download/convert scripts · CAD
onboarding tool · benchmark scripts (one command per ablation) · results CSV/JSON + plots · 60–90 s video ·
6–10 page report · model card with hardware, datasets and known failure modes.

## Increment status
| Increment | Status | Notes |
|---|---|---|
| 1 Foundations | **done 2026-09-12** | 30 tests, < 5 s, CPU only; `types`, `transforms`, `symmetry`, `render`, `data`, `evaluate` |
| 2 Alpha | **in progress** — laptop side done 2026-09-13, GPU side pending | `artefacts`, `segment`, `pose`, `pipeline`, `viz`, stratified report, `configs/`, `tools/run.py`, `tools/bop_eval.sh`, three adapters + Dockerfiles, 44 tests < 30 s. Waiting on: Docker builds, T-LESS FoundPose / MegaPose runs, official toolkit numbers, A0/A1/A5 table |
| 3 Beta | not started | |
| 4 Gamma | not started | |
| 5 Delta | not started | |
| 6 Stretch | not started | |

## Change log
- 2026-09-12 — initial record, D1–D19.
- 2026-09-12 — Foundations increment complete; D5 (T-LESS multi-view note), D8 (pixel convention), D16 (fixture facts) updated from implementation.
- 2026-09-12 — D15: bop_toolkit moved out of core deps (numpy<2 / opencv conflict); metrics reimplemented in core, official eval in a separate env.
- 2026-09-13 — Alpha laptop side: D3 (FoundPose is coarse-only, MegaPose hypotheses cached), D4 (official
  BOP'23 CNOS detections imported instead of run; `Segmenter` takes a View), D12 (frozen artefact schema,
  hash rules, `adapter_request.json` handshake, run directory), D18 (`configs/` groups singular,
  `artefacts.py`) updated from implementation.
