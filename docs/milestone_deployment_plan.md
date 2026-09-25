# Milestone 6b — Deployment: execution plan

The task-level plan for Milestone 6b (A10, answers RQ-F), written to be executed by someone — or
some model — with no access to the conversation that produced it. Read this top to bottom before
touching code.

- The **decision** is [`DECISIONS.md`](DECISIONS.md) (D6, D13, ADR-0001, ADR-0004).
- The **schedule** is [`MILESTONES.md`](MILESTONES.md) §"Milestone 6b".
- The **decision log** is [`milestone_deployment_decision.md`](milestone_deployment_decision.md) —
  **append to it as you work** (see §0.3). This is a hard requirement of the milestone, not a
  formality.
- The **numbers** end up in `results_deployment.md`, written last, quoting only tool-written files.

---

## 0. Before you start

### 0.1 Why this milestone exists

ADR-0004 and D13 define two latency budgets. The only one with a promise attached is the **update
path** — `refine → associate → fuse → confidence` for one ObjectTrack, given cached Detections and
coarse hypotheses — with a target of **p95 < 200 ms**. Delta measured **505 ms median / 739 ms
p95**, 3.7× over. This milestone measures that path properly, makes it faster without changing what
it computes, prices the remaining gap in accuracy, and reports the result honestly whether or not
200 ms is reached.

**Scope, already decided (P1–P5). Do not re-open.**
- Profile-guided **Python** fixes → ICP schedule sweep → accuracy–latency Pareto.
- C++/pybind11 **only if** a Python hot spot survives the fixes (ADR-0001). No ONNX/TensorRT.
- 200 ms is **reported, not required**. The accurate configuration stays the project default;
  A10 is a named operating point.
- Both D13 budgets measured; the full pipeline once, on a 24-image subset.
- T-LESS primary (official BOP AR exists); one XYZ-IBD cross-check row, core AR only.

### 0.2 House rules you must follow

- **One command per experiment row** (D12): `uv run python tools/run.py experiment=A<n> dataset=<d>`,
  writing a BOP CSV, `run_manifest.json` and plots under `outputs/`.
- **Numbers in README / results docs only ever come from `outputs/`.** Curated docs quote
  tool-written files; they never contain a number typed by hand.
- **Tests** (D16): `uv run pytest` green in < 60 s on the laptop, no GPU, no network. Every geometry
  change gets a synthetic-truth test in `tests/`. Real-data verification lives in
  `tools/check_deployment.py`, not in pytest.
- **Lint/type**: `uv run ruff format --check . && uv run ruff check . && uv run mypy binposert`
  must be clean. Line length 100.
- **Milestone shape**, copied from Delta/Epsilon: `tools/run_deployment.sh` (resumable, skips a row
  whose manifest exists, `FORCE=1` overrides, steps gated by a `STEPS` variable) ·
  `tools/deployment_report.py` (reads `outputs/` only; writes markdown + figure + JSON) ·
  `tools/check_deployment.py` (a `Checker` class printing `PASS`/`FAIL`, `sys.exit(1)` on any
  failure) · `docs/results_deployment.md` · the decision log · updates to `DECISIONS.md`
  (increment status + change log), `MILESTONES.md` (status + status log) and `README.md`.

### 0.3 The decision-log requirement

**Every decision you take gets an entry in
[`milestone_deployment_decision.md`](milestone_deployment_decision.md) at the moment you take it**,
numbered `P9`, `P10`, … (P1–P8 are already written), with this shape:

```markdown
## P<n> — <short title>

**Context.** What forced a choice.
**Decision.** What you did.
**Why.** The reason, with the measurement or file:line behind it.
**Alternative rejected.** What you did not do, and why not.
```

Also add the one-line summary to the table at the top of that file. Decisions that turn out wrong
keep their entry; the reversal is a new entry citing the first. A decision that changes a published
number or a repo-wide invariant additionally gets a dated line in the `DECISIONS.md` change log.

### 0.4 Host facts that will bite you (k8s-control)

- 16 cores, 31 GB RAM, one Pascal TITAN X (sm_61, cu118). `uv` is at `~/.local/bin` — not on PATH by
  default: `export PATH="$HOME/.local/bin:$PATH"`.
- **`pkill -f <pattern>` kills this session's own shell** because the pattern matches the shell's own
  command line. Use `pgrep -x <name>`, or the bracket trick on `ps` output
  (`ps -eo pid,args | grep "[s]pawn_main"`), and kill the resulting PIDs **in a command that mentions
  the process name nowhere else**. Kill in one command, restart in the next.
- Jobs started from a session die with it: launch long work as
  `(setsid nohup <cmd> > <log> 2>&1 < /dev/null &)`.
- **Never wait with `until ! pgrep -f <pattern>`** — the loop's own command line matches the pattern
  and it never exits. Wait on a *file* the job writes at the end.
- Keep stage parallelism at half the cores (`default_workers()` = 8). Never run a GPU stage beside a
  full CPU stage. The host had five unclean reboots under sustained load during Gamma (G21).
- **Open3D 0.19's parallel `cast_rays` corrupts output under multi-process load.** `RAY_THREADS = 1`
  (`binposert/render/raycast.py:28`), `CAST_RETRIES = 3`, the `_cast_checked` validation block
  (`:36-76`) and the render-level retry (`:110-119`) are load-bearing. The ROI work below makes them
  cheaper; it must not make them weaker. `tests/test_render.py` must stay green.

### 0.5 Where things are

| What | Where |
|---|---|
| Refinement | `binposert/refine/{refiner,icp,gate,cloud,crop,depth_init}.py` |
| Raycast renderer | `binposert/render/raycast.py` |
| Stage DAG, cache, hashing | `binposert/pipeline/{stages,cache,artefacts,manifest,pool}.py` |
| Refine stage adapter | `binposert/pipeline/refine_stage.py` |
| Symmetry | `binposert/symmetry/group.py` |
| Existing profiler | `tools/profile_update_path.py` |
| Closure-check pattern to copy | `tools/check_epsilon.py:36-43` (the `Checker` class) |
| Runner pattern to copy | `tools/run_delta.sh` (incl. the `official()` helper + `BOP_TARGETS`) |
| Report pattern to copy | `tools/nbv_report.py`, `tools/confidence_report.py` |
| Results-doc house style | `docs/results_epsilon.md`, `docs/results_delta.md` §8 |

---

## Phase 0 — Regenerate the profile before optimising anything

**Read decision P6 first.** The stored profile
(`outputs/tless_update_path_profile.json`, quoted in `README.md` and `docs/results_delta.md` §8)
is internally inconsistent in three ways and must not be used to rank fixes:

1. `43.392 s` traced over 303 refine calls = **143 ms/call**, against a measured median of
   **284 ms/call** in the same run. Profiling makes code slower, not twice as fast.
2. The table is cumulative-ordered, yet `Refiner.refine` is absent while its callees (`register`
   21.842 s, `check_gate` 303 calls, `_pinhole_rays` 1521 calls) are all present with correct counts.
   A caller's cumulative time always exceeds its callee's.
3. `cast_rays` is absent entirely, although cProfile traces pybind calls — `registration_icp` and
   `create_rays_pinhole` both appear.

Already ruled out: the `prof.enable()`/`disable()` bracket at `tools/profile_update_path.py:148-152`
is not the explanation by itself (an isolated reproduction of that pattern over 304 cycles traces
the caller correctly and ranks it first), and the artefact was produced by the current code (one
commit in the file's history, `7c9c4ec`; file mtime 2026-09-18 11:23 predates the artefact's
2026-09-19 01:03).

### Tasks

- [ ] **P0.1** Re-run the profiler unchanged and diff the new table against the stored one:
      `OMP_NUM_THREADS=1 uv run python tools/profile_update_path.py --dataset tless --row A8_k4
      --n-groups 12 --profile-groups 6 --out /tmp/reprofile.json`. Does `Refiner.refine` appear now?
      Does `cast_rays`? Does the traced total match `median_ms × n`?
- [ ] **P0.2** If entries are still missing, add a parallel per-callee wall-clock instrument
      (explicit `perf_counter` around the render, the ICP calls, the cloud builds and the gate inside
      `Refiner.refine`, behind a debug flag) so the accounting closes to ~100 % regardless of what
      cProfile reports. The deliverable is a table that **sums to the measured time**.
- [ ] **P0.3** Publish the corrected table. If it differs materially from the stored one, correct
      `docs/results_delta.md` §8 and the `README.md` paragraph that quotes it, with a dated
      `DECISIONS.md` change-log entry. A wrong published number is fixed, not inherited.
- [ ] **P0.4** Write the decision entry (`P9`) recording what the anomaly turned out to be.
- [ ] **P0.5** **Rank the Phase 2 fixes from the corrected table**, not from the list below.

### Working hypothesis (confirm or kill in P0.1–P0.2)

Refinement is **render-bound**, not ICP-bound. Direct measurement on this host (quiet,
`OMP_NUM_THREADS=1`, T-LESS obj 5 = 22 225 faces, 720×540, mean of 20):

| op | full frame | ROI 164×143 |
|---|---|---|
| `MeshRenderer.render` total | **55.5 ms** | **3.60 ms** |
| ↳ `cast_rays` | 18.7 | 2.53 |
| ↳ `_pinhole_rays` | 2.07 | 0.14 |
| ↳ `np.sum(normals*dirs, -1)` (`raycast.py:136`) | 5.88 | — |
| ↳ `rays.numpy()[...,3:6].astype` (`:131`) | 1.53 | — |
| ↳ two `@ R.T` (`:132`, `:134`) | 2.59 | — |
| `unproject_depth` | 0.80 | — |
| `cv2.distanceTransform` | 1.36 | — |

Five renders per refine ≈ 275 ms against three `registration_icp` calls ≈ 69 ms. T-LESS detection
bounding boxes are a median **0.73 %** of the frame (p90 5.3 %, p99 63 %). If this holds, the ROI
(F1) is the milestone's main lever and the ICP schedule is secondary.

**Acceptance:** a profile table that accounts for ~100 % of the measured refine time, published, with
any stale published numbers corrected.

---

## Phase 1 — The benchmark harness (D13's frozen protocol)

### `configs/benchmark.yaml` (new)

Not a Hydra group; a protocol file read by `tools/benchmark.py`, versioned, and copied verbatim into
every result JSON. See P7 for why a sample is an ObjectTrack.

```yaml
version: 1
protocol:
  warmup_samples: 50
  timed_samples: 1000
  batch: 1
  clock: perf_counter
  cuda_sync: true
  thread_env: {OMP_NUM_THREADS: 1, OPENBLAS_NUM_THREADS: 1, MKL_NUM_THREADS: 1, NUMEXPR_NUM_THREADS: 1}
  max_load_average: 2.0
  percentiles: [50, 90, 95, 99]
update_path:
  unit: object_track
  stages: [refine, associate, fuse, confidence]
  attribution: {refine: sum_over_members, fuse: per_track, associate: group_share, confidence: group_share}
  row: A8_k4
  dataset: tless
  group_order: seeded_shuffle
  seed: 0
  target_p95_ms: 200
full_pipeline:
  stages: [segment, coarse_pose, refine, associate, fuse, confidence]
  unit: image
  subset: {scenes: 2, images_per_scene: 12}
  outputs_root: outputs/bench
  run_external: true
  repeats: 1
memory:
  sample_interval_s: 0.2
  rss: [self, children]
  vram: nvidia_smi_compute_apps
```

### `tools/benchmark.py` (new) — wraps `tools/profile_update_path.py`, does not replace it

That tool already solves the hard parts: cached inputs held outside the timer, per-track attribution
of group-level costs, cProfile groups excluded from timings. Refactor rather than reimplement.

- [ ] **P1.1** Extract from its `main()` a generator
      `iter_track_samples(...) -> Iterator[TrackSample]` where `TrackSample` carries
      `{scene_id, n_members, refine_s, associate_s, fuse_s, confidence_s, total_s}`.
- [ ] **P1.2** Add `p99_ms` to `pct()` (`tools/profile_update_path.py:65-75`). Existing keys keep
      parsing, so the Δ16 JSON stays readable.
- [ ] **P1.3** `main()` keeps its current job as the cProfile hot-spot tool, with Phase 0's fixes.
- [ ] **P1.4** `tools/benchmark.py` owns the protocol: read the YAML, set the thread env, **refuse to
      start above load average 2.0**, drive the generator to 50 + 1000 samples, sample memory, write
      `outputs/<dataset>_benchmark_<row>.json`. Import the replay the way `tools/nbv_report.py:28`
      already imports from `tools/`.

**Output JSON** (`schema_version: 2`, a superset of the Δ16 schema): `protocol` verbatim,
`git_commit`, `host {cpu, cores, ram_gb, gpu}`, `dataset`, `row`, `refiner_params`,
`n_groups_consumed`, `per_stage_ms`, `per_track_total_ms`, `per_hypothesis_refine_ms`,
`hypotheses_per_track`, `target_p95_ms`, `memory {peak_rss_self_mb, peak_rss_children_mb,
peak_vram_mb, vram_baseline_mb}`, `load_average_start`, `load_average_end`, `single_threaded`,
`cuda_sync`.

- **Peak RAM:** `getrusage(RUSAGE_SELF|RUSAGE_CHILDREN).ru_maxrss` — the idiom already at
  `binposert/pipeline/manifest.py:46` — plus a 0.2 s sampler over the process tree (`VmHWM`/`VmRSS`).
- **Peak VRAM:** 0.2 s sampler on
  `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits`, max over the
  benchmark's process tree, with an idle baseline read first.
- **CUDA sync, honestly:** there is **no CUDA on the update path**. Report
  `cuda_sync: "not_applicable (no CUDA on the update path)"` and `peak_vram_mb: 0` rather than
  claiming a synchronised measurement that never happened. For the full pipeline the estimator
  subprocess boundary already synchronises the stage timing; if you add `torch.cuda.synchronize()`
  to the adapters' internal `time_s`, **gate it on `BINPOSERT_CUDA_SYNC=1`** — `time_s` is written
  into `hypotheses.parquet`, so an ungated change would alter artefact bytes under an unchanged hash.

### Full pipeline, measured once

- [ ] **P1.5** `tools/benchmark.py --full-pipeline`: write `outputs/bench/targets_bench.json`
      (2 scenes × 12 images, seeded, drawn from the dataset's real targets so `inst_count` stays
      correct), then run `tools/run.py experiment=A8 dataset=tless
      dataset.targets=outputs/bench/targets_bench.json outputs_root=outputs/bench run_external=true`
      as a subprocess. Read stage timings back from the `run_manifest.json` it writes
      (`record_stage`, `binposert/pipeline/stages.py:562`) and per-image timings from each stage's
      own artefact (`time_s` columns, `refine_details.seconds`).

> **`outputs_root=outputs/bench` is load-bearing.** `stage_config`
> (`binposert/pipeline/stages.py:497`) hashes only the config sections — the dataset name appears
> solely in the *cache path*. A subset config that kept `name: tless` / `split: test_primesense`
> would alias the existing caches, hit `_SUCCESS` and benchmark nothing.

Cost ≈ 30–45 min once (CNOS ≈ 19 s/image × 24). Report per-image median / p90 / p95 **with `n = 24`
printed next to every percentile** — D13 promises no target here, but a p95 of 24 samples is the
23rd value and must be labelled as such.

**Acceptance:** `tools/benchmark.py` reproduces a p95 within the sampling interval across two runs on
a quiet host, and the full-pipeline JSON carries stage-level percentiles plus peak VRAM and RSS.

---

## Phase 2 — The Python fixes

**Rank these from Phase 0's corrected table.** The ordering below is the pre-measurement hypothesis.

Two classes, because the class decides whether a cache can be reused (Phase 3).

### Class A — bit-identical, applied unconditionally

| # | Fix | Where | Why it is exact |
|---|---|---|---|
| F7 | Restrict `np.sum(normals_cam * dirs_cam, -1)` to hit pixels | `render/raycast.py:136` | The result on non-hit pixels is discarded two lines later by `normals_cam[~hit] = 0`; T-LESS hit rate ≈ 4 %. Same three-term dot in the same order. Measured **5.88 ms** — the largest single numpy op in a full-frame render. Also speeds `render_scene`, the NBV silhouettes and the galleries. |
| F3 | Build the scene point cloud once per call, not per ICP level | `refine/icp.py:57-58`, `refine/refiner.py:92-129` | `pts_cam`/`normals_cam` are identical across all three levels yet re-wrapped in `Vector3dVector` every level (12 copies/refine). Add an optional `src_pcd: o3d.geometry.PointCloud \| None = None` to `register`, keeping the array signature intact (ADR-0001 wants portable signatures). `registration_icp` does not mutate its source; GICP's `estimate_covariances` does, but idempotently — so A4 gains more and stays exact. |
| F4 | Drop the duplicated `visible_silhouette` | `refine/refiner.py:139` vs `refine/gate.py:87` | Same function, same three arguments, computed twice per refine. Return `mask_refined` on `GateDecision` and use it. |
| F5 | Vectorise the symmetry rotation stack | `symmetry/group.py:62` | 2.6 ms/refine, ~27 `Rotation.from_matrix` per call. `min` over the same multiset is the same float — **but that is a claim about scipy, so prove it with an exact-equality test, do not assume it.** |
| F5b | Same fix in `align_to_reference` | `symmetry/group.py:45` | On the update path via `fuse_track`; this is why `fuse` p95 is **28.6 ms** against a 2.5 ms median — 14 % of the whole budget. Not in the original task list. |

### Class B — numerically different, config-gated

**F1 — ROI (crop-local) refinement. Expected to be the milestone's main lever.**

Measured **15.4×** on the render (55.5 → 3.60 ms), and it drags every dependent per-pixel op with it
— `unproject_depth`, `np.nonzero`, `cv2.dilate`/`erode`, `distanceTransform`, every silhouette
`.sum()`, and the `z_window` comparison that currently runs over the whole depth image
(these are **F6** `gate.boundary_error_px` and **F8** `cloud.py:35-43` / `depth_init.py:35-43` /
`crop.py:36-47`; both ride on F1 and are exact given the crop).

- [ ] **P2.1** New `binposert/refine/roi.py`: frozen `Roi(x0, y0, w, h)` with `K_shifted(K)`
      (subtract `x0`/`y0` from `cx`/`cy` — exact in float64 for integer offsets), `crop(img)`,
      `size`, and constructors `from_bbox(bbox_xywh, pad)` / `from_sphere(T, radius, K, size)`.
- [ ] **P2.2** Thread the bbox that **already exists and is free**: `DETECTION_COLUMNS` carries
      `bbox_x/y/w/h` (`binposert/pipeline/artefacts.py:44-56`), and
      `binposert/pipeline/refine_stage.py:80` already builds `d_img` with those columns but reads
      only `mask_path` at `:86`. Read them there and pass an `Roi` into
      `Refiner.refine(view, det_mask, hyp, roi=...)`. Crop `view.depth` and `det_mask` once.
- [ ] **P2.3** `MeshRenderer.render` needs **no new parameter**: call
      `render(T, roi.K_shifted(view.K), roi.size)`.
- [ ] **P2.4** **Containment must be a guarantee, not an approximation.** ROI = (detection bbox
      dilated by `mask_dilate_px + erode_px + 1`) ∪ (projection of the model's bounding sphere at
      `T_start`, radius `diameter/2`), inflated by `roi_margin_px` (default 8), clipped to the image.
      Compute it once per `refine` call, after the depth initialisation.
- [ ] **P2.5** **Do not inflate by the displacement cap.** `alpha = 0.6` gives a 2.2 d box (≈ 34 % of
      the frame for obj 5 at 700 mm) and destroys the win. Instead check containment per render: if
      the sphere at the current pose leaves the ROI, abort and redo that call at full frame,
      recording `roi_fallback=True` and `roi_px` as new columns in `refine_details.parquet`
      (`run_refine` already appends unknown keys, `refine_stage.py:145-147`). **Report the fallback
      rate — it is a headline number.** T-LESS's p99 bbox is 63 % of the frame, so the guard will fire.

**The subtle hazard — test it, do not assume it.** `scene_cloud` (`cloud.py:47-49`) and
`visible_model_cloud` (`crop.py:50-53`) subsample with `rng.choice(len(pts), max_points)`. The chosen
indices depend on `len(pts)` and on `np.nonzero` ordering. Both are preserved by a crop that contains
the whole mask — **but only then.** An ROI that clips a single valid pixel silently changes which
points ICP sees. Assert element-for-element equality of both clouds, cropped vs full.

**Honest ceiling, already measured:** rendering obj 5 into a 164×143 ROI vs cropping the full-frame
render gives **identical silhouette masks** and **max |Δdepth| = 8.4e-4 mm**. The residue is float32:
Open3D builds ray grids in float32, and the cropped grid differs from the same sub-window of the full
grid in 19 975 of 61 440 direction components, max 2.38e-7. Through ICP a correspondence can flip in
a tie, so rare rows move more than that — Phase 3 measures it on real data.

**F2 — the ray-grid cache is deliberately NOT built.** Worth 2.07 ms of a 55.5 ms render (3.8 %), and
0.14 ms once the ROI lands. It also cannot be bit-identical: Open3D builds the grid in float32 and a
cached-and-rotated grid reproduces it only to ~1 float32 ulp (max Δ 1.19e-7 after rotation, at any
numpy precision). Record this as a closed decision with the two facts a future reader needs:
`create_rays_pinhole` directions are **unnormalised with z ≡ 1** (a hand-rolled replacement that
normalises silently breaks `depth = t_hit * dirs_cam[...,2]` at `raycast.py:133`), and the extrinsic
is applied world→camera so the origin is `T⁻¹[:3,3]`. A per-pose memo is also worthless:
`z_shift_mm == 0` in only 0.6 % of 5 452 cached rows.

### Must not change

`RAY_THREADS = 1` (`raycast.py:28`), `CAST_RETRIES = 3`, the whole `_cast_checked` validation block
(`:36-76`) and the render-level retry (`:110-119`). This is Beta finding #1 — silent corruption under
load, 126/1751 renders raising and 5 silently wrong at 15 workers. The ROI makes this validation
~17× cheaper (it was measured at 0.24 ms full frame and was never the problem); it does not get
weakened. `tests/test_render.py::test_corrupted_cast_is_retried_then_fails_loudly` stays green.

### The C++ question is answered by measurement, not skipped

If, after F1–F8, the survivors are `registration_icp` and `cast_rays` — **both already C++ inside
Open3D** — then ADR-0001's precondition ("port the slowest geometry stages *if* a Python hot spot
survives") is not met, `cpp/` is not created, and **that measurement is the finding for RQ-F**.
Record it as a decision with its numbers and add a dated status note to
`docs/adr/0001-python-first-cpp-after-profiling.md`. Presenting it as a skipped task would be wrong.

---

## Phase 3 — Cache invalidation: a proof, not a promise

**Read decision P8.** 200 run manifests exist (`find outputs/runs -name run_manifest.json | wc -l`),
representing ~13 h of GPU/CPU work.

- `STAGE_VERSIONS["refine"]` **stays `"4"`**.
- Add `roi: str = "none"` and `roi_margin_px: int = 8` to `RefinerParams`
  (`binposert/refine/refiner.py:31-45`), defaulting to current behaviour.
- **Do not edit `configs/refiner/*.yaml`.** The refine hash comes from the YAML section
  (`stage_config` → `hashable_config(cfg["refiner"])`), not the dataclass, so a defaulted field
  cannot move any existing hash; a sweep row passing `refiner.params.roi=bbox` on the command line
  lands in its own cache directory automatically. Flipping the default belongs to Finalisation's
  "re-run every must-have ablation from a clean `outputs/` on a tagged commit".
- Also do not *add* `mask_dilate_px` or `gate.delta_mm` to those YAMLs — they are absent today, so
  adding them changes the hash even at identical values.

### `tools/check_deployment.py --equivalence` — the evidence

Real cached data, not pytest (D16's 60 s budget stays intact).

- [ ] **P3.1** Sample **N = 300** rows from the A8 refine cache's `refine_details.parquet` (5 452
      rows), seeded, stratified by `object_id` × `accepted`.
- [ ] **P3.2** Re-run `Refiner.refine` at current code with `roi="none"` on the same View + mask.
      Assert **exact equality** (`==` on float64, `np.array_equal` on transforms) for: `cand_T_*`,
      the refined `T_*`, `iou_coarse`, `iou_refined`, `boundary_px`, `displacement_mm`,
      `displacement_deg`, `fitness`, `rmse_mm`, `n_correspondences`, `n_scene_points`,
      `depth_coverage`, `z_shift_mm`, `z_init_overlap`, `accepted`, `reason`, and every
      QualitySignal. Exclude `seconds`. **One `PASS` line per column**, house style.
- [ ] **P3.3** Re-run the same 300 with `roi="bbox"` and emit **the tolerance table**: max |Δ| per
      column, fraction of rows with any difference, count of gate-decision flips, max |Δt| mm and
      |ΔR| deg. This table goes verbatim into `docs/results_deployment.md` and is the honest
      statement of what the fast path changes.
      **Pre-committed stop rule: > 1 % gate flips, or any |Δt| > 0.01 mm, demotes the ROI to an
      ablation row rather than the operating point.** Decide this before you see the numbers.
- [ ] **P3.4** **Whole-cache guard:** re-resolve every stage hash of all 200 manifests
      (`resolve_refs`, `binposert/pipeline/stages.py:545` — no stage runs) and assert each equals the
      hash recorded in that manifest, **before and after** the change. Assert specifically that
      A8_k4's refine hash is unchanged. This turns "I believe nothing was invalidated" into a `PASS`
      line over the whole cache.
- [ ] **P3.5** **CI twin** `tests/test_refine_equivalence.py` (< 5 s, synthetic, no data, per D16):
      on the `tests/synth/` scene, `refine(roi="none")` vs `refine(roi="bbox")` within the documented
      tolerance; plus exact-equality tests for every Class-A fix (vectorised vs loop symmetry
      distance; `scene_cloud`/`visible_model_cloud` cropped vs full, **element for element including
      the subsample**; `boundary_error_px` cropped vs full; `unproject_depth` with a shifted K).

---

## Phase 4 — ICP schedule sweep and A10

Every A10 row is the A8 chain at k = 4 (CNOS → FoundPose → refine → associate → mean fusion →
confidence, ranked by Confidence), so the y-axis is directly comparable to A8_k4 (official **75.9**).
One row = one `tools/run.py` invocation with overrides, exactly as `tools/run_delta.sh` does for A8w.

A **coordinate** sweep from the default, not a product:

| axis | values |
|---|---|
| L `corr_dist_factors` | `[0.15,0.08,0.04]` (L3, default) · `[0.15,0.06]` (L2) · `[0.10]` (L1) |
| I `max_iterations` | 30 (default) · 15 · 8 |
| P `max_model_points` = `max_scene_points` | 3000 (default) · 1500 · 750 |

**Rows** (~12; regex for the report tool `^A10_(exact|l(?P<L>\d)_i(?P<I>\d+)_p(?P<P>\d+))$`):

- `A10_exact` — `roi=none`, default schedule. The reference; the checker asserts it reproduces
  A8_k4's per-GT errors exactly.
- `A10_l3_i30_p3000` — `roi=bbox`, default schedule. **The cost of the fixes alone, in AR.**
- One row per axis value: `A10_l2_i30_p3000`, `A10_l1_i30_p3000`, `A10_l3_i15_p3000`,
  `A10_l3_i8_p3000`, `A10_l3_i30_p1500`, `A10_l3_i30_p750`.
- 3–4 combinations on the promising direction: `A10_l2_i15_p1500`, `A10_l2_i8_p750`,
  `A10_l1_i15_p1500`.

The chosen operating point keeps its systematic name and is *declared* the A10 point in the results
doc — no duplicate row.

### How each point gets both a p95 and an AR

- **p95:** `tools/benchmark.py --row A10_<...> --dataset tless`, ~3–4 min per row. Run **serially,
  after every sweep row has finished** — never beside the worker pool. The harness aborts above load
  average 2.0 and records the load at start *and* end. **The replay must pass the same ROI the stage
  passes** (`tools/profile_update_path.py:135-141`), or it benchmarks a different function than it
  evaluates.
- **AR core:** each row's own `evaluate` (~6 min), labelled *core*, as Gamma and Delta label theirs.
- **AR official:** `tools/bop_eval.sh` costs **3 173 s per row** (measured,
  `outputs/runs/A8_k4_tless.bop_eval.log`) and races on the shared `third_party/bop_toolkit` clone.
  So run official for **four** rows only — `A10_exact` (must reproduce 75.9), `A10_l3_i30_p3000`, the
  chosen operating point, and the cheapest frontier point — serialised, overnight, ≈ 3.5 h, using
  `run_delta.sh`'s `official()` helper verbatim (including `BOP_TARGETS=$ev_dir/targets_subset.json`).

### Confidence-model caveat — state it up front

Model H/F (`models/confidence/v1/`) were fitted on the default schedule's signal distribution; every
cheap-schedule row applies them out of distribution. **Do not refit** — D11 fixes the fit protocol,
and a per-point refit makes the y-axis a moving target. Add one control row at the cheapest schedule
with `evaluate.score_signal=pose_score`, to separate "the geometry got worse" from "the confidence
ranking got worse".

### Cost envelope

Per row: refine ~10 min wall at `n_workers = 8`, associate/fuse/confidence seconds, evaluate ~6 min
→ ~17 min. Twelve rows ≈ **3.5 h** unattended, CPU only, from the cached `segment`/`coarse_pose` — so
no GPU stage runs and the "never GPU + full CPU together" rule holds by construction. Plus ~45 min of
serial benchmarking and ~3.5 h of official eval overnight.

**XYZ-IBD cross-check:** one row at the chosen operating point (refine 156 CPU-min → ~20 min wall),
plus its own benchmark run. **Core AR only** — only T-LESS has BOP19 targets. Say so in the results
doc rather than quietly omitting the column.

### `tools/run_deployment.sh`

House shape: `STEPS="equiv sweep bench official xyzibd report"`, skip a row whose manifest exists,
`FORCE=1`, `DATASET`, `N_WORKERS`, `(setsid nohup … &)` for the long legs, and a hard ordering
constraint — **`bench` refuses to start while `sweep` rows are running.** See §0.4 for the process
handling; never `pkill -f`.

---

## Phase 5 — Report

`tools/deployment_report.py` (the `nbv_report.py` / `confidence_report.py` shape: reads `outputs/`
only) writes:

- `outputs/tless_deployment_report.md`
- `outputs/tless_deployment_pareto.png` → curated copy at `docs/figures/deployment_tless_pareto.png`
  (Epsilon convention: the tool writes to `outputs/`, the curated copy lives in `docs/figures/`, and
  `tools/check_deployment.py` asserts both exist — cf. `tools/check_epsilon.py:193-199`)
- `outputs/tless_deployment_pareto.json` — one record per row
  `{row, params:{levels, iterations, points, roi}, ar_core, ar_official, p95_ms, median_ms, p99_ms,
  refine_ms_per_hyp, hyp_per_track, n_samples, roi_fallback_rate}`, so every number in the curated
  doc is quoted from a tool-written file.

**Pareto figure:** x = update-path **p95 per ObjectTrack** (ms), y = **BOP AR** (filled markers =
official, open = core), one point per A10 row, the Pareto frontier as a step line, a dashed vertical
at **200 ms** annotated *"reported, not required"*, `A8_k4` / `A10_exact` annotated as the default,
and the XYZ-IBD cross-check as a distinct marker noted as core AR. Bootstrap CI on AR, as the
Gamma/Epsilon curves do, and the sampling interval of p95 from the 1000 samples.

**`docs/results_deployment.md`** in house style: `# Deployment results — …` → unheaded provenance
block (dates, exit-criterion verdict, the RQ in italics, links to the decision log, the tool-written
source files, the runner and the checker with its result) ending **"Every number here comes from
those files."** → `## Summary` bullets → `## 1. Protocol` → numbered result sections → figures →
`## N. Reproduction` (a `bash` block of the exact commands in order) → `## Limitations`.

The closest existing precedent to copy is `docs/results_delta.md:304-340` (§8 Update-path profile).

---

## Exit criterion

Replaces "Profile → 1–3 C++ ports → update-path p95 reported" in `DECISIONS.md` and `MILESTONES.md`:

> The update path is measured under the frozen protocol (`configs/benchmark.yaml`: 50 warm-up and
> 1000 timed ObjectTrack samples, batch 1, `perf_counter` per stage, CUDA synchronised where CUDA is
> used); the profile-guided **Python** fixes are shown to cost no accuracy — bit-identical where
> claimed, tolerance-bounded and tabulated where not — **without invalidating a single cached
> stage**; the ICP schedule is swept; and the accuracy–latency Pareto (x = update-path p95, y = BOP
> AR) is reported on T-LESS with one XYZ-IBD cross-check row at the chosen operating point. **The
> 200 ms p95 target is reported, not required**: the accurate configuration remains the project
> default and A10 is a named operating point. The full pipeline is measured once on a documented
> image subset for stage-level median / p90 / p95 and peak VRAM / RAM, with no target (ADR-0004). A
> C++/pybind11 port is written **only if a Python hot spot survives the fixes** (ADR-0001); if none
> does, that measurement is the finding and `cpp/` is not created. ONNX / TensorRT export is out of
> scope, dropped with a one-line reason.

Amend the `MILESTONES.md` task list to match: the "Port the 1–3 slowest geometry stages to C++17"
checkbox becomes "C++ port **if** a Python hot spot survives — record the decision either way, with
the profile behind it", and the ONNX/TensorRT line is struck through with its reason.

---

## Deliverables

`configs/benchmark.yaml` · `tools/benchmark.py` · `tools/profile_update_path.py` (refactored:
shared replay + corrected cProfile mode) · `binposert/refine/roi.py` and the ROI path behind
`refiner.params.roi` · the Class-A fixes · `tools/check_deployment.py` (+ `--equivalence`) ·
`tests/test_refine_equivalence.py` · A10 rows (`outputs/runs/A10_*`, ~12 T-LESS + 1 XYZ-IBD) ·
`tools/run_deployment.sh` · `tools/deployment_report.py` → `outputs/{tless,xyzibd}_deployment_report.md`
+ `outputs/tless_deployment_pareto.{png,json}` · `docs/figures/deployment_tless_pareto.png` ·
`docs/results_deployment.md` · `docs/milestone_deployment_decision.md` (P1…Pn) · updates to
`DECISIONS.md` (D6/D13 notes, increment status table, change log), `MILESTONES.md`, `README.md`,
`docs/adr/0001-python-first-cpp-after-profiling.md` (dated status note) · the Phase 0 correction to
`docs/results_delta.md` §8 if the regenerated profile differs.

---

## Risks and responses

| Risk | Response |
|---|---|
| **A cached stage is invalidated by accident** — the most expensive failure here (200 manifests, ~13 h) | The three `configs/refiner/*.yaml` are not edited; every behaviour change is a defaulted dataclass field, so existing hashes cannot move. `check_deployment.py` re-resolves all 200 manifest hashes before and after and fails on any difference. |
| **The ROI changes results more than claimed** (a flipped correspondence cascades through ICP) | The tolerance table on 300 real cached hypotheses runs *before* the sweep; pre-committed stop rule (> 1 % gate flips or any \|Δt\| > 0.01 mm) demotes the ROI to an ablation row. ROI rows live in their own cache directories, so the default path is never at risk. |
| **A "bit-identical" fix is not** (scipy stacks, Open3D source reuse) | Every Class-A fix has an exact-equality (`==`) test on real and synthetic data. A failure moves the fix behind the `roi` flag rather than triggering a version bump. |
| **The ROI's containment guarantee is violated** (the pose leaves the box mid-ICP; a mask outside its bbox) | Per-render sphere-containment check with a full-frame redo and a recorded `roi_fallback` rate; the checker verifies on a sample that every detection mask lies inside its stored bbox. T-LESS's p99 bbox is 63 % of the frame, so the guard *will* fire — its rate is reported, not hidden. |
| **The benchmark measures a loaded machine** (the Δ16 profile already started at load 4.5) | Load-average guard at start, recorded at start *and* end; `OMP_NUM_THREADS=1`; the sweep and the benchmark never overlap; the harness refuses to run while a stage pool is alive. |
| **Host instability under sustained load** (five unclean reboots during Gamma) | `n_workers = 8`; no GPU stage beside CPU stages (the whole sweep runs from cached `segment`/`coarse_pose`); long jobs under `(setsid nohup … &)`; kill by `pgrep -x` / bracket-`ps`, never `pkill -f`. |
| **Open3D raycast corruption** | `RAY_THREADS=1`, `CAST_RETRIES=3` and the full `_cast_checked` validation are untouched; the ROI makes them cheaper, not weaker; the corruption test stays green. |
| **p95 stays above 200 ms** | ADR-0004 promises numbers, not adjectives. The Pareto prices the target in AR and the report says plainly what Python on this host can and cannot do. Pre-commit to publishing whatever comes out, as Epsilon did. |
| **Official AR is the schedule bottleneck** (3 173 s/row, races on the shared clone) | Official for four frontier rows only, serialised, overnight; everything else is core AR and labelled *core* (Alpha: core and official agree within 0.3 pt). |
| **The Confidence models are out of distribution on cheap schedules** | Frozen models (no refit), one `pose_score`-ranked control row at the cheapest schedule, and the caveat stated in the results doc. |
| **The pytest budget** (< 60 s, no data, no GPU) | Real-cache equivalence lives in `tools/check_deployment.py`; only the synthetic twin is in `tests/`. |
| **"No C++" reads as an unfinished milestone** | It is a *measured* decision: the surviving hot spots are Open3D's own C++. ADR-0001 gets a dated status note and the report presents it as the answer to RQ-F, with the before/after table as the evidence. |

---

## Verification (the closure checklist)

1. `uv run pytest -q` green, < 60 s, no GPU/network. `uv run ruff format --check . && uv run ruff
   check . && uv run mypy binposert` clean.
2. `uv run python tools/check_deployment.py --equivalence` — every Class-A column `PASS` at exact
   equality; the ROI tolerance table printed; the 200-manifest hash guard `PASS`.
3. `uv run python tools/check_deployment.py --dataset tless` → **0 failures**, including `A10_exact`
   reproducing A8_k4's per-GT errors and every log free of `Traceback`.
4. `tools/benchmark.py` reruns reproduce p95 within the sampling interval on a quiet host (load
   < 2.0, recorded at start and end).
5. The corrected profile accounts for ~100 % of measured refine time (vs ~50 % in the stored one).
6. Fresh-eyes check: every number in `docs/results_deployment.md` is traceable to a file under
   `outputs/`.
7. `DECISIONS.md` increment status row for `6b Deployment` updated with the date; change-log entry
   written; `MILESTONES.md` status column and status log updated; README's latency sentence updated.
