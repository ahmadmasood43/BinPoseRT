# Milestone Gamma — decision log

Every decision taken while executing Milestone 4 (multi-view association and fusion, started
2026-09-16 straight after Beta), in the order it was taken, with the evidence that forced it and the
alternative not chosen. [`DECISIONS.md`](DECISIONS.md) holds the standing decisions (D9, D10),
[`MILESTONES.md`](MILESTONES.md) the plan, [`results_gamma_tless.md`](results_gamma_tless.md) the
numbers. Numbers are T-LESS BOP19 unless stated; *core* = `binposert.evaluate`, otherwise official
`bop_toolkit`.

| # | Decision | Status |
|---|---|---|
| G1 | Start Gamma on 2026-09-16 on the GPU machine, from Beta's caches; XYZ-IBD download first | applied |
| G2 | Download only `base + models + val + test_all` of XYZ-IBD (10.7 GB), not `train_pbr` (72 GB) | applied |
| G3 | View groups: strided over a scene's target images; every image in at most one group; `n_views = 1` is the single-view row | applied |
| G4 | Association: visit Views in order, Hungarian per View against each track's highest-weight member, gate 0.5 d / 45° (symmetry-aware); unmatched hypotheses start tracks | applied (D10) |
| G5 | Fusion weight = seg_score · icp_fitness · depth_coverage · visible_fraction (NaN factors skipped); Refinement-rejected members stay in with ×0.2 | applied (D10) |
| G6 | Evaluate a FusedPose by projecting it into every View of its group (`project_to: all`); BOP protocol unchanged, GT restricted to the group images; official scores via a per-row `targets_subset.json` | applied |
| G7 | Rows per curve point: `none` (single view, same images) / `best` / `mean` / `mean + joint ICP` | applied |
| G8 | Joint ICP reuses `refine/` unchanged on world-frame union clouds; gated with Beta's tuned gate (α 0.6 d, β 90°, fitness ≥ 0.3, silhouette as signal) | applied (D10) |
| G9 | Extrinsic sweep on the 4-view mean row, core evaluator without VSD | applied |
| G10 | Association correctness = purity / completeness against `gt_index`, which BOP keeps consistent across the Views of a static scene | applied |
| G11 | XYZ-IBD is read through a per-camera symlink view (`tools/bop_camera_view.py`); camera `xyz` (the BOP-25 multi-view targets), grayscale | applied |
| G12 | Depth↔RGB offset estimated GT-free from depth-edge / RGB-edge alignment on val scenes; applied as `dataset.depth_shift_px`, part of every depth-reading stage's hash | applied (B20 closed) |
| G13 | Method names in BOP CSVs may not contain underscores (bop_toolkit splits on them); `tools/bop_eval.sh` hyphenates | applied |
| G14 | First A7 run rejected 1020 tracks as `no_depth`: a View cache keyed on image id only; fixed, row re-run, buggy outputs archived | applied |
| G15 | Open3D `cast_rays` still corrupts occasionally under a load average of ~20: every cast is validated and re-cast up to 3 times | applied |
| G17 | Joint multi-view ICP (D10 step 4) is *not* the default: it worsens multi-member tracks on T-LESS; A7 stays as the ablation row | applied (D10 revised) |
| G18 | CNOS on 1440 × 1080: proposal processing chunked (CUDA OOM at 100 proposals), `expandable_segments`, npz arrays read once per image (70 s of re-decompression per image); dataset `name` from the config labels outputs (`xyzibd`, not the camera-view dir) | applied |
| G19 | XYZ-IBD: no depth↔image shift (0 ± 0.5 px on 20 val images); FoundPose's vendored bop_toolkit table extended with the dataset (patch #2) | applied |
| G20 | Scene pools use a `ProcessPoolExecutor` that resubmits the jobs of a worker that died; `multiprocessing.Pool` hung forever when the raycast corruption killed a worker | applied |
| G21 | The host crashed five times during Gamma (44 days uptime before it); evidence points at hardware exposed by sustained full load — default stage parallelism halved, memtest / NVMe SMART / swap recommended | applied (process) |
| G16 | Long jobs are started with `setsid nohup` (a machine reboot / session end killed the sweep and a 6.8 GB download); `curl -C -` resumes partial archives | applied (process) |

---

## G1 — When and where

**Context.** The plan put Gamma in weeks 7–9 (from 2026-10-26) on the laptop; Beta closed on
2026-09-16 with all its caches on the GPU machine and the ~1 MB/s link still unsynced.

**Decision.** Start immediately, on the GPU machine, from the cached CNOS + FoundPose + tuned
point-to-plane refine stages (A2). The XYZ-IBD download is the first action of the milestone (D1's
"GPU work first" rule): it runs in the background while the geometry is written and tested.

## G2 — What to download

**Context.** D5 says "100+ GB". The Hugging Face mirror lists `train_pbr` at 72 GB and everything else
at 10.7 GB (`val` 7.7, `test_all` 2.9, base + models 0.005).

**Decision.** No PBR training images: nothing in Gamma or Delta trains on renders (D11 fits logistic
models on real val signals). At ~1.1 MB/s the useful part takes ~3 h instead of ~30.

## G3 — View groups

**Context.** T-LESS is the multi-view *debug* set (D5): 50 target images per scene, adjacent image ids
being neighbouring viewpoints on the capture hemisphere. A group must look like a calibrated
multi-camera rig, i.e. well-separated viewpoints, and the AR-vs-views curve needs many groups per
scene for its error bars.

**Decision.** With `n` candidate images and `k` views, group `g` is the images at indices
`g, g + n//k, g + 2·n//k, …` — `n//k` disjoint groups with maximally spread viewpoints, every image in at
most one group (48 of 50 used for k = 3, 4). `n_views = 1` gives 50 one-image groups per scene and
reproduces the single-view row exactly (k = 1 `none` = A2, core AR 47.55 on the same 1000 images).
For XYZ-IBD the candidate list is the BOP-25 multi-view targets file (five `xyz` views per scene).

**Not chosen.** Random groups (not reproducible across rows without extra bookkeeping); one group per
scene (no error bars); consecutive image ids (near-identical viewpoints, no multi-view information).

## G4 — Association

**Context.** D10 asks for geometric gating plus Hungarian assignment, one-to-one per View, unassigned
allowed; repeated objects (XYZ-IBD has 26–33 copies of one part per image) must not be merged.

**Decision.** Views are visited in group order. Each existing ObjectTrack is represented by its
highest-weight member in the world frame; the View's hypotheses of the same `object_id` are assigned
one-to-one by `scipy.optimize.linear_sum_assignment` on cost = translation (mm) + 0.5 × symmetry-aware
rotation (deg), restricted to pairs within 0.5 d and 45°. Anything a track cannot take starts a new
track. Nothing is discarded at this stage: a track with one member is a single-view object.

**Evidence.** On T-LESS 2-view groups from A2's refined hypotheses: 5452 hypotheses → 4274 tracks,
1178 with two members; against GT instances (G10) **track purity 100 %** (no track joins two
instances) and **completeness 95.2 %** — the 127 split instances are members whose pose was wrong
enough to fail the gate, which is the right outcome before they are averaged.

**Not chosen.** Global assignment over all Views at once (a k-partite matching; not needed with
calibrated extrinsics — the greedy view-by-view Hungarian left no impure track); a running mean as the
track representative (the highest-weight member is what D10 aligns to anyway).

## G5 — Fusion weights

**Decision.** The hand-set product of D10 with exponents 1 on `seg_score`, `icp_fitness`,
`depth_coverage`, `visible_fraction`; a NaN signal drops out of the product (single-view rows without
refinement still fuse); a member the Refinement rejected keeps its coarse pose with the weight × 0.2,
floored at 1e-3, so a track whose members were all rejected still yields a pose. Delta replaces the
product with Model H's probability behind the same interface.

## G6 — How a FusedPose is scored

**Context.** The BOP protocol scores `T_camera_object` per image; a FusedPose is `T_world_object`.

**Decision.** The `fuse` stage writes, next to `fused.parquet`, a `hypotheses.parquet` with the fused
pose projected into every View of its group (`T_camera_object = T_camera_world · T_world_object`), so the
unchanged `evaluate` stage scores it; GT is restricted to the group images (the `evaluate` stage takes
an image set; `n_images` is reported). The official number uses the same image set through a
`targets_subset.json` written by the evaluate stage and passed to bop_toolkit (`BOP_TARGETS`).
`project_to: members` (only Views with a Detection) is kept as an option.

**Why `all`.** The point of a world-frame object is that it exists in every View; predicting it where the
detector missed it is the benefit, predicting a wrong one where it was not seen is the cost, and the
curve should show both.

## G7 — Rows

`none` (each View keeps its own refined hypothesis; scored on exactly the group images — the fair
single-view reference), `best` (highest-weight member projected everywhere: what selecting a view
buys before any averaging), `mean` (D10 steps 2–3), `mean + joint ICP` (A7, D10 step 4). k = 1 has only
`none`.

## G8 — Joint ICP

**Decision.** `JointRefiner.polish` builds the union of every member View's masked depth cloud in the
world frame (`scene_cloud` per View, normals rotated), the union of the per-View visibility-cropped
model clouds in the object frame (`visible_model_cloud` per View, re-rendered at every coarse-to-fine
step), and calls `register` with `T_world_object` in the role of `T_camera_object` — the registration
only needs a transform from the object frame into the frame the observed points live in. Gate: Beta's
tuned values; the per-View silhouette IoU at the polished pose is averaged into a signal, never a
rejection. Rejections keep the mean.

## G9 — Extrinsic sweep

δt ∈ {0, 1, 2, 5, 10} mm × δθ ∈ {0, 0.1, 0.25, 0.5, 1}°, applied to every View but the first of a
4-view group as a rigid perturbation of `T_world_camera` in the camera frame (random direction, seeded
by scene, group and image), before association and fusion, `mean` row, core evaluator without VSD
(25 rows × 8 min with VSD was not worth it for a heat-map whose shape is the result).

## G10 — Association correctness

**Context.** The exit criterion asks for "association correctness on repeated objects vs GT" without
saying what the GT of an association is.

**Decision.** BOP's `scene_gt` lists the same physical object at the same `gt_index` in every image of a
scene (checked: the world-frame GT poses of scene 1, images 1/17/30 agree to 0.1 mm), so an instance is
`(scene_id, gt_index)`. Every hypothesis is labelled with the nearest same-object GT instance in its
View (translation within 0.5 d). Purity = share of multi-member tracks whose labelled members refer to
one instance; completeness = share of (instance, View) pairs that sit in the instance's largest track;
mixed tracks and split instances are counted and drawn in the gallery.

## G11 — XYZ-IBD

**Context.** The BOP-25 layout has per-camera directories (`rgb_realsense`, `gray_xyz`, `depth_xyz`,
`scene_camera_xyz.json`, …); the multi-view targets list five `xyz`-camera images per scene, and the
`xyz` camera is grayscale (1440 × 1080).

**Decision.** `tools/bop_camera_view.py xyzibd xyz` builds `data/bop/xyzibd_xyz/` as symlinks in the
standard layout (`rgb → gray_xyz`); the loader, the three adapters and bop_toolkit run unchanged and the
dataset is a config entry (D5). Grayscale PNGs are read as three identical channels by every consumer;
CNOS/FoundPose (DINOv2) on grayscale is a measured risk, with `realsense` (RGB, 1280 × 720, same
scenes) as the fallback camera through the same tool.

## G12 — The depth↔RGB offset (closes B20)

**Context.** Beta measured ICP converging +2.2–2.5 mm along camera y from GT with perfect fitness and
refused to subtract a number measured against test GT.

**Decision.** `tools/depth_rgb_offset.py`: depth discontinuities (Sobel on a median-filtered depth map,
> 40 mm) are matched to Canny edges of the RGB image over integer shifts ± 8 px with a robust inlier
objective (mean of `exp(-(d/1.5 px)²)`), refined to sub-pixel with a parabola, restricted to the
CNOS Detection masks (dilated 7 px) so the bin and the table do not vote; images without a clear peak
(score < 0.4 or gain < 0.02 over the unshifted score — the low-elevation views whose depth is too noisy)
are dropped; the median of the rest is the shift. On val scenes {1, 6, 11, 16} (every 5th target image,
8 of 40 with a clear peak): **du = −0.19 px, dv = −3.28 px ≈ −2.4 mm at 775 mm** — the depth map sits
3.3 px too low, the size and direction of Beta's bias, obtained without a pose annotation. Applied through
`dataset.depth_shift_px` (nearest-neighbour translation of the depth map in `BopDataset.load_depth`),
which is part of the refine / fuse / evaluate cache hashes.

**Val check** (A2's refinement re-run on the four val scenes, GT used only to *read* the effect, 852 vs
846 accepted hypotheses): the median translation error of successful, well-fitted refinements goes
from (+0.49, **+2.39**, −0.37) mm to (+0.52, **+0.56**, −0.31) mm; per-hypothesis success (MSSD < 0.1 d)
64.0 → 66.8 %; within 0.05 d 24.6 → 39.0 %; median MSSD of successes 5.03 → 4.08 mm. The shift is
enabled in `configs/dataset/tless.yaml` and every Gamma row (including the k = 1 single-view row, which
is A2 on corrected depth) uses it; the three unshifted rows computed before the decision are kept under
`outputs/runs/noshift/`. On all 20 scenes the k = 1 row (A2 on corrected depth, same 1000 images, same
gate) goes from core AR 47.55 to **50.86** (VSD 41.5 → 48.0, MSSD 50.1 → 51.5, MSPD 51.0 → 53.2;
rejection rate 7.7 %, unchanged) — the largest single change since the depth initialisation, for a
two-number sensor constant.

**Not chosen.** Fitting the offset to the ICP residual against GT (tuning on the benchmark); a 3D
extrinsic correction between depth and RGB sensors (the data is already registered per pixel by the
Primesense driver; what remains is a 2D image offset).

## G13 — bop_toolkit file names

`eval_bop19_pose.py` splits the result file name on `_` to get method and `dataset-split`, so a method
name such as `binposert-A6_k2_mean` broke the official evaluation. `tools/bop_eval.sh` now takes the
dataset-split from the *last* underscore and hyphenates the method (`binposert-A6-k2-mean`); the
scores directory is linked under the CSV's own name too. Dataset names likewise carry no underscore
(`dataset.bop_name` for the XYZ-IBD camera view).

## G14 — The first A7 run was wrong

**Evidence.** A7 k = 2 scored 63.0 (below the mean's 63.9) with 1020 of 4265 tracks rejected as
`no_depth` although the single-view stage saw 22 depth-less hypotheses in all. All 1020 were
single-member tracks with full depth coverage and zero scene points.

**Cause.** `fuse_scene` cached Views by image id only; the projection step of an earlier track loaded a
View *without* depth, and every later track seen in that View polished against `depth = None`. The
mini-fixture test could not see it (every track spans both Views there, so the depth-loaded View was
always cached first).

**Fix.** Cache key `(image_id, with_depth)`; the early rejection reason distinguishes `no_depth`
(no valid depth under the mask) from `depth_window` (nothing within the z window); the pipeline test
asserts every track polished against scene points. The buggy fuse/evaluate outputs were deleted and the
manifest archived under `outputs/runs/noshift/A7_k2_tless_buggy_viewcache`; the row is re-run.

## G15 — Raycast corruption, again

**Evidence.** The A7 k = 2 re-run died in a worker with `IndexError: index 2097293 is out of bounds
for axis 0 with size 540` inside `MeshRenderer.render` — a primitive-id array with bit 21 set, the
Beta signature — while the test suite (its own process pool) ran next to the 15-worker fuse stage
(load average ~20). Beta's `nthreads=1` + one BLAS thread per worker made the failure rare, not
impossible.

**Decision.** `_cast_checked` validates every `cast_rays` answer (array shapes, primitive and geometry
ids within range, finite positive hit distances) and re-casts up to three times; a persistent failure
raises `RenderCorrupted` instead of letting garbage through. Tested by wrapping the scene with a
flaky `cast_rays` (one corrupted answer, then a persistently invalid one). On XYZ-IBD the same
signature (`IndexError: index 2097319 … size 1080`) then surfaced *after* a validated cast, inside
numpy's boolean-mask assignment — the corruption also lands in memory numpy reuses once Embree's
threads are done — so the whole render (cast + post-processing) is retried, not only the cast.

## G16 — Detached jobs and resumable downloads (process)

The GPU machine rebooted mid-milestone; the nohup'd sweep and the 85 %-complete `val` download died
with the session. `tools/run_gamma.sh` is resumable by construction (D12 caches); the download was not
(`urllib.request.urlretrieve`). `tools/resume_xyzibd.sh` continues partial archives with
`curl -C -` and both jobs now start under `setsid nohup … < /dev/null &`, outside the session's process
group. Finding processes to stop: `pgrep -x <name>` or a pattern split with quotes
(`'run_gam'"m"'a.sh'`) — a plain `pgrep -f pattern` inside a `bash -c` command line matches, and
kills, the shell issuing it.

## G17 — The joint polish is dropped from the default path

**Evidence.** A7 (mean + joint ICP) scores below A6 (mean) at every k: 63.1 vs 63.9 (k = 2), 67.8 vs
69.3 (k = 3), with 87–89 % of tracks accepted by the gate. Per track against the labelled GT
instance (world-frame MSSD, 2707 / 1999 labelled tracks at k = 2 / 3):

| members | tracks | success mean → joint | median MSSD mean → joint | worsened > 1 mm | improved > 1 mm |
|---|---|---|---|---|---|
| 1 (k = 2) | 1586 | 75.3 → 75.0 % | 4.72 → 4.74 mm | 9 % | 7 % |
| 2 (k = 2) | 1121 | 95.8 → 92.8 % | 3.40 → 3.98 mm | 37 % | 8 % |
| 1 (k = 3) | 812 | 61.0 → 60.8 % | 5.29 → 5.39 mm | 11 % | 9 % |
| 2 (k = 3) | 646 | 92.3 → 91.0 % | 3.47 → 4.08 mm | 31 % | 11 % |
| 3 (k = 3) | 541 | 98.5 → 94.1 % | 2.95 → 3.70 mm | 40 % | 6 % |

A second ICP pass on a single view is neutral; registering one model against the *union* of several
Views' clouds is worse than averaging the per-view registrations, and the more Views, the worse. The
union cloud carries each View's residual errors (extrinsics estimated per image, the remaining
depth↔RGB misregistration after the 2-D shift, depth noise that differs by elevation) as an
inconsistent surface, and ICP settles on a compromise pulled towards the denser View; the
Lie-algebra mean of per-view poses averages those errors instead. The mean's 3-member tracks are
right 98.5 % of the time — there is little left for a polish to fix.

**Decision.** D10 step 4 is not part of the default multi-view path; `fusion=mean` is A6's and the
project's default, A7 remains the ablation row that documents this. The joint registration's
fitness / RMSE are still available as `multiview_residual_mm` for Delta if a run pays the 8 min per
row; the default signals are the mean's view count and dispersion.

**Not chosen.** Weighting Views inside the joint ICP, or polishing only when the dispersion is large:
both are tuning against the effect rather than its cause; a per-view extrinsic refinement (bundle
adjustment of camera poses from the object correspondences) is the principled fix and is out of
scope (D10 excludes joint render-based optimisation).

## G18 — CNOS at XYZ-IBD resolution

Three things broke or crawled at 1440 × 1080 that were invisible on T-LESS's 720 × 540:

1. `process_rgb_proposals` replicates the full-resolution image once per proposal before cropping
   (N × 3 × H × W float32): 1.9 GB for one image with ~100 proposals → CUDA OOM on the 12 GB card
   at image 162. Patched to chunks of 16 (same tensors, bounded peak; `docker/cnos/patch.sh` #2), and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the segmenter's `env` (not hashed).
2. The adapter read `data["segmentation"][k]` per detection from the per-image `npz`; `NpzFile`
   re-reads and decompresses the member (80 × 1080 × 1440 float32 ≈ 500 MB) on every access —
   ~70 s per image of pure I/O, i.e. most of what the T-LESS "19 s per image" was too. Arrays are now
   read once per image; the cached-image pass runs at 1.6 s per image.
3. `BopDataset.name` was the root directory's name, so the camera view produced
   `outputs/xyzibd_xyz/…` and `outputs/runs/<row>_xyzibd_xyz/` while every tool addressed the dataset
   as `xyzibd`. The config `name` now labels outputs and run directories (T-LESS and the fixture are
   unchanged: their config names equal their directory names).

Also fixed on the way: CNOS's template renderer applied a millimetre centroid as a metre offset
(blank templates for 13 of 15 objects) and reports `category_id = template index + 1`, which only
equals the object id when the ids are contiguous (XYZ-IBD's are not) — see G11's tool and
`docker/cnos/patch.sh` #1; the adapter maps category ids through the sorted template list.

## G19 — XYZ-IBD sensor offset and FoundPose onboarding

`tools/depth_rgb_offset.py --dataset xyzibd` on 20 val images (scenes 0/5/10/15, CNOS masks): every
image's best shift lies within ±0.6 px of zero and the robust objective is flat around it (no image
passes the clear-peak filter because there is nothing to gain), i.e. the `xyz` camera's gray image
and depth map are co-registered — one structured-light optical path, unlike T-LESS's RGB + depth pair.
`depth_shift_px` stays `null` for XYZ-IBD; the estimator's "no shift" answer is as much a result as
T-LESS's −3.3 px.

FoundPose's template synthesis reads dataset facts from a hard-coded table in its vendored
bop_toolkit (`dataset_params.py`) and raised `KeyError: 'xyzibd_xyz'`; `docker/foundpose/patch.sh` #2
adds `xyzibd` / `xyzibd_xyz` (object ids with gaps, symmetric ids from `models_info.json`, 1440 × 1080,
val object depth 604–828 mm for the view spheres, full azimuth / elevation range). Its template synthesis then refused object 15 — a 296 mm bar whose
model origin sits at one end (269 mm reach), which spans 677 px from the principal point at the sampled
716 mm radius and touches the 1080-px image border ("The model does not fit the viewport"). Patch #3
scales the view-sphere radius per object so the model's reach fits 90 % of the shorter half-extent
(templates are cropped and resized to 420 px anyway; small objects keep the sampled radius). That
still left one in-plane-rotated view of object 15 on the border, so patch #4 re-renders any view whose
mask touches the border from 25 % further away (up to five times) instead of aborting the onboarding.

## G20 — A dying worker must not hang a stage

**Evidence.** The XYZ-IBD k = 4 single-view evaluate sat for 1 h 49 min with every worker asleep in
`futex_wait`, one of them a *replacement* spawned mid-run: `multiprocessing.Pool` replaces a worker
that dies from a signal but the task it held is lost and `starmap` waits for it forever. The killer
is the same Open3D raycast corruption (G15) — this time as a crash rather than an exception.

**Decision.** `binposert.pipeline.pool.map_scenes` (used by refine, fuse, evaluate and the refinement
scoring) runs scene jobs in a `concurrent.futures.ProcessPoolExecutor` of single-threaded spawned
workers; a dead worker raises `BrokenProcessPool`, the jobs that did not finish are resubmitted to a
fresh pool (three attempts), and only a job that keeps dying fails the stage. Tested with a job that
SIGKILLs its own worker once and one that always does.

## G21 — The host's crashes

**Evidence.** `last -x`: one clean shutdown after 44 days of uptime (Jul 28 → Sep 11), then, from the
day Alpha started on this machine, eight boots in four days of which five have no shutdown record
(Sep 14 ×2, Sep 15, Sep 16 15:30 — during the first Gamma sweep —, Sep 17 18:05). The journal of the
crashed boots is itself corrupted ("system.journal corrupted or uncleanly shut down"); the root
filesystem developed a bad inode that had to be deleted by hand (a file in the venv's `pygments`
package: every `import pytest` hung unkillably in `d_alloc_parallel` until then). There is no swap.

**What this project did that a dev box does not usually see:** all 16 threads at 100 % for hours
(15 single-threaded workers) with the GPU at 100 % beside them, a load average of 112 for an hour when
the executor's workers were not thread-pinned, and ~30 GB of writes (templates, masks, caches).
`k10temp` reads 66 °C at a load of 4; nothing was measured under full load.

**Reading.** Python cannot crash a kernel or corrupt an inode. The single-bit corruptions this project
kept seeing in user memory — a valid array index with bit 17, 18 or 21 set (Beta's raycast finding,
G15, G20) — are the fingerprint of marginal RAM or memory timing rather than of an Embree overrun,
which trashes ranges, not one high bit. Sustained full load exposes marginal hardware (heat, power,
memory timing); the project is the trigger, not the cause. Recommended, in order: memtest86+ overnight;
`nvme smart-log` / `smartctl -a` on both NVMe devices (media errors, unsafe-shutdown count); `fsck`
of both filesystems from a live boot; disable any XMP/EXPO memory profile; add swap so memory pressure
degrades instead of freezing.

**Decision.** `n_workers: 0` now means half the logical cores (was all but one); GPU stages are not
run beside a full CPU stage (`tools/run_gamma_xyzibd.sh` sequences them). The raycast validation and
retry and the dying-worker recovery stay in — on healthy hardware they cost nothing.
