"""BOP 6D localisation protocol (ViVo variant is out of scope): per (scene, image, object) the top-n
predictions are matched greedily to the n ground-truth poses; recall is averaged over thresholds.

Valid ground truth follows bop_toolkit ``eval_calc_scores``: with a targets file (BOP19), the
``inst_count`` most visible instances of each target object are valid; without one, instances with
``visib_fract >= 0.1``. As in ``eval_bop19_pose``, recall at each threshold is pooled over *all*
valid GT instances (not averaged per object) and AR_x is the mean over thresholds;
AR = mean(AR_VSD, AR_MSSD, AR_MSPD). Per-object recalls are reported alongside.
``n_model_points <= 0`` uses every model vertex, as BOP does with ``models_eval``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from binposert.data import BopDataset
from binposert.evaluate.bop_csv import PosePrediction
from binposert.evaluate.metrics import _ray_length_factor, mspd, mssd, vsd_from_distances
from binposert.render import MeshRenderer
from binposert.types import GroundTruthPose

MIN_VISIB = 0.1
MSSD_THRESH = np.arange(0.05, 0.51, 0.05)  # x diameter
MSPD_THRESH = np.arange(5, 51, 5)  # x (width / 640) px
VSD_TAUS = np.arange(0.05, 0.51, 0.05)  # x diameter
VSD_THRESH: npt.NDArray[np.float64] = np.arange(0.05, 0.51, 0.05)


@dataclass
class LocalisationReport:
    ar_mssd: float
    ar_mspd: float
    ar_vsd: float
    n_gt: int
    per_object: dict[int, dict[str, float]] = field(default_factory=dict)
    # per-GT rows for stratified analysis: scene_id, image_id, object_id, gt_index,
    # visible_fraction, mssd_mm, mspd_px, success_0.1d, and the per-GT recall fractions
    # ar_mssd / ar_mspd / ar_vsd (share of thresholds passed by the closest prediction; NaN if
    # the metric was not computed). Pooled means of these are the stratified AR numbers.
    rows: list[dict[str, float]] = field(default_factory=list)

    @property
    def ar(self) -> float:
        return float(np.mean([self.ar_vsd, self.ar_mssd, self.ar_mspd]))


@dataclass
class _SceneResult:
    tp_mssd: dict[int, npt.NDArray[np.float64]]
    tp_mspd: dict[int, npt.NDArray[np.float64]]
    tp_vsd: dict[int, npt.NDArray[np.float64]]
    n_valid: dict[int, int]
    rows: list[dict[str, float]]
    vsd_available: bool


def _evaluate_scene(
    scene_id: int,
    by_key: dict[tuple[int, int, int], list[PosePrediction]],
    dataset: BopDataset,
    n_model_points: int,
    with_vsd: bool,
    images: set[tuple[int, int]] | None = None,
) -> _SceneResult:
    """All images of one scene (or those in ``images``); renderers and model points are built per
    worker process."""
    renderers: dict[int, MeshRenderer] = {}
    pts_cache: dict[int, npt.NDArray[np.float64]] = {}
    tp_mssd: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(len(MSSD_THRESH)))
    tp_mspd: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(len(MSPD_THRESH)))
    tp_vsd: dict[int, npt.NDArray[np.float64]] = defaultdict(
        lambda: np.zeros((len(VSD_TAUS), len(VSD_THRESH)))
    )
    n_valid: dict[int, int] = defaultdict(int)
    rows: list[dict[str, float]] = []
    vsd_available = False

    for image_id in dataset.image_ids(scene_id):
        if images is not None and (scene_id, image_id) not in images:
            continue
        view, gts = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=with_vsd)
        object_ids = dataset.target_object_ids(scene_id, image_id)
        if object_ids is None:
            object_ids = sorted({g.object_id for g in gts})
        for object_id in object_ids:
            gts_obj = [g for g in gts if g.object_id == object_id]
            valid = valid_ground_truth(dataset, scene_id, image_id, object_id, gts_obj)
            if not valid:
                continue
            model = dataset.load_model(object_id)
            if object_id not in pts_cache:
                pts_cache[object_id] = (
                    model.sample_points(n_model_points, seed=object_id)
                    if n_model_points > 0
                    else model.vertices
                )
                renderers[object_id] = MeshRenderer.from_model(model)
            pts = pts_cache[object_id]
            # n_top = number of valid GT (bop_toolkit: the target's inst_count)
            cand = sorted(by_key.get((scene_id, image_id, object_id), []), key=lambda p: -p.score)
            cand = cand[: len(valid)]
            width = view.image_size[1]
            thr_mssd = np.asarray(MSSD_THRESH * model.diameter, dtype=np.float64)
            thr_mspd = np.asarray(MSPD_THRESH * (width / 640.0), dtype=np.float64)
            taus = np.asarray(VSD_TAUS * model.diameter, dtype=np.float64)
            use_vsd = with_vsd and view.depth is not None

            e_mssd = np.full((len(cand), len(valid)), np.inf)
            e_mspd = np.full((len(cand), len(valid)), np.inf)
            e_vsd = np.full((len(cand), len(valid), len(taus)), np.inf)
            dist_est: list[npt.NDArray[np.float64]] = []
            dist_gt: list[npt.NDArray[np.float64]] = []
            dist_test: npt.NDArray[np.float64] | None = None
            if use_vsd:
                # render every pose once; VSD of a pair works on the two distance images
                assert view.depth is not None
                ray = _ray_length_factor(view.K, view.image_size)
                dist_test = view.depth * ray
                rend = renderers[object_id]
                dist_est = [
                    rend.render(p.T_camera_object, view.K, view.image_size).depth * ray
                    for p in cand
                ]
                dist_gt = [
                    rend.render(g.T_camera_object, view.K, view.image_size).depth * ray
                    for g in valid
                ]
            for i, p in enumerate(cand):
                for j, g in enumerate(valid):
                    e_mssd[i, j] = mssd(p.T_camera_object, g.T_camera_object, pts, model.symmetry)
                    e_mspd[i, j] = mspd(
                        p.T_camera_object, g.T_camera_object, pts, model.symmetry, view.K
                    )
                    if use_vsd:
                        assert dist_test is not None
                        e_vsd[i, j] = vsd_from_distances(
                            dist_est[i], dist_gt[j], dist_test, tau_mm=taus
                        )

            tp_mssd[object_id] += _true_positives(e_mssd, thr_mssd)
            tp_mspd[object_id] += _true_positives(e_mspd, thr_mspd)
            if use_vsd:
                vsd_available = True
                for k in range(len(taus)):
                    tp_vsd[object_id][k] += _true_positives(e_vsd[:, :, k], VSD_THRESH)
            n_valid[object_id] += len(valid)
            rows.extend(
                _gt_rows(
                    scene_id,
                    image_id,
                    object_id,
                    valid,
                    e_mssd,
                    e_mspd,
                    e_vsd if use_vsd else None,
                    thr_mssd,
                    thr_mspd,
                    model.diameter,
                )
            )
    return _SceneResult(
        dict(tp_mssd), dict(tp_mspd), dict(tp_vsd), dict(n_valid), rows, vsd_available
    )


def evaluate_localisation(
    preds: list[PosePrediction],
    dataset: BopDataset,
    n_model_points: int = 2000,
    with_vsd: bool = True,
    n_workers: int = 1,
    images: set[tuple[int, int]] | None = None,
) -> LocalisationReport:
    """``n_workers > 1`` evaluates scenes in parallel (spawned) processes; results are identical.
    ``images`` restricts the ground truth to the listed ``(scene_id, image_id)`` pairs."""
    by_key: dict[tuple[int, int, int], list[PosePrediction]] = defaultdict(list)
    for p in preds:
        by_key[(p.scene_id, p.image_id, p.object_id)].append(p)

    scene_ids = dataset.scene_ids
    if images is not None:
        wanted = {s for s, _ in images}
        scene_ids = [s for s in scene_ids if s in wanted]
    jobs = [(sid, by_key, dataset, n_model_points, with_vsd, images) for sid in scene_ids]
    if n_workers > 1 and len(scene_ids) > 1:
        from binposert.pipeline.pool import map_scenes

        results = map_scenes(_evaluate_scene, jobs, n_workers)
    else:
        results = [_evaluate_scene(*job) for job in jobs]

    tp_mssd: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(len(MSSD_THRESH)))
    tp_mspd: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(len(MSPD_THRESH)))
    tp_vsd: dict[int, npt.NDArray[np.float64]] = defaultdict(
        lambda: np.zeros((len(VSD_TAUS), len(VSD_THRESH)))
    )
    n_valid: dict[int, int] = defaultdict(int)
    rows: list[dict[str, float]] = []
    vsd_available = False
    for r in results:
        for oid, v in r.tp_mssd.items():
            tp_mssd[oid] += v
        for oid, v in r.tp_mspd.items():
            tp_mspd[oid] += v
        for oid, v in r.tp_vsd.items():
            tp_vsd[oid] += v
        for oid, n in r.n_valid.items():
            n_valid[oid] += n
        rows.extend(r.rows)
        vsd_available |= r.vsd_available

    per_object: dict[int, dict[str, float]] = {}
    for oid, n in n_valid.items():
        per_object[oid] = {
            "ar_mssd": float(np.mean(tp_mssd[oid] / n)),
            "ar_mspd": float(np.mean(tp_mspd[oid] / n)),
            "ar_vsd": float(np.mean(tp_vsd[oid] / n)) if vsd_available else float("nan"),
            "n_gt": float(n),
        }

    n_total = int(sum(n_valid.values()))

    def _pooled(tp: dict[int, npt.NDArray[np.float64]]) -> float:
        if n_total == 0:
            return float("nan")
        return float(np.mean(sum(tp.values()) / n_total))

    return LocalisationReport(
        ar_mssd=_pooled(tp_mssd),
        ar_mspd=_pooled(tp_mspd),
        ar_vsd=_pooled(tp_vsd) if vsd_available else float("nan"),
        n_gt=n_total,
        per_object=per_object,
        rows=rows,
    )


def valid_ground_truth(
    dataset: BopDataset,
    scene_id: int,
    image_id: int,
    object_id: int,
    gts_obj: list[GroundTruthPose],
) -> list[GroundTruthPose]:
    if dataset.targets is not None:
        k = dataset.targets.get((scene_id, image_id), {}).get(object_id, 0)
        ranked = sorted(gts_obj, key=lambda g: g.visible_fraction, reverse=True)
        return sorted(ranked[:k], key=lambda g: g.gt_index)
    return [g for g in gts_obj if not (g.visible_fraction < MIN_VISIB)]


def _true_positives(
    errors: npt.NDArray[np.float64], thresholds: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Greedy BOP matching per threshold: number of matched GT for each threshold."""
    n_pred, n_gt = errors.shape
    out = np.zeros(len(thresholds))
    for k, thr in enumerate(thresholds):
        matched_gt: set[int] = set()
        for i in range(n_pred):  # candidates already sorted by descending score
            best_j, best_e = -1, np.inf
            for j in range(n_gt):
                if j in matched_gt:
                    continue
                if errors[i, j] < best_e:
                    best_j, best_e = j, errors[i, j]
            if best_j >= 0 and best_e < thr:
                matched_gt.add(best_j)
        out[k] = len(matched_gt)
    return out


def _gt_rows(
    scene_id: int,
    image_id: int,
    object_id: int,
    valid: list[GroundTruthPose],
    e_mssd: npt.NDArray[np.float64],
    e_mspd: npt.NDArray[np.float64],
    e_vsd: npt.NDArray[np.float64] | None,
    thr_mssd: npt.NDArray[np.float64],
    thr_mspd: npt.NDArray[np.float64],
    diameter: float,
) -> list[dict[str, float]]:
    rows = []
    n_pred = e_mssd.shape[0]
    for j, g in enumerate(valid):
        best = float(e_mssd[:, j].min()) if n_pred else float("inf")
        best_p = float(e_mspd[:, j].min()) if n_pred else float("inf")
        ar_vsd = float("nan")
        if e_vsd is not None:
            if n_pred:
                best_v = e_vsd[:, j, :].min(axis=0)  # per tau
                ar_vsd = float(np.mean(best_v[:, None] < VSD_THRESH[None, :]))
            else:
                ar_vsd = 0.0
        rows.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "object_id": object_id,
                "gt_index": g.gt_index,
                "visible_fraction": g.visible_fraction,
                "mssd_mm": best,
                "mspd_px": best_p,
                "success_0.1d": float(best < 0.1 * diameter),
                "ar_mssd": float(np.mean(best < thr_mssd)),
                "ar_mspd": float(np.mean(best_p < thr_mspd)),
                "ar_vsd": ar_vsd,
            }
        )
    return rows
