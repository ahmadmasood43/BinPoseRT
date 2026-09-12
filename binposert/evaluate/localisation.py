"""BOP 6D localisation protocol (ViVo variant is out of scope): per (scene, image, object) the top-n
predictions are matched greedily to the n ground-truth poses; recall is averaged over thresholds.

Ground truth with ``visib_fract < 0.1`` is ignored, as in BOP. Recall is pooled per object over all
images and thresholds, then averaged over objects (bop_toolkit ``eval_calc_scores`` convention).
AR = mean(AR_VSD, AR_MSSD, AR_MSPD).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from binposert.data import BopDataset
from binposert.evaluate.bop_csv import PosePrediction
from binposert.evaluate.metrics import mspd, mssd, vsd
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
    # per-GT rows for stratified analysis:
    # scene, image, object, gt_index, visible_fraction, mssd_mm, mspd_px, success_0.1d
    rows: list[dict[str, float]] = field(default_factory=list)

    @property
    def ar(self) -> float:
        return float(np.mean([self.ar_vsd, self.ar_mssd, self.ar_mspd]))


def evaluate_localisation(
    preds: list[PosePrediction],
    dataset: BopDataset,
    n_model_points: int = 2000,
    with_vsd: bool = True,
) -> LocalisationReport:
    by_key: dict[tuple[int, int, int], list[PosePrediction]] = defaultdict(list)
    for p in preds:
        by_key[(p.scene_id, p.image_id, p.object_id)].append(p)

    renderers: dict[int, MeshRenderer] = {}
    pts_cache: dict[int, npt.NDArray[np.float64]] = {}

    # per object: summed true positives per threshold, and number of valid GT
    tp_mssd: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(len(MSSD_THRESH)))
    tp_mspd: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(len(MSPD_THRESH)))
    tp_vsd: dict[int, npt.NDArray[np.float64]] = defaultdict(
        lambda: np.zeros((len(VSD_TAUS), len(VSD_THRESH)))
    )
    n_valid: dict[int, int] = defaultdict(int)
    rows: list[dict[str, float]] = []
    vsd_available = False

    for scene_id in dataset.scene_ids:
        for image_id in dataset.image_ids(scene_id):
            view, gts = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=with_vsd)
            for object_id in sorted({g.object_id for g in gts}):
                gts_obj = [g for g in gts if g.object_id == object_id]
                valid = [g for g in gts_obj if not (g.visible_fraction < MIN_VISIB)]
                if not valid:
                    continue
                model = dataset.load_model(object_id)
                if object_id not in pts_cache:
                    pts_cache[object_id] = model.sample_points(n_model_points, seed=object_id)
                    renderers[object_id] = MeshRenderer.from_model(model)
                pts = pts_cache[object_id]
                cand = sorted(
                    by_key.get((scene_id, image_id, object_id), []), key=lambda p: -p.score
                )
                cand = cand[: len(gts_obj)]
                width = view.image_size[1]
                thr_mssd = np.asarray(MSSD_THRESH * model.diameter, dtype=np.float64)
                thr_mspd = np.asarray(MSPD_THRESH * (width / 640.0), dtype=np.float64)
                taus = np.asarray(VSD_TAUS * model.diameter, dtype=np.float64)
                use_vsd = with_vsd and view.depth is not None

                e_mssd = np.full((len(cand), len(valid)), np.inf)
                e_mspd = np.full((len(cand), len(valid)), np.inf)
                e_vsd = np.full((len(cand), len(valid), len(taus)), np.inf)
                for i, p in enumerate(cand):
                    for j, g in enumerate(valid):
                        e_mssd[i, j] = mssd(
                            p.T_camera_object, g.T_camera_object, pts, model.symmetry
                        )
                        e_mspd[i, j] = mspd(
                            p.T_camera_object, g.T_camera_object, pts, model.symmetry, view.K
                        )
                        if use_vsd:
                            assert view.depth is not None
                            e_vsd[i, j] = vsd(
                                p.T_camera_object,
                                g.T_camera_object,
                                renderers[object_id],
                                view.depth,
                                view.K,
                                tau_mm=taus,
                            )

                tp_mssd[object_id] += _true_positives(e_mssd, thr_mssd)
                tp_mspd[object_id] += _true_positives(e_mspd, thr_mspd)
                if use_vsd:
                    vsd_available = True
                    for k in range(len(taus)):
                        tp_vsd[object_id][k] += _true_positives(e_vsd[:, :, k], VSD_THRESH)
                n_valid[object_id] += len(valid)
                rows.extend(
                    _gt_rows(scene_id, image_id, object_id, valid, e_mssd, e_mspd, model.diameter)
                )

    per_object: dict[int, dict[str, float]] = {}
    for oid, n in n_valid.items():
        per_object[oid] = {
            "ar_mssd": float(np.mean(tp_mssd[oid] / n)),
            "ar_mspd": float(np.mean(tp_mspd[oid] / n)),
            "ar_vsd": float(np.mean(tp_vsd[oid] / n)) if vsd_available else float("nan"),
            "n_gt": float(n),
        }

    def _mean_over_objects(key: str) -> float:
        vals = [v[key] for v in per_object.values()]
        return float(np.mean(vals)) if vals else float("nan")

    return LocalisationReport(
        ar_mssd=_mean_over_objects("ar_mssd"),
        ar_mspd=_mean_over_objects("ar_mspd"),
        ar_vsd=_mean_over_objects("ar_vsd"),
        n_gt=int(sum(n_valid.values())),
        per_object=per_object,
        rows=rows,
    )


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
    diameter: float,
) -> list[dict[str, float]]:
    rows = []
    for j, g in enumerate(valid):
        best = float(e_mssd[:, j].min()) if e_mssd.shape[0] else float("inf")
        best_p = float(e_mspd[:, j].min()) if e_mspd.shape[0] else float("inf")
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
            }
        )
    return rows
