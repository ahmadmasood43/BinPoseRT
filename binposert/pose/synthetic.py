"""A CPU estimator with a *known* error: ground truth plus seeded noise.

Used for pipeline tests on the mini fixture and, in Beta, for controlled initial-error sweeps.
The perturbation of each Detection is a deterministic function of (seed, scene, image, gt_index,
hypothesis) so re-runs are bit-identical.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from binposert.data import BopDataset
from binposert.transforms import make_T
from binposert.types import (
    Detection,
    GroundTruthPose,
    ObjectModel,
    PoseHypothesis,
    QualitySignals,
    Stage,
    View,
)


class PerturbedGroundTruthEstimator:
    name = "synthetic"

    def __init__(
        self,
        dataset: BopDataset,
        sigma_t_mm: float = 5.0,
        sigma_rot_deg: float = 5.0,
        n_hypotheses: int = 1,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.sigma_t_mm = sigma_t_mm
        self.sigma_rot_deg = sigma_rot_deg
        self.n_hypotheses = n_hypotheses
        self.seed = seed

    def estimate(
        self, view: View, detection: Detection, model: ObjectModel
    ) -> list[PoseHypothesis]:
        _, gts = self.dataset.load_view(
            view.scene_id, view.image_id, load_rgb=False, load_depth=False
        )
        # The GT segmenter uses gt_index as detection_id; any other segmenter gets the GT of the
        # same object whose visible mask overlaps the Detection most.
        gt = next((g for g in gts if g.gt_index == detection.detection_id), None)
        if gt is None or gt.object_id != detection.object_id:
            gt = self._best_overlap(
                view, detection, [g for g in gts if g.object_id == detection.object_id]
            )
        if gt is None:
            return []
        out: list[PoseHypothesis] = []
        for h in range(self.n_hypotheses):
            rng = np.random.default_rng(
                _stable_seed(self.seed, view.scene_id, view.image_id, gt.gt_index, h)
            )
            rotvec = rng.normal(0.0, np.radians(self.sigma_rot_deg), 3)
            dt = rng.normal(0.0, self.sigma_t_mm, 3)
            dR = Rotation.from_rotvec(rotvec).as_matrix()
            T = gt.T_camera_object @ make_T(dR, [0, 0, 0])
            T[:3, 3] += dt
            err = float(np.linalg.norm(dt))
            out.append(
                PoseHypothesis(
                    camera_id=view.camera_id,
                    object_id=detection.object_id,
                    detection_id=detection.detection_id,
                    hypothesis_id=h,
                    T_camera_object=T,
                    stage=Stage.COARSE,
                    signals=QualitySignals(
                        seg_score=detection.score,
                        pose_score=float(np.exp(-err / max(self.sigma_t_mm, 1e-9))),
                        visible_fraction=gt.visible_fraction,
                    ),
                    source=self.name,
                )
            )
        return out

    def _best_overlap(
        self, view: View, detection: Detection, candidates: list[GroundTruthPose]
    ) -> GroundTruthPose | None:
        best: GroundTruthPose | None = None
        best_iou = 0.0
        for g in candidates:
            m = self.dataset.gt_mask(view.scene_id, view.image_id, g.gt_index, visible_only=True)
            inter = np.logical_and(m, detection.mask).sum()
            union = np.logical_or(m, detection.mask).sum()
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = g, iou
        return best


def _stable_seed(*parts: int) -> int:
    h = hashlib.sha256(",".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")
