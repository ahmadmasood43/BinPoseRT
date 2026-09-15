"""Synthetic PoseEstimator: ground truth plus seeded Gaussian noise (tests, demos, fixtures)."""

from __future__ import annotations

import numpy as np

from binposert.data import BopDataset
from binposert.transforms import make_T, rotvec_T
from binposert.types import Detection, PoseHypothesis, QualitySignals, Stage, View


class GroundTruthPerturbedEstimator:
    """One hypothesis per Detection whose ``detection_id`` names a GT index of the View.

    The perturbation is drawn from a generator seeded by ``(seed, scene_id, image_id, gt_index)`` so
    the output is deterministic regardless of iteration order (a cache-stability requirement).
    """

    name = "gt_perturbed"

    def __init__(
        self, dataset: BopDataset, t_sigma_mm: float = 0.0, r_sigma_deg: float = 0.0, seed: int = 0
    ) -> None:
        self.dataset = dataset
        self.t_sigma_mm = t_sigma_mm
        self.r_sigma_deg = r_sigma_deg
        self.seed = seed

    def estimate(self, view: View, detections: list[Detection]) -> list[PoseHypothesis]:
        gts = {g.gt_index: g for g in self.dataset.ground_truth(view.scene_id, view.image_id)}
        out: list[PoseHypothesis] = []
        for d in detections:
            gt = gts.get(d.detection_id)
            if gt is None or gt.object_id != d.object_id:
                continue
            rng = np.random.default_rng([self.seed, view.scene_id, view.image_id, d.detection_id])
            axis = rng.normal(size=3)
            angle = rng.normal(0.0, self.r_sigma_deg)
            dt = rng.normal(0.0, self.t_sigma_mm, size=3)
            # Rotate about the object's own origin and shift in the camera frame, so the error
            # magnitudes are the configured sigmas regardless of the object's distance.
            R_noise = rotvec_T(axis, angle)[:3, :3] if self.r_sigma_deg > 0 else np.eye(3)
            T_gt = gt.T_camera_object
            T = make_T(T_gt[:3, :3] @ R_noise, T_gt[:3, 3] + dt)
            out.append(
                PoseHypothesis(
                    camera_id=view.camera_id,
                    object_id=d.object_id,
                    detection_id=d.detection_id,
                    hypothesis_id=0,
                    T_camera_object=T,
                    stage=Stage.COARSE,
                    signals=QualitySignals(seg_score=d.score, pose_score=1.0),
                    source=self.name,
                )
            )
        return out
