"""Core domain types (see CONTEXT.md for the vocabulary these implement).

All rigid transforms are 4x4 float64 numpy arrays named ``T_a_b``: they map points expressed
in frame ``b`` into frame ``a``. Units are millimetres for translation, matching BOP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Mat4 = npt.NDArray[np.float64]
Mat3 = npt.NDArray[np.float64]


# ----------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class SymmetryGroup:
    """Finite list of object-frame transforms under which the model is indistinguishable.

    ``transforms[0]`` is always the identity. Continuous axes are already sampled (see ADR-0003).
    """

    transforms: tuple[Mat4, ...]

    def __post_init__(self) -> None:
        if len(self.transforms) == 0:
            raise ValueError("SymmetryGroup must contain at least the identity")
        if not np.allclose(self.transforms[0], np.eye(4)):
            raise ValueError("SymmetryGroup.transforms[0] must be the identity")

    def __len__(self) -> int:
        return len(self.transforms)

    @property
    def is_trivial(self) -> bool:
        return len(self.transforms) == 1

    @staticmethod
    def trivial() -> SymmetryGroup:
        return SymmetryGroup((np.eye(4),))


@dataclass(frozen=True)
class ObjectModel:
    """A CAD model with a stable id, its diameter (mm), and its SymmetryGroup."""

    object_id: int
    vertices: npt.NDArray[np.float64]  # (V, 3) mm, object frame
    faces: npt.NDArray[np.int64]  # (F, 3)
    diameter: float
    symmetry: SymmetryGroup = field(default_factory=SymmetryGroup.trivial)
    mesh_path: Path | None = None

    def sample_points(self, n: int, seed: int = 0) -> npt.NDArray[np.float64]:
        """Uniformly sample ``n`` surface points (area-weighted). Deterministic for a given seed."""
        rng = np.random.default_rng(seed)
        v = self.vertices
        tri = v[self.faces]  # (F,3,3)
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        probs = areas / areas.sum()
        idx = rng.choice(len(self.faces), size=n, p=probs)
        r1, r2 = rng.random(n), rng.random(n)
        s1 = np.sqrt(r1)
        a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
        return (1 - s1)[:, None] * a + (s1 * (1 - r2))[:, None] * b + (s1 * r2)[:, None] * c


@dataclass(frozen=True)
class View:
    """One camera capture of a Scene."""

    camera_id: str
    K: Mat3
    T_world_camera: Mat4
    rgb: npt.NDArray[np.uint8] | None  # (H, W, 3)
    depth: npt.NDArray[np.float64] | None  # (H, W) mm, 0 = missing
    image_size: tuple[int, int]  # (H, W)
    # BOP bookkeeping so results can be written back in BOP format
    scene_id: int = -1
    image_id: int = -1

    @property
    def T_camera_world(self) -> Mat4:
        return np.linalg.inv(self.T_world_camera)


@dataclass(frozen=True)
class GroundTruthPose:
    """One annotated object pose in one View (BOP scene_gt + scene_gt_info)."""

    object_id: int
    T_camera_object: Mat4
    visible_fraction: float
    gt_index: int  # index within the BOP scene_gt list for this image


@dataclass(frozen=True)
class Scene:
    """One bin state observed by one or more Views."""

    dataset: str
    scene_id: int
    views: tuple[View, ...]
    ground_truth: dict[str, tuple[GroundTruthPose, ...]] = field(
        default_factory=dict
    )  # by camera_id

    def view(self, camera_id: str) -> View:
        for v in self.views:
            if v.camera_id == camera_id:
                return v
        raise KeyError(camera_id)

    @property
    def is_multi_view(self) -> bool:
        return len(self.views) > 1


# ----------------------------------------------------------------------------- per-view results


@dataclass(frozen=True)
class Detection:
    """One mask for one object_id in one View. Carries no pose."""

    camera_id: str
    object_id: int
    mask: npt.NDArray[np.bool_]  # (H, W)
    score: float
    detection_id: int  # unique within the View

    @property
    def bbox_xywh(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )


class Stage(str, Enum):
    COARSE = "coarse"
    REFINED = "refined"


@dataclass
class QualitySignals:
    """Measurable evidence about a pose. Fixed, versioned schema (D11). NaN = not available.

    Bump ``SCHEMA_VERSION`` whenever a field is added, removed or its meaning changes.
    """

    SCHEMA_VERSION = 1

    seg_score: float = math.nan
    pose_score: float = math.nan
    reproj_error_px: float = math.nan
    n_inliers: float = math.nan
    icp_fitness: float = math.nan
    icp_rmse_mm: float = math.nan
    depth_coverage: float = math.nan
    visible_fraction: float = math.nan
    silhouette_iou: float = math.nan
    displacement_mm: float = math.nan
    displacement_deg: float = math.nan
    multiview_residual_mm: float = math.nan
    dispersion_mm: float = math.nan
    dispersion_deg: float = math.nan
    n_views: float = math.nan

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_row(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> QualitySignals:
        names = set(cls.field_names())
        return cls(**{k: float(v) for k, v in row.items() if k in names})


@dataclass(frozen=True)
class PoseHypothesis:
    """One candidate T_camera_object for one Detection."""

    camera_id: str
    object_id: int
    detection_id: int
    hypothesis_id: int  # unique within (camera_id, detection_id)
    T_camera_object: Mat4
    stage: Stage
    signals: QualitySignals = field(default_factory=QualitySignals)
    source: str = ""  # estimator / refiner name
    rejection_reason: str | None = None  # set by a Refinement that declined to change the pose


# ----------------------------------------------------------------------------- cross-view results


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_VIEW = "request_view"


@dataclass
class ObjectTrack:
    """One physical object in the Scene, owning the hypotheses associated to it across Views."""

    track_id: int
    object_id: int
    hypotheses: list[PoseHypothesis] = field(default_factory=list)

    @property
    def camera_ids(self) -> list[str]:
        return sorted({h.camera_id for h in self.hypotheses})


@dataclass(frozen=True)
class FusedPose:
    """Final world-frame pose of an ObjectTrack with its Confidence and Verdict."""

    track_id: int
    object_id: int
    T_world_object: Mat4
    confidence: float
    verdict: Verdict
    signals: QualitySignals = field(default_factory=QualitySignals)
