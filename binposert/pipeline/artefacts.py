"""On-disk artefact formats shared by every stage and by the estimator adapters (D12).

This module deliberately imports only numpy, pandas, pyarrow and imageio so that the adapters, which
run inside the estimator Docker images with the repository on ``PYTHONPATH``, can write artefacts
without installing the core's other dependencies.

Layout of one stage output directory::

    <stage_dir>/
        _SUCCESS                      written last; the directory is valid only if present
        stage.json                    stage name, version, config, upstream hashes (provenance)
        detections.parquet            Detections (segment stage)
        masks/<scene:06d>/<image:06d>_<detection:06d>.png
        hypotheses.parquet            PoseHypotheses (coarse_pose / refine stages)

4x4 transforms are stored row-major as 16 float columns ``T_00 .. T_33``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pandas as pd

from binposert.types import Detection, PoseHypothesis, QualitySignals, Stage

SUCCESS_MARKER = "_SUCCESS"
STAGE_INFO = "stage.json"
DETECTIONS_FILE = "detections.parquet"
HYPOTHESES_FILE = "hypotheses.parquet"
MASK_DIR = "masks"

ARTEFACT_SCHEMA_VERSION = 1

TRANSFORM_COLUMNS = [f"T_{i}{j}" for i in range(4) for j in range(4)]

DETECTION_COLUMNS = [
    "scene_id",
    "image_id",
    "camera_id",
    "object_id",
    "detection_id",
    "score",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "mask_path",
    "time_s",
]

HYPOTHESIS_COLUMNS = [
    "scene_id",
    "image_id",
    "camera_id",
    "object_id",
    "detection_id",
    "hypothesis_id",
    "stage",
    "source",
    "rejection_reason",
    *TRANSFORM_COLUMNS,
    *QualitySignals.field_names(),
    "time_s",
]


def camera_id_for(image_id: int) -> str:
    """Duplicate of :func:`binposert.data.camera_id_for`, kept here to avoid importing cv2."""
    return f"{int(image_id):06d}"


def mask_relpath(scene_id: int, image_id: int, detection_id: int) -> str:
    return f"{MASK_DIR}/{scene_id:06d}/{image_id:06d}_{detection_id:06d}.png"


def transform_to_columns(T: npt.ArrayLike) -> dict[str, float]:
    flat = np.asarray(T, dtype=np.float64).reshape(16)
    return {c: float(v) for c, v in zip(TRANSFORM_COLUMNS, flat, strict=True)}


def columns_to_transform(row: Any) -> npt.NDArray[np.float64]:
    return np.array([float(row[c]) for c in TRANSFORM_COLUMNS], dtype=np.float64).reshape(4, 4)


# ----------------------------------------------------------------------------- detections


@dataclass(frozen=True)
class DetectionRecord:
    """One Detection plus the BOP ids needed to file it; the mask is written to ``mask_path``."""

    scene_id: int
    image_id: int
    detection: Detection
    time_s: float = float("nan")


class DetectionWriter:
    """Streams Detections of one stage run to ``<stage_dir>``; call :meth:`close` at the end."""

    def __init__(self, stage_dir: str | Path) -> None:
        self.dir = Path(stage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []

    def add(self, rec: DetectionRecord) -> None:
        d = rec.detection
        rel = mask_relpath(rec.scene_id, rec.image_id, d.detection_id)
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, (d.mask.astype(np.uint8) * 255))
        x, y, w, h = d.bbox_xywh
        self._rows.append(
            {
                "scene_id": int(rec.scene_id),
                "image_id": int(rec.image_id),
                "camera_id": d.camera_id,
                "object_id": int(d.object_id),
                "detection_id": int(d.detection_id),
                "score": float(d.score),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(w),
                "bbox_h": int(h),
                "mask_path": rel,
                "time_s": float(rec.time_s),
            }
        )

    def close(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=DETECTION_COLUMNS)
        df.to_parquet(self.dir / DETECTIONS_FILE, index=False)
        return df


def read_detections_table(stage_dir: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(stage_dir) / DETECTIONS_FILE)
    _validate_columns(df, DETECTION_COLUMNS, DETECTIONS_FILE)
    return df


def load_mask(stage_dir: str | Path, mask_path: str) -> npt.NDArray[np.bool_]:
    raw = iio.imread(Path(stage_dir) / mask_path)
    if raw.ndim == 3:
        raw = raw[..., 0]
    return np.asarray(raw) > 0


def detection_from_row(stage_dir: str | Path, row: Any) -> Detection:
    return Detection(
        camera_id=str(row["camera_id"]),
        object_id=int(row["object_id"]),
        mask=load_mask(stage_dir, str(row["mask_path"])),
        score=float(row["score"]),
        detection_id=int(row["detection_id"]),
    )


def read_detections(
    stage_dir: str | Path, scene_id: int, image_id: int, load_masks: bool = True
) -> list[Detection]:
    """Detections of one View, ordered by detection_id. Masks are all-False if not loaded."""
    df = read_detections_table(stage_dir)
    sel = df[(df.scene_id == scene_id) & (df.image_id == image_id)].sort_values("detection_id")
    out: list[Detection] = []
    for _, row in sel.iterrows():
        if load_masks:
            out.append(detection_from_row(stage_dir, row))
        else:
            out.append(
                Detection(
                    camera_id=str(row["camera_id"]),
                    object_id=int(row["object_id"]),
                    mask=np.zeros((1, 1), dtype=bool),
                    score=float(row["score"]),
                    detection_id=int(row["detection_id"]),
                )
            )
    return out


# ----------------------------------------------------------------------------- hypotheses


@dataclass(frozen=True)
class HypothesisRecord:
    scene_id: int
    image_id: int
    hypothesis: PoseHypothesis
    time_s: float = float("nan")


def hypothesis_to_row(rec: HypothesisRecord) -> dict[str, Any]:
    h = rec.hypothesis
    row: dict[str, Any] = {
        "scene_id": int(rec.scene_id),
        "image_id": int(rec.image_id),
        "camera_id": h.camera_id,
        "object_id": int(h.object_id),
        "detection_id": int(h.detection_id),
        "hypothesis_id": int(h.hypothesis_id),
        "stage": Stage(h.stage).value,
        "source": h.source,
        "rejection_reason": h.rejection_reason,
    }
    row.update(transform_to_columns(h.T_camera_object))
    row.update(h.signals.to_row())
    row["time_s"] = float(rec.time_s)
    return row


def hypothesis_from_row(row: Any) -> PoseHypothesis:
    reason = row["rejection_reason"]
    if reason is None or (isinstance(reason, float) and np.isnan(reason)):
        reason = None
    return PoseHypothesis(
        camera_id=str(row["camera_id"]),
        object_id=int(row["object_id"]),
        detection_id=int(row["detection_id"]),
        hypothesis_id=int(row["hypothesis_id"]),
        T_camera_object=columns_to_transform(row),
        stage=Stage(str(row["stage"])),
        signals=QualitySignals.from_row({k: row[k] for k in QualitySignals.field_names()}),
        source=str(row["source"]),
        rejection_reason=None if reason is None else str(reason),
    )


def write_hypotheses(stage_dir: str | Path, recs: list[HypothesisRecord]) -> pd.DataFrame:
    d = Path(stage_dir)
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([hypothesis_to_row(r) for r in recs], columns=HYPOTHESIS_COLUMNS)
    df["rejection_reason"] = df["rejection_reason"].astype(object)
    df.to_parquet(d / HYPOTHESES_FILE, index=False)
    return df


def read_hypotheses_table(stage_dir: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(stage_dir) / HYPOTHESES_FILE)
    _validate_columns(df, HYPOTHESIS_COLUMNS, HYPOTHESES_FILE)
    return df


def read_hypotheses(stage_dir: str | Path, scene_id: int, image_id: int) -> list[PoseHypothesis]:
    df = read_hypotheses_table(stage_dir)
    sel = df[(df.scene_id == scene_id) & (df.image_id == image_id)]
    sel = sel.sort_values(["detection_id", "hypothesis_id"])
    return [hypothesis_from_row(row) for _, row in sel.iterrows()]


# ----------------------------------------------------------------------------- stage bookkeeping


def write_stage_info(stage_dir: str | Path, info: dict[str, Any]) -> None:
    d = Path(stage_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / STAGE_INFO, "w") as f:
        json.dump({"artefact_schema_version": ARTEFACT_SCHEMA_VERSION, **info}, f, indent=2)


def read_stage_info(stage_dir: str | Path) -> dict[str, Any]:
    with open(Path(stage_dir) / STAGE_INFO) as f:
        data: dict[str, Any] = json.load(f)
    return data


def mark_success(stage_dir: str | Path) -> None:
    (Path(stage_dir) / SUCCESS_MARKER).touch()


def is_complete(stage_dir: str | Path) -> bool:
    return (Path(stage_dir) / SUCCESS_MARKER).exists()


def _validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing} (artefact schema mismatch)")
