"""On-disk artefacts exchanged between stages (D12): Detections and PoseHypotheses tables.

Layout of one stage directory (``outputs/<dataset>/<split>/<stage>/<hash>/``)::

    _SUCCESS                       written last; a directory without it is incomplete
    meta.json                      stage, artefact kind, schema version, source, config, inputs
    detections.parquet             DETECTION_COLUMNS         (segment)
    masks/<scene>_<image>_<det>.png                          (segment)
    pose_hypotheses.parquet        POSE_HYPOTHESIS_COLUMNS   (coarse_pose, refine)

This module deliberately imports only numpy / pandas / imageio so the estimator adapters can use it
inside their own containers (``pip install --no-deps``); everything typed lives in ``types.py``.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pandas as pd

from binposert.types import (
    ARTEFACT_SCHEMA_VERSION,
    DETECTION_COLUMNS,
    DETECTIONS_FILE,
    MASK_FILENAME,
    META_FILE,
    POSE_COLUMNS,
    POSE_HYPOTHESES_FILE,
    POSE_HYPOTHESIS_COLUMNS,
    SUCCESS_MARKER,
    Detection,
    PoseHypothesis,
    QualitySignals,
    Stage,
)

_INT_COLUMNS = {
    "scene_id",
    "image_id",
    "object_id",
    "detection_id",
    "hypothesis_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
}
_STR_COLUMNS = {"camera_id", "stage", "source", "rejection_reason", "mask_path"}


class ArtefactError(RuntimeError):
    """Raised when an artefact directory is missing, incomplete or violates the schema."""


# ----------------------------------------------------------------------------- markers / meta


def is_complete(directory: str | Path) -> bool:
    return (Path(directory) / SUCCESS_MARKER).exists()


def write_meta(directory: str | Path, **fields: Any) -> None:
    meta = {
        "schema_version": ARTEFACT_SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **fields,
    }
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / META_FILE, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)


def read_meta(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / META_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def mark_success(directory: str | Path) -> None:
    """Write the ``_SUCCESS`` marker. Must be the very last thing a stage does."""
    (Path(directory) / SUCCESS_MARKER).write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )


# ----------------------------------------------------------------------------- detections


def mask_relpath(scene_id: int, image_id: int, detection_id: int) -> str:
    return MASK_FILENAME.format(scene_id=scene_id, image_id=image_id, detection_id=detection_id)


class DetectionsWriter:
    """Accumulates Detection rows and mask PNGs for one artefact directory."""

    def __init__(self, directory: str | Path, source: str) -> None:
        self.directory = Path(directory)
        self.source = source
        self.rows: list[dict[str, Any]] = []
        (self.directory / "masks").mkdir(parents=True, exist_ok=True)

    def add(self, scene_id: int, image_id: int, det: Detection, time_s: float = math.nan) -> None:
        rel = mask_relpath(scene_id, image_id, det.detection_id)
        iio.imwrite(self.directory / rel, (det.mask.astype(np.uint8) * 255))
        x, y, w, h = det.bbox_xywh
        self.rows.append(
            {
                "scene_id": int(scene_id),
                "image_id": int(image_id),
                "camera_id": det.camera_id,
                "object_id": int(det.object_id),
                "detection_id": int(det.detection_id),
                "score": float(det.score),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "mask_path": rel,
                "source": self.source,
                "time_s": float(time_s),
            }
        )

    def add_mask(
        self,
        scene_id: int,
        image_id: int,
        object_id: int,
        detection_id: int,
        mask: npt.NDArray[np.bool_],
        score: float,
        time_s: float = math.nan,
    ) -> None:
        """Convenience for adapters that do not construct typed Detections."""
        det = Detection(
            camera_id=f"{image_id:06d}",
            object_id=object_id,
            mask=np.asarray(mask, dtype=bool),
            score=score,
            detection_id=detection_id,
        )
        self.add(scene_id, image_id, det, time_s)

    def finish(self, **meta: Any) -> pd.DataFrame:
        df = _frame(self.rows, DETECTION_COLUMNS)
        df.to_parquet(self.directory / DETECTIONS_FILE, index=False)
        write_meta(self.directory, artefact="detections", source=self.source, **meta)
        return df


def read_detections_table(directory: str | Path, validate: bool = True) -> pd.DataFrame:
    directory = Path(directory)
    if validate and not is_complete(directory):
        raise ArtefactError(f"{directory} has no {SUCCESS_MARKER} marker (incomplete stage output)")
    path = directory / DETECTIONS_FILE
    if not path.exists():
        raise ArtefactError(f"{path} not found")
    df = pd.read_parquet(path)
    if validate:
        _validate_columns(df, DETECTION_COLUMNS, path)
    return df


def read_mask(directory: str | Path, mask_path: str) -> npt.NDArray[np.bool_]:
    return np.asarray(iio.imread(Path(directory) / mask_path)) > 0


def detections_from_rows(directory: str | Path, rows: pd.DataFrame) -> list[Detection]:
    return [
        Detection(
            camera_id=str(r["camera_id"]),
            object_id=int(r["object_id"]),
            mask=read_mask(directory, str(r["mask_path"])),
            score=float(r["score"]),
            detection_id=int(r["detection_id"]),
        )
        for r in rows.to_dict("records")
    ]


# ----------------------------------------------------------------------------- pose hypotheses


class PoseHypothesesWriter:
    def __init__(self, directory: str | Path, source: str, stage: Stage = Stage.COARSE) -> None:
        self.directory = Path(directory)
        self.source = source
        self.stage = stage
        self.rows: list[dict[str, Any]] = []
        self.directory.mkdir(parents=True, exist_ok=True)

    def add(
        self, scene_id: int, image_id: int, hyp: PoseHypothesis, time_s: float = math.nan
    ) -> None:
        T = np.asarray(hyp.T_camera_object, dtype=np.float64).reshape(16)
        row: dict[str, Any] = {
            "scene_id": int(scene_id),
            "image_id": int(image_id),
            "camera_id": hyp.camera_id,
            "object_id": int(hyp.object_id),
            "detection_id": int(hyp.detection_id),
            "hypothesis_id": int(hyp.hypothesis_id),
            "stage": Stage(hyp.stage).value,
            "source": hyp.source or self.source,
            "rejection_reason": hyp.rejection_reason or "",
            **{c: float(v) for c, v in zip(POSE_COLUMNS, T)},  # noqa: B905 (py3.9 envs)
            "time_s": float(time_s),
            **hyp.signals.to_row(),
        }
        self.rows.append(row)

    def add_pose(
        self,
        scene_id: int,
        image_id: int,
        object_id: int,
        detection_id: int,
        hypothesis_id: int,
        T_camera_object: npt.ArrayLike,
        time_s: float = math.nan,
        **signals: float,
    ) -> None:
        """Convenience for adapters: ``signals`` are QualitySignals field names."""
        hyp = PoseHypothesis(
            camera_id=f"{image_id:06d}",
            object_id=object_id,
            detection_id=detection_id,
            hypothesis_id=hypothesis_id,
            T_camera_object=np.asarray(T_camera_object, dtype=np.float64).reshape(4, 4),
            stage=self.stage,
            signals=QualitySignals(**signals),
            source=self.source,
        )
        self.add(scene_id, image_id, hyp, time_s)

    def finish(self, **meta: Any) -> pd.DataFrame:
        df = _frame(self.rows, POSE_HYPOTHESIS_COLUMNS)
        df.to_parquet(self.directory / POSE_HYPOTHESES_FILE, index=False)
        write_meta(
            self.directory,
            artefact="pose_hypotheses",
            source=self.source,
            hypothesis_stage=self.stage.value,
            **meta,
        )
        return df


def read_pose_hypotheses_table(directory: str | Path, validate: bool = True) -> pd.DataFrame:
    directory = Path(directory)
    if validate and not is_complete(directory):
        raise ArtefactError(f"{directory} has no {SUCCESS_MARKER} marker (incomplete stage output)")
    path = directory / POSE_HYPOTHESES_FILE
    if not path.exists():
        raise ArtefactError(f"{path} not found")
    df = pd.read_parquet(path)
    if validate:
        _validate_columns(df, POSE_HYPOTHESIS_COLUMNS, path)
    return df


def pose_hypotheses_from_rows(rows: pd.DataFrame) -> list[PoseHypothesis]:
    signal_names = QualitySignals.field_names()
    out: list[PoseHypothesis] = []
    for r in rows.to_dict("records"):
        T = np.array([r[c] for c in POSE_COLUMNS], dtype=np.float64).reshape(4, 4)
        reason = str(r.get("rejection_reason", "") or "")
        out.append(
            PoseHypothesis(
                camera_id=str(r["camera_id"]),
                object_id=int(r["object_id"]),
                detection_id=int(r["detection_id"]),
                hypothesis_id=int(r["hypothesis_id"]),
                T_camera_object=T,
                stage=Stage(str(r["stage"])),
                signals=QualitySignals.from_row({k: r[k] for k in signal_names if k in r}),
                source=str(r.get("source", "")),
                rejection_reason=reason or None,
            )
        )
    return out


def pose_matrix(row: Any) -> npt.NDArray[np.float64]:
    """4x4 ``T_camera_object`` from one table row (Series, dict or namedtuple)."""
    get = row.get if hasattr(row, "get") else lambda c: getattr(row, c)
    return np.array([float(get(c)) for c in POSE_COLUMNS], dtype=np.float64).reshape(4, 4)


# ----------------------------------------------------------------------------- BOP / COCO RLE
# The BOP challenge distributes detections (e.g. the default CNOS detections) as JSON lists of
# ``{scene_id, image_id, category_id, bbox [x, y, w, h], score, time, segmentation}`` where
# ``segmentation`` is a run-length encoding of the mask in column-major (Fortran) order, starting
# with the length of the initial background run. ``counts`` is either an uncompressed list
# (bop_toolkit / CNOS) or the compressed COCO string.


def rle_to_mask(rle: dict[str, Any]) -> npt.NDArray[np.bool_]:
    h, w = int(rle["size"][0]), int(rle["size"][1])
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = _decode_coco_counts(counts)
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for run in counts:
        run = int(run)
        if value:
            flat[pos : pos + run] = True
        pos += run
        value = not value
    if pos != h * w:
        raise ArtefactError(f"RLE covers {pos} pixels, mask has {h * w}")
    return flat.reshape((w, h)).T  # Fortran order


def mask_to_rle(mask: npt.NDArray[np.bool_]) -> dict[str, Any]:
    """Uncompressed RLE exactly as CNOS / bop_toolkit ``mask_to_rle`` produce it."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    counts: list[int] = []
    if flat.size == 0:
        return {"counts": [0], "size": [int(mask.shape[0]), int(mask.shape[1])]}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    edges = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(edges).tolist()
    if flat[0]:  # encoding starts with a background run
        counts.append(0)
    counts.extend(int(r) for r in runs)
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _decode_coco_counts(s: str) -> list[int]:
    """COCO compressed RLE string → run lengths (pycocotools ``rleFrString``)."""
    counts: list[int] = []
    i = 0
    while i < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


# ----------------------------------------------------------------------------- internals


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(columns))
    for c in columns:
        if c in _INT_COLUMNS:
            df[c] = df[c].astype("int64")
        elif c in _STR_COLUMNS:
            df[c] = df[c].astype("string")
        else:
            df[c] = df[c].astype("float64")
    return df


def _validate_columns(df: pd.DataFrame, expected: tuple[str, ...], path: Path) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ArtefactError(
            f"{path}: missing columns {missing} (schema v{ARTEFACT_SCHEMA_VERSION})"
        )
