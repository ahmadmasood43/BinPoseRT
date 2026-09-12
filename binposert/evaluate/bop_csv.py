"""BOP result file: ``scene_id,im_id,obj_id,score,R,t,time`` (R row-major 9, t 3 in mm)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binposert.transforms import Mat4, make_T


@dataclass(frozen=True)
class PosePrediction:
    scene_id: int
    image_id: int
    object_id: int
    score: float
    T_camera_object: Mat4
    time_s: float = -1.0


def write_bop_csv(path: str | Path, preds: list[PosePrediction]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for p in preds:
            R = " ".join(f"{x:.8f}" for x in p.T_camera_object[:3, :3].ravel())
            t = " ".join(f"{x:.6f}" for x in p.T_camera_object[:3, 3])
            w.writerow(
                [p.scene_id, p.image_id, p.object_id, f"{p.score:.6f}", R, t, f"{p.time_s:.6f}"]
            )


def read_bop_csv(path: str | Path) -> list[PosePrediction]:
    out: list[PosePrediction] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            R = np.array([float(x) for x in row["R"].split()]).reshape(3, 3)
            t = np.array([float(x) for x in row["t"].split()])
            out.append(
                PosePrediction(
                    scene_id=int(row["scene_id"]),
                    image_id=int(row["im_id"]),
                    object_id=int(row["obj_id"]),
                    score=float(row["score"]),
                    T_camera_object=make_T(R, t),
                    time_s=float(row.get("time", -1.0)),
                )
            )
    return out
