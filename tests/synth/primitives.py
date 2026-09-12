from __future__ import annotations

import numpy as np
import numpy.typing as npt
import trimesh

from binposert.symmetry import from_bop_model_info
from binposert.transforms import Mat4, make_T
from binposert.types import Mat3, ObjectModel, SymmetryGroup


def simple_K(w: int = 320, h: int = 240, f: float = 400.0) -> Mat3:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def box_model(
    object_id: int = 1, extents=(60.0, 40.0, 20.0), symmetric: bool = True
) -> ObjectModel:
    """A box (mm). With ``symmetric`` the 180° flips about each axis form its SymmetryGroup."""
    m = trimesh.creation.box(extents=extents)
    sym = SymmetryGroup.trivial()
    if symmetric:
        flips = []
        for axis in np.eye(3):
            R = trimesh.transformations.rotation_matrix(np.pi, axis)[:3, :3]
            flips.append(make_T(R, [0, 0, 0]).ravel().tolist())
        sym = from_bop_model_info({"symmetries_discrete": flips})
    return ObjectModel(
        object_id=object_id,
        vertices=np.asarray(m.vertices, dtype=np.float64),
        faces=np.asarray(m.faces, dtype=np.int64),
        diameter=float(np.linalg.norm(extents)),
        symmetry=sym,
    )


def cylinder_model(
    object_id: int = 2,
    radius: float = 15.0,
    height: float = 60.0,
    steps: int = 36,
    flip: bool = True,
) -> ObjectModel:
    """A cylinder along +z (mm) with a continuous axial symmetry and optional 180° flip about x."""
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    info: dict = {"symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]}
    if flip:
        R = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
        info["symmetries_discrete"] = [make_T(R, [0, 0, 0]).ravel().tolist()]
    return ObjectModel(
        object_id=object_id,
        vertices=np.asarray(m.vertices, dtype=np.float64),
        faces=np.asarray(m.faces, dtype=np.int64),
        diameter=float(np.sqrt((2 * radius) ** 2 + height**2)),
        symmetry=from_bop_model_info(info, continuous_steps=steps),
    )


def look_at_T_world_camera(
    eye: npt.ArrayLike, target: npt.ArrayLike = (0, 0, 0), up: npt.ArrayLike = (0, 0, 1)
) -> Mat4:
    """OpenCV camera convention: +z forward, +x right, +y down. Returns T_world_camera."""
    eye_v = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye_v
    z /= np.linalg.norm(z)
    up_v = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up_v)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)  # columns = camera axes in world
    return make_T(R, eye_v)
