from __future__ import annotations

from typing import Any

import numpy as np

from binposert.transforms import Mat4, make_T, rotation_distance_deg, rotvec_T, se3_distance
from binposert.types import SymmetryGroup

DEFAULT_CONTINUOUS_STEPS = 36


def from_bop_model_info(
    info: dict[str, Any], continuous_steps: int = DEFAULT_CONTINUOUS_STEPS
) -> SymmetryGroup:
    """Build a flat SymmetryGroup from one entry of BOP ``models_info.json``.

    Discrete symmetries are taken verbatim; each continuous axis is sampled at ``continuous_steps``
    evenly spaced angles (excluding 0, which the identity already covers). Products of a discrete
    symmetry with a continuous one are also included so the group is closed for the common
    "cylinder with a 180° flip" case.
    """
    discrete: list[Mat4] = [np.eye(4)]
    for flat in info.get("symmetries_discrete", []):
        S = np.asarray(flat, dtype=np.float64).reshape(4, 4)
        discrete.append(S)

    continuous: list[Mat4] = [np.eye(4)]
    for axis_spec in info.get("symmetries_continuous", []):
        axis = np.asarray(axis_spec["axis"], dtype=np.float64)
        offset = np.asarray(axis_spec.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        to_origin = make_T(np.eye(3), -offset)
        back = make_T(np.eye(3), offset)
        for k in range(1, continuous_steps):
            angle = 360.0 * k / continuous_steps
            continuous.append(back @ rotvec_T(axis, angle) @ to_origin)

    transforms: list[Mat4] = []
    for D in discrete:
        for C in continuous:
            transforms.append(D @ C)
    return SymmetryGroup(tuple(_dedupe(transforms)))


def align_to_reference(T_ref: Mat4, T: Mat4, group: SymmetryGroup) -> tuple[Mat4, int]:
    """Return ``T @ S*`` and the index of ``S* = argmin_S d_SE3(T_ref, T @ S)``.

    Right-multiplication: the symmetry acts in the object frame, before ``T`` maps object points
    out of it.
    """
    if group.is_trivial:
        return T, 0
    best_i, best_d, best_T = 0, np.inf, T
    for i, S in enumerate(group.transforms):
        TS = T @ S
        d = se3_distance(T_ref, TS)
        if d < best_d:
            best_i, best_d, best_T = i, d, TS
    return best_T, best_i


def sym_aware_rotation_distance_deg(T_a: Mat4, T_b: Mat4, group: SymmetryGroup) -> float:
    return min(rotation_distance_deg(T_a, T_b @ S) for S in group.transforms)


def sym_aware_se3_distance(T_a: Mat4, T_b: Mat4, group: SymmetryGroup) -> float:
    return min(se3_distance(T_a, T_b @ S) for S in group.transforms)


def _dedupe(Ts: list[Mat4], atol: float = 1e-6) -> list[Mat4]:
    out: list[Mat4] = []
    for T in Ts:
        if not any(np.allclose(T, U, atol=atol) for U in out):
            out.append(T)
    return out
