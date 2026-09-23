"""Simulated pick (D14): ``T_robot_gripper = T_robot_world @ T_world_object @ T_object_gripper``.

A Grasp Pose is ``T_object_gripper``, authored once per ObjectModel and stored in a JSON file
(``models/grasps/<dataset>.json``). The gripper frame follows the usual parallel-jaw convention:
``z`` is the approach direction (pointing from the gripper into the object), ``x`` the closing
direction of the jaws, the origin the point between the finger tips at the grasp. The perception
system emits ``T_robot_gripper`` and stops there (CONTEXT.md); ``T_robot_world`` is the robot's
calibration and has no dataset value, so the demo takes it as a parameter.

:func:`author_bbox_grasp` proposes a grasp from the model's bounding box — jaws across the
smallest extent, approach along the middle one — which is how the committed files were
seeded; they are plain JSON and are meant to be edited by hand. :func:`render_pick` draws the
Scene's FusedPoses (coloured by Verdict), the used camera frames and the gripper at the
accepted picks with Open3D's offscreen renderer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from binposert.transforms import Mat4, make_T
from binposert.types import FusedPose, ObjectModel, Verdict, View

VERDICT_COLOURS = {
    Verdict.ACCEPT: (0.20, 0.70, 0.30),
    Verdict.REQUEST_VIEW: (0.95, 0.70, 0.15),
    Verdict.REJECT: (0.85, 0.25, 0.25),
}


@dataclass(frozen=True)
class GraspPose:
    object_id: int
    name: str
    T_object_gripper: Mat4
    jaw_opening_mm: float  # what the jaws must open to, for the drawing and a sanity check

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["T_object_gripper"] = np.asarray(self.T_object_gripper).ravel().tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraspPose:
        return cls(
            object_id=int(d["object_id"]),
            name=str(d["name"]),
            T_object_gripper=np.asarray(d["T_object_gripper"], dtype=np.float64).reshape(4, 4),
            jaw_opening_mm=float(d["jaw_opening_mm"]),
        )


def load_grasps(path: str | Path) -> dict[int, GraspPose]:
    with open(path) as f:
        items = json.load(f)
    return {int(it["object_id"]): GraspPose.from_dict(it) for it in items}


def save_grasps(path: str | Path, grasps: dict[int, GraspPose]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump([g.to_dict() for _, g in sorted(grasps.items())], f, indent=1)


def author_bbox_grasp(model: ObjectModel) -> GraspPose:
    """A grasp proposal from the axis-aligned bounding box in the object frame: the gripper origin
    at the box centre, jaws closing (``x``) across the smallest extent, approaching (``z``) along
    the middle extent from its positive side, ``y`` along the largest."""
    lo, hi = model.vertices.min(axis=0), model.vertices.max(axis=0)
    centre = 0.5 * (lo + hi)
    extents = hi - lo
    order = np.argsort(extents)  # smallest, middle, largest
    x = np.eye(3)[order[0]]
    z = -np.eye(3)[order[1]]  # approach from the +middle side, pointing into the object
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return GraspPose(
        object_id=model.object_id,
        name="bbox: jaws across the smallest extent, approach along the middle one",
        T_object_gripper=make_T(R, centre),
        jaw_opening_mm=float(extents[order[0]]),
    )


def pick_transform(T_robot_world: Mat4, T_world_object: Mat4, T_object_gripper: Mat4) -> Mat4:
    """The D14 chain, spelled out so the frame order is visible where it is used."""
    return T_robot_world @ T_world_object @ T_object_gripper


@dataclass(frozen=True)
class Pick:
    track_id: int
    object_id: int
    confidence: float
    verdict: Verdict
    T_world_object: Mat4
    T_world_gripper: Mat4
    T_robot_gripper: Mat4


def plan_picks(
    fused: list[FusedPose], grasps: dict[int, GraspPose], T_robot_world: Mat4
) -> list[Pick]:
    """One Pick per FusedPose that has a Grasp Pose, most confident first; the caller decides
    which Verdicts it acts on (the demo picks ``accept`` only)."""
    picks = []
    for fp in fused:
        g = grasps.get(fp.object_id)
        if g is None:
            continue
        T_wg = fp.T_world_object @ g.T_object_gripper
        picks.append(
            Pick(
                fp.track_id,
                fp.object_id,
                float(fp.confidence),
                fp.verdict,
                fp.T_world_object,
                T_wg,
                pick_transform(T_robot_world, fp.T_world_object, g.T_object_gripper),
            )
        )
    return sorted(picks, key=lambda p: -p.confidence)


# ----------------------------------------------------------------------------- drawing


def gripper_geometry(jaw_opening_mm: float, finger_mm: float = 40.0, width_mm: float = 25.0) -> Any:
    """A parallel-jaw gripper in its own frame: two fingers along ``−z`` closing along ``x`` at
    ``jaw_opening_mm`` (plus clearance), a palm bar joining them; the tips sit at the origin."""
    import open3d as o3d

    gap = jaw_opening_mm + 10.0
    t = 6.0  # finger thickness
    parts = []
    for sx in (-1.0, 1.0):
        f = o3d.geometry.TriangleMesh.create_box(t, width_mm, finger_mm)
        f.translate([sx * gap / 2 - t / 2, -width_mm / 2, -finger_mm])
        parts.append(f)
    palm = o3d.geometry.TriangleMesh.create_box(gap + 2 * t, width_mm, t)
    palm.translate([-(gap + 2 * t) / 2, -width_mm / 2, -finger_mm - t])
    parts.append(palm)
    mesh = parts[0]
    for p in parts[1:]:
        mesh += p
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.25, 0.35, 0.85])
    return mesh


def _mesh(model: ObjectModel, T: Mat4, colour: tuple[float, float, float]) -> Any:
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(model.vertices), o3d.utility.Vector3iVector(model.faces)
    )
    m.transform(T)
    m.compute_vertex_normals()
    m.paint_uniform_color(list(colour))
    return m


def _frame(T: Mat4, size: float) -> Any:
    import open3d as o3d

    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    f.transform(T)
    return f


def _camera_wire(view: View, size: float) -> Any:
    """A small pyramid for a camera pose."""
    import open3d as o3d

    h, w = view.image_size
    fx, fy, cx, cy = view.K[0, 0], view.K[1, 1], view.K[0, 2], view.K[1, 2]
    z = size
    corners = np.array(
        [
            [0, 0, 0],
            [(0 - cx) / fx * z, (0 - cy) / fy * z, z],
            [(w - cx) / fx * z, (0 - cy) / fy * z, z],
            [(w - cx) / fx * z, (h - cy) / fy * z, z],
            [(0 - cx) / fx * z, (h - cy) / fy * z, z],
        ]
    )
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(np.asarray(lines))
    )
    ls.transform(view.T_world_camera)
    ls.paint_uniform_color([0.3, 0.3, 0.3])
    return ls


def render_pick(
    models: dict[int, ObjectModel],
    fused: list[FusedPose],
    picks: list[Pick],
    used_views: list[View],
    grasps: dict[int, GraspPose],
    path: str | Path,
    T_robot_world: Mat4 | None = None,
    image_size: tuple[int, int] = (720, 960),
    eye: npt.ArrayLike | None = None,
    n_gripper: int = 3,
) -> None:
    """Draw the Scene: every FusedPose coloured by Verdict, the used camera frames, the world
    (and robot) frames, and the gripper at the ``n_gripper`` most confident accepted picks."""
    import open3d as o3d

    h, w = image_size
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    lit = o3d.visualization.rendering.MaterialRecord()
    lit.shader = "defaultLit"
    line = o3d.visualization.rendering.MaterialRecord()
    line.shader = "unlitLine"
    line.line_width = 2.0
    diam = float(np.median([models[fp.object_id].diameter for fp in fused])) if fused else 100.0
    for i, fp in enumerate(fused):
        scene.add_geometry(
            f"obj{i}",
            _mesh(models[fp.object_id], fp.T_world_object, VERDICT_COLOURS[fp.verdict]),
            lit,
        )
    for i, v in enumerate(used_views):
        scene.add_geometry(f"cam{i}", _camera_wire(v, 0.6 * diam), line)
        scene.add_geometry(f"camf{i}", _frame(v.T_world_camera, 0.3 * diam), lit)
    scene.add_geometry("world", _frame(np.eye(4), 0.8 * diam), lit)
    if T_robot_world is not None:
        scene.add_geometry("robot", _frame(np.linalg.inv(T_robot_world), 1.2 * diam), lit)
    accepted = [p for p in picks if p.verdict == Verdict.ACCEPT][:n_gripper]
    for i, p in enumerate(accepted):
        g = gripper_geometry(grasps[p.object_id].jaw_opening_mm)
        g.transform(p.T_world_gripper)
        scene.add_geometry(f"grip{i}", g, lit)
        scene.add_geometry(f"gripf{i}", _frame(p.T_world_gripper, 0.5 * diam), lit)
    pts = (
        np.concatenate([fp.T_world_object[:3, 3][None] for fp in fused])
        if fused
        else np.zeros((1, 3))
    )
    # frame the bin, not its outliers: the median position and the 10–90 % extent
    centre = np.median(pts, axis=0)
    span = float(np.ptp(np.percentile(pts, [10, 90], axis=0), axis=0).max()) + 3 * diam
    # look from the cameras' side of the bin (world z may point away from them), slightly off
    # their mean optical axis so the depth of the pile is visible
    if used_views:
        look = np.mean([v.T_world_camera[:3, 2] for v in used_views], axis=0)
        side = np.mean([v.T_world_camera[:3, 0] for v in used_views], axis=0)
    else:
        look, side = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    look /= np.linalg.norm(look)
    side -= look * (side @ look)
    side /= np.linalg.norm(side)
    if eye is None:
        eye = centre - look * 1.6 * span + side * 0.8 * span
    up = -look  # towards the cameras
    renderer.setup_camera(40.0, centre.tolist(), np.asarray(eye, dtype=float).tolist(), up.tolist())
    scene.scene.set_sun_light([-0.3, -0.5, -1.0], [1.0, 1.0, 1.0], 80000)
    scene.scene.enable_sun_light(True)
    img = renderer.render_to_image()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(path), img)


def chain_text(pick: Pick, grasp: GraspPose, T_robot_world: Mat4) -> str:
    """The transform chain of one pick, printed matrix by matrix."""

    def m(T: Mat4) -> str:
        return "\n".join("    " + " ".join(f"{x:9.3f}" for x in row) for row in T)

    return "\n".join(
        [
            f"track {pick.track_id}, object {pick.object_id}: Confidence {pick.confidence:.3f}, "
            f"Verdict {pick.verdict.value}, grasp '{grasp.name}'",
            "  T_robot_world =",
            m(T_robot_world),
            "  T_world_object =",
            m(pick.T_world_object),
            "  T_object_gripper =",
            m(grasp.T_object_gripper),
            "  T_robot_gripper = T_robot_world @ T_world_object @ T_object_gripper =",
            m(pick.T_robot_gripper),
        ]
    )


__all__ = [
    "VERDICT_COLOURS",
    "GraspPose",
    "Pick",
    "author_bbox_grasp",
    "chain_text",
    "gripper_geometry",
    "load_grasps",
    "pick_transform",
    "plan_picks",
    "render_pick",
    "save_grasps",
]
