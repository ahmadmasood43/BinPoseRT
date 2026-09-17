"""The ``associate`` and ``fuse`` stages (D10, D12).

A *view group* is the set of Views of one BOP scene that the multi-view stages treat as one Scene.
Groups are built from the dataset's target images (or a BOP-25 multi-view targets file) by
striding: with ``n_views = k`` and ``n`` candidate images per scene, group ``g`` holds the images at
indices ``g, g + n//k, g + 2·n//k, …`` so consecutive (nearly identical) T-LESS viewpoints never
share a group and every image is used at most once. ``n_views = 1`` reproduces the single-view rows.

``associate`` writes ``tracks.parquet`` (every hypothesis of every group with its ``track_id``,
fusion weight and the extrinsics it was lifted with — optionally perturbed for the calibration
sweep) and ``groups.json``. ``fuse`` writes ``fused.parquet`` (one row per ObjectTrack) and a
``hypotheses.parquet`` holding the fused pose projected into every View of its group, so the
unchanged ``evaluate`` stage scores multi-view rows exactly as it scores single-view ones.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from binposert.data import BopDataset
from binposert.multiview import (
    AssociationParams,
    FusionParams,
    JointIcpParams,
    JointRefiner,
    WeightParams,
    WorldHypothesis,
    associate,
    association_summary,
    fuse_track,
    hypothesis_weight,
    lift_to_world,
)
from binposert.pipeline.artefacts import (
    HYPOTHESIS_COLUMNS,
    HypothesisRecord,
    hypothesis_from_row,
    hypothesis_to_row,
    load_mask,
    read_detections_table,
    read_hypotheses_table,
    transform_to_columns,
)
from binposert.pipeline.pool import map_scenes
from binposert.refine import GateParams
from binposert.transforms import Mat4, invert, rotvec_T
from binposert.types import ObjectTrack, PoseHypothesis, QualitySignals, Stage, View

TRACKS_FILE = "tracks.parquet"
GROUPS_FILE = "groups.json"
FUSED_FILE = "fused.parquet"
EXTRINSIC_COLUMNS = [f"Twc_{i}{j}" for i in range(4) for j in range(4)]
FUSION_METHODS = ("none", "best", "mean")


# ----------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class GroupParams:
    n_views: int = 2
    source: str = "targets"  # "targets" (the dataset's image list) | "targets_multiview"
    targets_multiview: str = ""  # BOP-25 multi-view targets file, relative to the dataset root
    max_groups_per_scene: int = 0  # 0 = every disjoint strided group


@dataclass(frozen=True)
class ExtrinsicNoise:
    """Calibration-error sweep: every View but the first of a group gets a rigid perturbation in
    its camera frame, of fixed magnitude and a direction drawn from ``seed``."""

    t_mm: float = 0.0
    deg: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class AssociateStageParams:
    groups: GroupParams = dataclasses.field(default_factory=GroupParams)
    association: AssociationParams = dataclasses.field(default_factory=AssociationParams)
    weights: WeightParams = dataclasses.field(default_factory=WeightParams)
    extrinsic_noise: ExtrinsicNoise = dataclasses.field(default_factory=ExtrinsicNoise)


@dataclass(frozen=True)
class FuseStageParams:
    method: str = "mean"  # "none" (per-view pass-through) | "best" | "mean"
    joint_icp: JointIcpParams | None = None  # the D10 step-4 polish, when set
    mean_iterations: int = 5
    project_to: str = "all"  # "all": every View of the group | "members": the track's Views only


def associate_params_from_config(section: dict[str, Any]) -> AssociateStageParams:
    p = dict(section.get("params", {}))
    return AssociateStageParams(
        groups=GroupParams(**p.get("groups", {})),
        association=AssociationParams(**p.get("association", {})),
        weights=WeightParams(**p.get("weights", {})),
        extrinsic_noise=ExtrinsicNoise(**p.get("extrinsic_noise", {})),
    )


def fuse_params_from_config(section: dict[str, Any]) -> FuseStageParams:
    p = dict(section.get("params", {}))
    joint = p.get("joint_icp")
    joint_params: JointIcpParams | None = None
    if joint:
        j = dict(joint)
        gate = GateParams(**j.pop("gate", {}))
        if "corr_dist_factors" in j:
            j["corr_dist_factors"] = tuple(float(x) for x in j["corr_dist_factors"])
        joint_params = JointIcpParams(gate=gate, **j)
    method = str(p.get("method", "mean"))
    if method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method {method!r}; choose from {FUSION_METHODS}")
    return FuseStageParams(
        method=method,
        joint_icp=joint_params,
        mean_iterations=int(p.get("mean_iterations", 5)),
        project_to=str(p.get("project_to", "all")),
    )


# ----------------------------------------------------------------------------- view groups


def strided_groups(image_ids: list[int], n_views: int, max_groups: int = 0) -> list[list[int]]:
    ids = sorted(image_ids)
    n, k = len(ids), max(1, n_views)
    if n < k:
        return []
    m = n // k
    groups = [[ids[g + j * m] for j in range(k)] for g in range(m)]
    return groups[:max_groups] if max_groups > 0 else groups


def load_multiview_targets(path: Path) -> dict[int, list[int]]:
    """BOP-25 ``test_targets_multiview_*.json`` -> ``{scene_id: [image_id, …]}`` (listed order)."""
    with open(path) as f:
        items = json.load(f)
    out: dict[int, list[int]] = {}
    for it in items:
        ims = out.setdefault(int(it["scene_id"]), [])
        for pair in it["im_id"]:
            image_id = int(pair[1]) if isinstance(pair, list) else int(pair)
            if image_id not in ims:
                ims.append(image_id)
    return out


def make_groups(dataset: BopDataset, p: GroupParams) -> dict[int, list[list[int]]]:
    """``{scene_id: [[image_id, …], …]}`` for every scene of the dataset."""
    if p.source == "targets":
        candidates = {sid: dataset.image_ids(sid) for sid in dataset.scene_ids}
    elif p.source == "targets_multiview":
        path = Path(p.targets_multiview)
        if not path.is_absolute():
            path = dataset.root / path
        listed = load_multiview_targets(path)
        candidates = {sid: listed[sid] for sid in dataset.scene_ids if sid in listed}
    else:
        raise ValueError(f"unknown group source {p.source!r}")
    groups: dict[int, list[list[int]]] = {}
    for sid, ims in candidates.items():
        if p.source == "targets_multiview":
            # keep the listed order (it is the benchmark's view order), chunked, not sorted
            k = max(1, p.n_views)
            m = len(ims) // k
            gs = [[ims[g + j * m] for j in range(k)] for g in range(m)]
            groups[sid] = gs[: p.max_groups_per_scene] if p.max_groups_per_scene > 0 else gs
        else:
            groups[sid] = strided_groups(ims, p.n_views, p.max_groups_per_scene)
    return groups


def perturbed_extrinsics(
    T_world_camera: Mat4, noise: ExtrinsicNoise, scene_id: int, group_id: int, image_id: int
) -> Mat4:
    if noise.t_mm == 0.0 and noise.deg == 0.0:
        return T_world_camera
    rng = np.random.default_rng([noise.seed, scene_id, group_id, image_id])
    axis = rng.normal(size=3)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    delta = rotvec_T(axis, noise.deg, t=noise.t_mm * direction)
    return T_world_camera @ delta


# ----------------------------------------------------------------------------- associate


def associate_scene(
    scene_id: int,
    dataset: BopDataset,
    pose_dir: str,
    groups: list[list[int]],
    params: AssociateStageParams,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hyps = read_hypotheses_table(pose_dir)
    hyps = hyps[hyps.scene_id == scene_id]
    models = {oid: dataset.load_model(oid) for oid in sorted(hyps.object_id.unique())}
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "n_tracks": 0,
        "n_hypotheses": 0,
        "n_gated_out": 0,
        "n_new_tracks_after_first_view": 0,
    }
    next_track = 0
    for group_id, image_ids in enumerate(groups):
        by_view: dict[str, list[WorldHypothesis]] = {}
        extrinsics: dict[str, Mat4] = {}
        for k, image_id in enumerate(image_ids):
            view, _ = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=False)
            T_wc = view.T_world_camera
            if k > 0:
                T_wc = perturbed_extrinsics(
                    T_wc, params.extrinsic_noise, scene_id, group_id, image_id
                )
            extrinsics[view.camera_id] = T_wc
            by_view[view.camera_id] = []
            for _, h_row in hyps[hyps.image_id == image_id].iterrows():
                h = hypothesis_from_row(h_row)
                w = hypothesis_weight(h.signals, h.rejection_reason is not None, params.weights)
                by_view[view.camera_id].append(lift_to_world(h, T_wc, w))
        res = associate(by_view, models, params.association, first_track_id=next_track)
        next_track += len(res.tracks)
        s = association_summary(res)
        for key in stats:
            stats[key] += s[key]
        in_group = hyps[hyps.image_id.isin(image_ids)]
        times: dict[int, float] = {
            int(i): float(t) for i, t in zip(in_group["image_id"], in_group["time_s"], strict=True)
        }
        for t in res.tracks:
            for m in res.members[t.track_id]:
                h = m.hypothesis
                row: dict[str, Any] = hypothesis_to_row(
                    HypothesisRecord(
                        scene_id, int(h.camera_id), h, times.get(int(h.camera_id), float("nan"))
                    )
                )
                row.update({"group_id": group_id, "track_id": t.track_id, "weight": m.weight})
                row.update(
                    dict(
                        zip(
                            EXTRINSIC_COLUMNS, extrinsics[h.camera_id].ravel().tolist(), strict=True
                        )
                    )
                )
                rows.append(row)
    return rows, stats


def run_associate(
    dataset: BopDataset, pose_dir: Path, out_dir: Path, params: AssociateStageParams
) -> dict[str, Any]:
    t0 = time.perf_counter()
    groups = make_groups(dataset, params.groups)
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for sid in dataset.scene_ids:
        r, s = associate_scene(sid, dataset, str(pose_dir), groups.get(sid, []), params)
        rows.extend(r)
        for k, v in s.items():
            stats[k] = stats.get(k, 0) + v
    cols = HYPOTHESIS_COLUMNS + ["group_id", "track_id", "weight", *EXTRINSIC_COLUMNS]
    table = pd.DataFrame(rows, columns=cols)
    table["rejection_reason"] = table["rejection_reason"].astype(object)
    table.to_parquet(out_dir / TRACKS_FILE, index=False)
    with open(out_dir / GROUPS_FILE, "w") as f:
        json.dump({str(k): v for k, v in groups.items()}, f)
    n_groups = sum(len(v) for v in groups.values())
    sizes = (
        table.groupby(["scene_id", "group_id", "track_id"]).size()
        if len(table)
        else pd.Series(dtype=int)
    )
    summary = {
        **stats,
        "n_groups": n_groups,
        "n_views": params.groups.n_views,
        "views_per_track": {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "wall_seconds": time.perf_counter() - t0,
    }
    return summary


# ----------------------------------------------------------------------------- fuse


def _view_of(row: Any) -> Mat4:
    return np.asarray([float(row[c]) for c in EXTRINSIC_COLUMNS], dtype=np.float64).reshape(4, 4)


def fuse_scene(
    scene_id: int,
    dataset: BopDataset,
    seg_dir: str,
    assoc_dir: str,
    params: FuseStageParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tracks = pd.read_parquet(Path(assoc_dir) / TRACKS_FILE)
    tracks = tracks[tracks.scene_id == scene_id]
    with open(Path(assoc_dir) / GROUPS_FILE) as f:
        groups: list[list[int]] = json.load(f).get(str(scene_id), [])
    dets = read_detections_table(seg_dir) if params.joint_icp is not None else None
    fused_rows: list[dict[str, Any]] = []
    hyp_rows: list[dict[str, Any]] = []
    refiners: dict[int, JointRefiner] = {}
    views_cache: dict[tuple[int, bool], View] = {}

    def view(image_id: int, with_depth: bool) -> View:
        # keyed on with_depth too: a View cached without depth for the projection step must not
        # be handed to the joint polish (it would see every track as "no_depth")
        key = (image_id, with_depth)
        if key not in views_cache:
            v, _ = dataset.load_view(scene_id, image_id, load_rgb=False, load_depth=with_depth)
            views_cache[key] = v
        return views_cache[key]

    for group_id, image_ids in enumerate(groups):
        g = tracks[tracks.group_id == group_id]
        if params.method == "none":
            for _, row in g.iterrows():
                hyp_rows.append({c: row[c] for c in HYPOTHESIS_COLUMNS})
            continue
        for track_key, members_df in g.groupby("track_id"):
            track_id = int(cast(Any, track_key))
            t0 = time.perf_counter()
            members: list[WorldHypothesis] = []
            extrinsics: dict[str, Mat4] = {}
            for _, row in members_df.iterrows():
                h = hypothesis_from_row(row)
                T_wc = _view_of(row)
                extrinsics[h.camera_id] = T_wc
                members.append(lift_to_world(h, T_wc, float(row["weight"])))
            object_id = int(members_df.object_id.iloc[0])
            model = dataset.load_model(object_id)
            track = ObjectTrack(track_id, object_id, [m.hypothesis for m in members])
            fp = FusionParams(method=params.method, mean_iterations=params.mean_iterations)
            out = fuse_track(track, members, model, fp)
            T_fused = out.fused.T_world_object
            signals = out.fused.signals
            joint: dict[str, Any] = {"joint_accepted": None, "joint_reason": None}
            if params.joint_icp is not None and dets is not None:
                if object_id not in refiners:
                    refiners[object_id] = JointRefiner(model, params.joint_icp)
                d_scene = dets[dets.scene_id == scene_id]
                pairs = []
                for m in members:
                    image_id = int(m.hypothesis.camera_id)
                    v = view(image_id, with_depth=True)
                    v = dataclasses.replace(v, T_world_camera=extrinsics[m.hypothesis.camera_id])
                    d_row = d_scene[
                        (d_scene.image_id == image_id)
                        & (d_scene.detection_id == m.hypothesis.detection_id)
                    ]
                    if len(d_row) == 0:
                        continue
                    pairs.append((v, load_mask(seg_dir, str(d_row.iloc[0]["mask_path"]))))
                jo = refiners[object_id].polish(T_fused, pairs)
                T_fused = jo.T_world_object
                signals = _with_joint_signals(signals, jo)
                joint = {
                    "joint_accepted": jo.accepted,
                    "joint_reason": jo.reason,
                    "joint_fitness": jo.registration.fitness if jo.registration else float("nan"),
                    "joint_rmse_mm": jo.registration.inlier_rmse_mm
                    if jo.registration
                    else float("nan"),
                    "joint_displacement_mm": jo.displacement_mm,
                    "joint_displacement_deg": jo.displacement_deg,
                    "joint_iou": jo.silhouette_iou,
                    "joint_n_scene_points": jo.n_scene_points,
                    "joint_seconds": jo.seconds,
                }
                joint.update(
                    {f"cand_{k}": v for k, v in transform_to_columns(jo.T_candidate).items()}
                )
            seconds = time.perf_counter() - t0
            fused_rows.append(
                {
                    "scene_id": scene_id,
                    "group_id": group_id,
                    "track_id": track_id,
                    "object_id": object_id,
                    "image_ids": ",".join(str(i) for i in image_ids),
                    "member_image_ids": ",".join(m.hypothesis.camera_id for m in members),
                    "n_members": len(members),
                    "n_aligned": out.n_aligned,
                    "method": params.method,
                    "weight_sum": float(out.weights.sum()),
                    **transform_to_columns(T_fused),
                    **{f"ref_{k}": v for k, v in transform_to_columns(out.T_reference).items()},
                    **signals.to_row(),
                    **joint,
                    "seconds": seconds,
                }
            )
            # project into the group's Views for evaluation: an object seen in one View is a
            # world-frame object and is predicted in every View (``all``), or only where a
            # Detection exists (``members``)
            by_cam = {m.hypothesis.camera_id: m.hypothesis for m in members}
            prior = (
                float(np.nanmax(members_df["time_s"].to_numpy(dtype=float)))
                if len(members_df)
                else 0.0
            )
            targets = (
                image_ids
                if params.project_to == "all"
                else [int(m.hypothesis.camera_id) for m in members]
            )
            for image_id in targets:
                v = view(image_id, with_depth=False)
                cid = v.camera_id
                T_wc = extrinsics.get(cid, v.T_world_camera)
                src = by_cam.get(cid)
                h = PoseHypothesis(
                    camera_id=cid,
                    object_id=object_id,
                    detection_id=src.detection_id if src else -1,
                    hypothesis_id=track_id,
                    T_camera_object=invert(T_wc) @ T_fused,
                    stage=Stage.REFINED,
                    signals=signals,
                    source=f"fused:{params.method}" + ("+joint" if params.joint_icp else ""),
                    rejection_reason=None,
                )
                hyp_rows.append(
                    hypothesis_to_row(HypothesisRecord(scene_id, image_id, h, prior + seconds))
                )
    return fused_rows, hyp_rows


def _with_joint_signals(s: QualitySignals, jo: Any) -> QualitySignals:
    out = dataclasses.replace(s)
    if jo.registration is not None:
        out.multiview_residual_mm = float(jo.registration.inlier_rmse_mm)
        if jo.accepted:
            out.icp_fitness = float(jo.registration.fitness)
            out.icp_rmse_mm = float(jo.registration.inlier_rmse_mm)
    if not np.isnan(jo.silhouette_iou):
        out.silhouette_iou = float(jo.silhouette_iou)
    return out


def run_fuse(
    dataset: BopDataset,
    seg_dir: Path,
    assoc_dir: Path,
    out_dir: Path,
    params: FuseStageParams,
    n_workers: int = 1,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    scene_ids = dataset.scene_ids
    jobs = [(sid, dataset, str(seg_dir), str(assoc_dir), params) for sid in scene_ids]
    if n_workers > 1 and len(scene_ids) > 1 and params.joint_icp is not None:
        results = map_scenes(fuse_scene, jobs, n_workers)
    else:
        results = [fuse_scene(*job) for job in jobs]
    fused = [r for rows, _ in results for r in rows]
    hyps = [h for _, rows in results for h in rows]
    table = pd.DataFrame(hyps, columns=HYPOTHESIS_COLUMNS)
    table["rejection_reason"] = table["rejection_reason"].astype(object)
    table.to_parquet(out_dir / "hypotheses.parquet", index=False)
    fused_df = pd.DataFrame(fused)
    if len(fused_df):
        for c in ("joint_reason",):
            if c in fused_df:
                fused_df[c] = fused_df[c].astype(object)
    fused_df.to_parquet(out_dir / FUSED_FILE, index=False)
    shutil.copyfile(assoc_dir / GROUPS_FILE, out_dir / GROUPS_FILE)  # evaluate scores these images
    summary: dict[str, Any] = {
        "method": params.method,
        "joint_icp": params.joint_icp is not None,
        "n_tracks": len(fused_df),
        "n_projected_hypotheses": len(table),
        "wall_seconds": time.perf_counter() - t0,
    }
    if len(fused_df):
        multi = fused_df[fused_df.n_views > 1]
        summary.update(
            {
                "n_multi_view_tracks": int(len(multi)),
                "n_members_aligned_by_symmetry": int(fused_df.n_aligned.sum()),
                "median_dispersion_mm": float(multi.dispersion_mm.median()) if len(multi) else None,
                "median_dispersion_deg": float(multi.dispersion_deg.median())
                if len(multi)
                else None,
                "median_seconds": float(fused_df.seconds.median()),
            }
        )
        if "joint_accepted" in fused_df and fused_df.joint_accepted.notna().any():
            ja = fused_df.joint_accepted.astype(bool)
            summary["joint_acceptance_rate"] = float(ja.mean())
            summary["joint_reasons"] = {
                str(k): int(v) for k, v in fused_df.joint_reason.value_counts(dropna=True).items()
            }
    return summary


__all__ = [
    "FUSED_FILE",
    "GROUPS_FILE",
    "TRACKS_FILE",
    "AssociateStageParams",
    "ExtrinsicNoise",
    "FuseStageParams",
    "GroupParams",
    "associate_params_from_config",
    "fuse_params_from_config",
    "make_groups",
    "run_associate",
    "run_fuse",
    "strided_groups",
]
