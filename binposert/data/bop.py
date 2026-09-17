"""BOP dataset loader.

Layout (https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)::

    <root>/models/obj_XXXXXX.ply, models_info.json
    <root>/<split>/<scene_id:06d>/{scene_camera,scene_gt,scene_gt_info}.json
    <root>/<split>/<scene_id:06d>/{rgb,depth,mask,mask_visib}/...

Every image of a scene directory is a calibrated View of the same static Scene (D5), so a Scene
can be built from any subset of its image ids. Depth is converted to millimetres with
``depth_scale``. Ground-truth poses ``cam_R_m2c``/``cam_t_m2c`` are ``T_camera_object``;
``cam_R_w2c``/``cam_t_w2c`` are ``T_camera_world`` and are inverted into ``T_world_camera``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import trimesh

from binposert.data.depth_offset import shift_depth
from binposert.symmetry import from_bop_model_info
from binposert.transforms import invert, make_T
from binposert.types import GroundTruthPose, ObjectModel, Scene, View

MODEL_DIR_CANDIDATES = ("models", "models_cad", "models_eval", "models_reconst")


def camera_id_for(image_id: int) -> str:
    """The camera_id of a BOP image within its scene: the zero-padded image id."""
    return f"{int(image_id):06d}"


def load_targets(path: str | Path) -> dict[tuple[int, int], dict[int, int]]:
    """BOP test targets -> ``{(scene_id, image_id): {object_id: instance_count}}``."""
    with open(path) as f:
        items = json.load(f)
    out: dict[tuple[int, int], dict[int, int]] = {}
    for it in items:
        key = (int(it["scene_id"]), int(it["im_id"]))
        out.setdefault(key, {})[int(it["obj_id"])] = int(it.get("inst_count", 1))
    return out


def load_ply(path: str | Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    m = trimesh.load(str(path), force="mesh", process=False)
    assert isinstance(m, trimesh.Trimesh)
    return (
        np.asarray(m.vertices, dtype=np.float64),
        np.asarray(m.faces, dtype=np.int64),
    )


class BopDataset:
    """Read-only view of one split of a BOP-format dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        models_dir: str | None = None,
        continuous_symmetry_steps: int = 36,
        targets: str | Path | None = None,
        depth_shift_px: tuple[float, float] | None = None,
        name: str | None = None,
    ) -> None:
        """``targets`` names a BOP ``test_targets_*.json`` (relative to ``root`` or absolute); when
        given, enumeration and evaluation are restricted to the listed images and objects.
        ``depth_shift_px = (du, dv)`` translates every depth map onto the RGB image (the sensor's
        depth↔RGB registration offset, estimated GT-free with ``tools/depth_rgb_offset.py``).
        ``name`` labels outputs and run directories (default: the root directory's name; a camera
        view such as ``xyzibd_xyz`` is still the dataset ``xyzibd``)."""
        self.root = Path(root)
        self._name = name
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"BOP split directory not found: {self.split_dir}")
        self.models_dir = self._find_models_dir(models_dir)
        with open(self.models_dir / "models_info.json") as f:
            self.models_info: dict[str, dict[str, Any]] = json.load(f)
        self._continuous_steps = continuous_symmetry_steps
        self._models: dict[int, ObjectModel] = {}
        self._scene_json: dict[tuple[int, str], dict[str, Any]] = {}
        self.depth_shift_px = (
            (float(depth_shift_px[0]), float(depth_shift_px[1])) if depth_shift_px else None
        )
        self.targets: dict[tuple[int, int], dict[int, int]] | None = None
        self.targets_file: Path | None = None
        if targets is not None:
            tp = Path(targets)
            self.targets_file = tp if tp.is_absolute() else self.root / tp
            self.targets = load_targets(self.targets_file)

    # ------------------------------------------------------------------ enumeration

    @property
    def name(self) -> str:
        return self._name or self.root.name

    @property
    def scene_ids(self) -> list[int]:
        ids = sorted(
            int(p.name) for p in self.split_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
        if self.targets is not None:
            wanted = {s for s, _ in self.targets}
            ids = [s for s in ids if s in wanted]
        return ids

    def image_ids(self, scene_id: int) -> list[int]:
        ids = sorted(int(k) for k in self._scene_file(scene_id, "scene_camera"))
        if self.targets is not None:
            ids = [i for i in ids if (scene_id, i) in self.targets]
        return ids

    def target_object_ids(self, scene_id: int, image_id: int) -> list[int] | None:
        """Objects to evaluate in a View per the targets file (None = every annotated object)."""
        if self.targets is None:
            return None
        return sorted(self.targets.get((scene_id, image_id), {}))

    @property
    def object_ids(self) -> list[int]:
        return sorted(int(k) for k in self.models_info)

    def scene_dir(self, scene_id: int) -> Path:
        return self.split_dir / f"{scene_id:06d}"

    # ------------------------------------------------------------------ models

    def load_model(self, object_id: int) -> ObjectModel:
        if object_id not in self._models:
            info = self.models_info[str(object_id)]
            path = self.models_dir / f"obj_{object_id:06d}.ply"
            vertices, faces = load_ply(path)
            self._models[object_id] = ObjectModel(
                object_id=object_id,
                vertices=vertices,
                faces=faces,
                diameter=float(info["diameter"]),
                symmetry=from_bop_model_info(info, continuous_steps=self._continuous_steps),
                mesh_path=path,
            )
        return self._models[object_id]

    # ------------------------------------------------------------------ views

    def camera(self, scene_id: int, image_id: int) -> dict[str, Any]:
        return self._scene_file(scene_id, "scene_camera")[str(image_id)]

    def load_view(
        self,
        scene_id: int,
        image_id: int,
        load_rgb: bool = True,
        load_depth: bool = True,
    ) -> tuple[View, list[GroundTruthPose]]:
        cam = self.camera(scene_id, image_id)
        K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        # BOP-25 multi-camera datasets (XYZ-IBD) spell the extrinsics ``R_w2c`` / ``t_w2c``
        R_key = "cam_R_w2c" if "cam_R_w2c" in cam else "R_w2c"
        if R_key in cam:
            T_camera_world = make_T(cam[R_key], cam[R_key.replace("R_", "t_")])
            T_world_camera = invert(T_camera_world)
        else:
            T_world_camera = np.eye(4)

        rgb = self.load_rgb(scene_id, image_id) if load_rgb else None
        depth = self.load_depth(scene_id, image_id) if load_depth else None
        if rgb is not None:
            size = (int(rgb.shape[0]), int(rgb.shape[1]))
        elif depth is not None:
            size = (int(depth.shape[0]), int(depth.shape[1]))
        else:
            size = self._image_size(scene_id, image_id)

        view = View(
            camera_id=camera_id_for(image_id),
            K=K,
            T_world_camera=T_world_camera,
            rgb=rgb,
            depth=depth,
            image_size=size,
            scene_id=scene_id,
            image_id=image_id,
        )
        return view, self.ground_truth(scene_id, image_id)

    def ground_truth(self, scene_id: int, image_id: int) -> list[GroundTruthPose]:
        gts = self._scene_file(scene_id, "scene_gt").get(str(image_id), [])
        infos = self._scene_file(scene_id, "scene_gt_info", optional=True).get(str(image_id), [])
        out: list[GroundTruthPose] = []
        for i, g in enumerate(gts):
            visib = float(infos[i]["visib_fract"]) if i < len(infos) else 1.0
            out.append(
                GroundTruthPose(
                    object_id=int(g["obj_id"]),
                    T_camera_object=make_T(g["cam_R_m2c"], g["cam_t_m2c"]),
                    visible_fraction=visib,
                    gt_index=i,
                )
            )
        return out

    def load_rgb(self, scene_id: int, image_id: int) -> npt.NDArray[np.uint8]:
        d = self.scene_dir(scene_id)
        for ext in ("png", "jpg"):
            p = d / "rgb" / f"{image_id:06d}.{ext}"
            if p.exists():
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise OSError(f"cannot read {p}")
                return np.ascontiguousarray(bgr[:, :, ::-1])
        raise FileNotFoundError(f"no rgb image for scene {scene_id} image {image_id} in {d}")

    def load_depth(self, scene_id: int, image_id: int) -> npt.NDArray[np.float64] | None:
        p = self.scene_dir(scene_id) / "depth" / f"{image_id:06d}.png"
        if not p.exists():
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise OSError(f"cannot read {p}")
        scale = float(self.camera(scene_id, image_id).get("depth_scale", 1.0))
        depth = raw.astype(np.float64) * scale
        if self.depth_shift_px is not None:
            depth = shift_depth(depth, *self.depth_shift_px)
        return depth

    def gt_mask(
        self, scene_id: int, image_id: int, gt_index: int, visible_only: bool = True
    ) -> npt.NDArray[np.bool_]:
        sub = "mask_visib" if visible_only else "mask"
        p = self.scene_dir(scene_id) / sub / f"{image_id:06d}_{gt_index:06d}.png"
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(p)
        return raw > 0

    # ------------------------------------------------------------------ scenes

    def load_scene(
        self,
        scene_id: int,
        image_ids: list[int] | None = None,
        load_rgb: bool = True,
        load_depth: bool = True,
    ) -> Scene:
        ids = self.image_ids(scene_id) if image_ids is None else list(image_ids)
        views: list[View] = []
        gt: dict[str, tuple[GroundTruthPose, ...]] = {}
        for image_id in ids:
            view, gts = self.load_view(scene_id, image_id, load_rgb, load_depth)
            views.append(view)
            gt[view.camera_id] = tuple(gts)
        return Scene(dataset=self.name, scene_id=scene_id, views=tuple(views), ground_truth=gt)

    # ------------------------------------------------------------------ internals

    def _find_models_dir(self, models_dir: str | None) -> Path:
        if models_dir is not None:
            p = self.root / models_dir
            if not p.is_dir():
                raise FileNotFoundError(p)
            return p
        for cand in MODEL_DIR_CANDIDATES:
            if (self.root / cand / "models_info.json").exists():
                return self.root / cand
        raise FileNotFoundError(f"no models directory with models_info.json under {self.root}")

    def _scene_file(self, scene_id: int, name: str, optional: bool = False) -> dict[str, Any]:
        key = (scene_id, name)
        if key not in self._scene_json:
            p = self.scene_dir(scene_id) / f"{name}.json"
            if not p.exists():
                if optional:
                    self._scene_json[key] = {}
                    return self._scene_json[key]
                raise FileNotFoundError(p)
            with open(p) as f:
                self._scene_json[key] = json.load(f)
        return self._scene_json[key]

    def _image_size(self, scene_id: int, image_id: int) -> tuple[int, int]:
        d = self.scene_dir(scene_id)
        for sub, pat in (
            ("depth", f"{image_id:06d}.png"),
            ("rgb", f"{image_id:06d}.png"),
            ("rgb", f"{image_id:06d}.jpg"),
        ):
            p = d / sub / pat
            if p.exists():
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    return (int(raw.shape[0]), int(raw.shape[1]))
        raise FileNotFoundError(f"no image to infer size for scene {scene_id} image {image_id}")
