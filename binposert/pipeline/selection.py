"""Which (scene, image, object) triples a run covers: BOP targets file and/or explicit subsets."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from binposert.data import BopDataset, load_targets, targets_by_image


@dataclass(frozen=True)
class Selection:
    image_keys: tuple[tuple[int, int], ...]  # sorted (scene_id, image_id)
    objects: dict[tuple[int, int], dict[int, int]] | None  # per image {object_id: inst_count}

    def object_ids(self, scene_id: int, image_id: int) -> list[int] | None:
        if self.objects is None:
            return None
        return sorted(self.objects.get((scene_id, image_id), {}))

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.image_keys)

    def __len__(self) -> int:
        return len(self.image_keys)

    @property
    def scene_ids(self) -> list[int]:
        return sorted({s for s, _ in self.image_keys})

    def as_targets(self) -> dict[tuple[int, int], list[int] | None]:
        return {k: self.object_ids(*k) for k in self.image_keys}

    @classmethod
    def from_config(cls, dataset: BopDataset, cfg: dict[str, Any]) -> Selection:
        objects: dict[tuple[int, int], dict[int, int]] | None = None
        targets_file = cfg.get("targets")
        if targets_file:
            path = Path(dataset.root) / str(targets_file)
            objects = targets_by_image(load_targets(path))
            keys = sorted(objects)
        else:
            keys = [(s, i) for s in dataset.scene_ids for i in dataset.image_ids(s)]

        scene_ids = cfg.get("scene_ids")
        if scene_ids:
            allowed = {int(s) for s in scene_ids}
            keys = [k for k in keys if k[0] in allowed]
        image_ids = cfg.get("image_ids")
        if image_ids:
            allowed_i = {int(i) for i in image_ids}
            keys = [k for k in keys if k[1] in allowed_i]
        per_scene = cfg.get("max_images_per_scene")
        if per_scene:
            kept: list[tuple[int, int]] = []
            count: dict[int, int] = {}
            for k in keys:
                if count.get(k[0], 0) < int(per_scene):
                    kept.append(k)
                    count[k[0]] = count.get(k[0], 0) + 1
            keys = kept
        if objects is not None:
            objects = {k: objects[k] for k in keys}
        return cls(image_keys=tuple(keys), objects=objects)
