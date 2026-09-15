from __future__ import annotations

from typing import Protocol

from binposert.types import Detection, View


class Segmenter(Protocol):
    """``(View, object_ids) -> list[Detection]``. Neural segmenters run remotely and are consumed
    through :class:`CachedSegmenter`; only ground truth runs in-process."""

    name: str

    def segment(self, view: View, object_ids: list[int]) -> list[Detection]: ...
