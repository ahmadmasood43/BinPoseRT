"""``Segmenter``: ``(View, object_ids) → list[Detection]`` (D4).

The View carries the rgb image and ``K`` that D4 names explicitly, plus the BOP ids a cache reader
needs to find its rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from binposert.types import Detection, View


class Segmenter(Protocol):
    name: str

    def segment(self, view: View, object_ids: Sequence[int] | None = None) -> list[Detection]:
        """Return one Detection per found object mask in ``view``.

        ``object_ids`` restricts the result to those ids; ``None`` means every onboarded object.
        Detection ids are unique within the View and stable across calls.
        """
        ...
