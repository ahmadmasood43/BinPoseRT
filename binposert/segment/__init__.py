"""Segmenter plugin interface and the two in-scope implementations (D4).

``GroundTruthSegmenter`` reads BOP ``mask_visib`` (the A0 upper bound); ``CachedSegmenter`` reads a
Detections artefact written by a GPU adapter (CNOS). Both return the same ``Detection`` objects, so
every downstream stage is agnostic to where masks came from.
"""

from binposert.segment.base import Segmenter
from binposert.segment.cached import CachedSegmenter
from binposert.segment.ground_truth import GroundTruthSegmenter

__all__ = ["CachedSegmenter", "GroundTruthSegmenter", "Segmenter"]
