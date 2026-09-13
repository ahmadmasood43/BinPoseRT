"""PoseEstimator plugin interface and the cache-backed / synthetic implementations (D3).

Neural estimators (FoundPose, MegaPose) never run inside the core: their Docker adapters write a
PoseHypotheses artefact which ``CachedPoseEstimator`` serves. ``PerturbedGroundTruthEstimator``
is the CPU stand-in that makes every downstream stage testable on the mini fixture with a known,
controllable initial error.
"""

from binposert.pose.base import PoseEstimator
from binposert.pose.cached import CachedPoseEstimator
from binposert.pose.synthetic import PerturbedGroundTruthEstimator

__all__ = ["CachedPoseEstimator", "PerturbedGroundTruthEstimator", "PoseEstimator"]
