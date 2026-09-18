"""ConfidenceModels: feature schema, labelled tables, Model H / F, calibration, Verdict (D11)."""

from binposert.confidence.calibration import (
    ReliabilityBin,
    RiskCoverage,
    brier,
    ece,
    pr_auc,
    reliability_table,
    risk_coverage,
    roc_auc,
    summary,
)
from binposert.confidence.labels import label_fused, label_hypotheses
from binposert.confidence.model import ConfidenceModel, file_fingerprint, platt_scaling
from binposert.confidence.schema import SCHEMA_VERSION, FeatureSchema, hypothesis_extras
from binposert.confidence.verdict import VerdictThresholds, choose_thresholds, verdict_rates

__all__ = [
    "SCHEMA_VERSION",
    "ConfidenceModel",
    "FeatureSchema",
    "ReliabilityBin",
    "RiskCoverage",
    "VerdictThresholds",
    "brier",
    "choose_thresholds",
    "ece",
    "file_fingerprint",
    "hypothesis_extras",
    "label_fused",
    "label_hypotheses",
    "platt_scaling",
    "pr_auc",
    "reliability_table",
    "risk_coverage",
    "roc_auc",
    "summary",
    "verdict_rates",
]
