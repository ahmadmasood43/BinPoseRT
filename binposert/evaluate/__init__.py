"""Pose-error metrics, BOP result files and the BOP localisation protocol (D11, D15)."""

from binposert.evaluate.bop_csv import PosePrediction, read_bop_csv, write_bop_csv
from binposert.evaluate.localisation import LocalisationReport, evaluate_localisation
from binposert.evaluate.metrics import add, adi, mspd, mssd, project, vsd
from binposert.evaluate.stratified import (
    INITIAL_ERROR_BINS_MM,
    VISIBILITY_BINS,
    stratify,
    to_markdown,
)

__all__ = [
    "INITIAL_ERROR_BINS_MM",
    "VISIBILITY_BINS",
    "LocalisationReport",
    "PosePrediction",
    "add",
    "adi",
    "evaluate_localisation",
    "mspd",
    "mssd",
    "project",
    "read_bop_csv",
    "stratify",
    "to_markdown",
    "vsd",
    "write_bop_csv",
]
