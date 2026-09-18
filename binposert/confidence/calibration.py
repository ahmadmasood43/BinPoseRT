"""Calibration and discrimination metrics of a ConfidenceModel (D11, RQ-D).

Discrimination: ROC-AUC and PR-AUC (average precision, successes are the positive class).
Calibration: Brier score and the expected calibration error over ``n_bins`` equal-width
confidence bins (the reliability diagram is the same binning, tabulated); an equal-mass variant is
reported alongside because equal-width bins are nearly empty at the extremes. Selective
prediction: the risk–coverage curve — accept the ``c`` most confident poses, plot their failure
rate — and its area (AURC), with the excess over the oracle ordering (E-AURC) so that a hard
dataset and a bad model are not confused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
DEFAULT_BINS = 10


def _arrays(y: npt.ArrayLike, p: npt.ArrayLike) -> tuple[npt.NDArray[np.bool_], F64]:
    yy = np.asarray(y, dtype=bool).ravel()
    pp = np.asarray(p, dtype=np.float64).ravel()
    if yy.shape != pp.shape:
        raise ValueError("labels and confidences differ in length")
    if len(pp) and (np.isnan(pp).any() or (pp < 0).any() or (pp > 1).any()):
        raise ValueError("confidences must lie in [0, 1]")
    return yy, pp


def roc_auc(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    yy, pp = _arrays(y, p)
    if yy.all() or not yy.any():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(yy, pp))


def pr_auc(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    yy, pp = _arrays(y, p)
    if not yy.any():
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(yy, pp))


def brier(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    yy, pp = _arrays(y, p)
    return float(np.mean((pp - yy.astype(np.float64)) ** 2)) if len(pp) else float("nan")


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    confidence: float  # mean confidence in the bin
    accuracy: float  # observed success rate in the bin


def reliability_table(
    y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = DEFAULT_BINS, strategy: str = "uniform"
) -> list[ReliabilityBin]:
    """Bins of the reliability diagram. ``uniform``: equal-width bins on [0, 1]; ``quantile``:
    equal-mass bins (edges at the confidence quantiles)."""
    yy, pp = _arrays(y, p)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        if len(pp) == 0:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            edges = np.unique(np.quantile(pp, np.linspace(0.0, 1.0, n_bins + 1)))
            edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")
    # right-closed bins except the first, so a confidence of exactly 1.0 lands in the last bin
    idx = np.clip(np.searchsorted(edges, pp, side="left") - 1, 0, len(edges) - 2)
    out: list[ReliabilityBin] = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        out.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                n=n,
                confidence=float(pp[sel].mean()) if n else float("nan"),
                accuracy=float(yy[sel].mean()) if n else float("nan"),
            )
        )
    return out


def ece(
    y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = DEFAULT_BINS, strategy: str = "uniform"
) -> float:
    """Expected calibration error: bin-mass-weighted |accuracy − confidence|."""
    bins = reliability_table(y, p, n_bins, strategy)
    total = sum(b.n for b in bins)
    if total == 0:
        return float("nan")
    return float(sum(b.n / total * abs(b.accuracy - b.confidence) for b in bins if b.n))


def mce(y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = DEFAULT_BINS) -> float:
    """Maximum calibration error over the non-empty equal-width bins."""
    bins = [b for b in reliability_table(y, p, n_bins) if b.n]
    return float(max(abs(b.accuracy - b.confidence) for b in bins)) if bins else float("nan")


@dataclass
class RiskCoverage:
    coverage: F64  # fraction of poses accepted, ascending
    risk: F64  # failure rate among the accepted
    aurc: float
    e_aurc: float  # AURC minus the oracle's (perfect ordering) AURC
    thresholds: F64  # confidence of the last accepted pose at each coverage

    def risk_at(self, coverage: float) -> float:
        """Failure rate when the most confident ``coverage`` fraction is accepted."""
        if len(self.coverage) == 0:
            return float("nan")
        k = int(np.searchsorted(self.coverage, coverage, side="left"))
        return float(self.risk[min(k, len(self.risk) - 1)])

    def coverage_at_risk(self, max_risk: float) -> float:
        """Largest coverage whose failure rate stays at or below ``max_risk``."""
        ok = np.nonzero(self.risk <= max_risk)[0]
        return float(self.coverage[ok[-1]]) if len(ok) else 0.0


def risk_coverage(y: npt.ArrayLike, p: npt.ArrayLike) -> RiskCoverage:
    yy, pp = _arrays(y, p)
    n = len(pp)
    if n == 0:
        z = np.zeros(0)
        return RiskCoverage(z, z, float("nan"), float("nan"), z)
    order = np.argsort(-pp, kind="stable")
    fail = (~yy[order]).astype(np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    coverage = k / n
    risk = np.cumsum(fail) / k
    aurc = float(np.mean(risk))
    # oracle: every success before every failure
    fail_sorted = np.sort(fail)
    oracle = float(np.mean(np.cumsum(fail_sorted) / k))
    return RiskCoverage(coverage, risk, aurc, aurc - oracle, pp[order])


def summary(y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Every headline number of the calibration analysis in one dict."""
    yy, pp = _arrays(y, p)
    rc = risk_coverage(yy, pp)
    return {
        "n": int(len(yy)),
        "base_rate": float(yy.mean()) if len(yy) else float("nan"),
        "roc_auc": roc_auc(yy, pp),
        "pr_auc": pr_auc(yy, pp),
        "brier": brier(yy, pp),
        "brier_base_rate": float(yy.mean() * (1 - yy.mean())) if len(yy) else float("nan"),
        "ece": ece(yy, pp, n_bins),
        "ece_quantile": ece(yy, pp, n_bins, "quantile"),
        "mce": mce(yy, pp, n_bins),
        "aurc": rc.aurc,
        "e_aurc": rc.e_aurc,
        "risk_at_50": rc.risk_at(0.5),
        "risk_at_80": rc.risk_at(0.8),
        "coverage_at_risk_05": rc.coverage_at_risk(0.05),
        "coverage_at_risk_10": rc.coverage_at_risk(0.10),
        "mean_confidence": float(pp.mean()) if len(pp) else float("nan"),
    }


__all__ = [
    "DEFAULT_BINS",
    "ReliabilityBin",
    "RiskCoverage",
    "brier",
    "ece",
    "mce",
    "pr_auc",
    "reliability_table",
    "risk_coverage",
    "roc_auc",
    "summary",
]
