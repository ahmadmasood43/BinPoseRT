"""Feature schema of the ConfidenceModels (D11).

A ConfidenceModel reads a fixed, versioned list of columns from a hypotheses / fused table and turns
it into a feature matrix. The rules are the ones D11 fixes:

* every QualitySignals field is a feature; a NaN (or infinite — an ICP that found no
  correspondences reports an infinite RMSE) signal becomes a constant (0) plus a
  ``<signal>_missing`` indicator, so single-view rows (no ``dispersion_*``, no ``n_views``) and
  multi-view rows share one schema and the model learns what "not available" means;
* distances in millimetres are divided by the object diameter (``<signal>/d``): a 3 mm ICP residual
  is noise on a 300 mm XYZ-IBD bar and a gross error on a 60 mm T-LESS part, and the same fitted
  model must serve both;
* a few columns outside QualitySignals are features too (``extras``): whether the Refinement
  rejected the hypothesis, the log diameter, and for FusedPoses the track shape (member count,
  symmetry alignments, weight mass) and the Model H probabilities of the members.

``SCHEMA_VERSION`` is bumped whenever a feature is added, removed or transformed differently; a
fitted model stores the version it was fitted with and refuses a table of another version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from binposert.types import QualitySignals

SCHEMA_VERSION = 1
SIGNALS: tuple[str, ...] = tuple(QualitySignals.field_names())
DIAMETER_NORMALISED: tuple[str, ...] = (
    "icp_rmse_mm",
    "displacement_mm",
    "multiview_residual_mm",
    "dispersion_mm",
)
MISSING_SUFFIX = "_missing"
IMPUTE_VALUE = 0.0

HYPOTHESIS_EXTRAS: tuple[str, ...] = ("rejected", "log_diameter")
FUSED_EXTRAS: tuple[str, ...] = (
    "rejected",  # share of members the Refinement rejected
    "log_diameter",
    "n_members",
    "n_aligned",
    "weight_sum",
    "p_h_mean",
    "p_h_min",
    "p_h_max",
)


def signal_feature_name(signal: str) -> str:
    return f"{signal}/d" if signal in DIAMETER_NORMALISED else signal


@dataclass(frozen=True)
class FeatureSchema:
    """The columns a ConfidenceModel reads and the features it derives from them."""

    kind: str  # "hypothesis" (Model H) | "fused" (Model F)
    version: int = SCHEMA_VERSION
    signals: tuple[str, ...] = SIGNALS
    extras: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def hypothesis() -> FeatureSchema:
        return FeatureSchema("hypothesis", extras=HYPOTHESIS_EXTRAS)

    @staticmethod
    def fused() -> FeatureSchema:
        return FeatureSchema("fused", extras=FUSED_EXTRAS)

    @property
    def feature_names(self) -> list[str]:
        names = [signal_feature_name(s) for s in self.signals]
        names += [f"{s}{MISSING_SUFFIX}" for s in self.signals]
        names += list(self.extras)
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def required_columns(self) -> list[str]:
        return [*self.signals, "diameter", *[e for e in self.extras if e != "log_diameter"]]

    def featurize(self, table: pd.DataFrame) -> npt.NDArray[np.float64]:
        """``(n_rows, n_features)`` float64 matrix; raises on a missing column or a non-finite
        extra (extras are never optional — the table builder fills them)."""
        missing = [c for c in self.required_columns() if c not in table.columns]
        if missing:
            raise ValueError(f"table lacks columns {missing} required by schema {self.kind}")
        n = len(table)
        diameter = table["diameter"].to_numpy(dtype=np.float64)
        if n and not (np.isfinite(diameter).all() and (diameter > 0).all()):
            raise ValueError("diameter must be finite and positive for every row")
        cols: list[npt.NDArray[np.float64]] = []
        for s in self.signals:
            v = table[s].to_numpy(dtype=np.float64)
            if s in DIAMETER_NORMALISED:
                v = v / diameter
            cols.append(np.where(np.isfinite(v), v, IMPUTE_VALUE))
        for s in self.signals:
            absent = ~np.isfinite(table[s].to_numpy(dtype=np.float64))
            cols.append(absent.astype(np.float64))
        for e in self.extras:
            if e == "log_diameter":
                v = np.log(diameter)
            else:
                v = table[e].to_numpy(dtype=np.float64)
            if n and not np.isfinite(v).all():
                raise ValueError(f"extra feature {e!r} has non-finite values")
            cols.append(v)
        if n == 0:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.stack(cols, axis=1)

    def features_of_signal(self, signal: str) -> list[str]:
        """Every feature derived from one QualitySignals field (for the ablate-one-signal table)."""
        if signal not in self.signals:
            raise KeyError(signal)
        return [signal_feature_name(signal), f"{signal}{MISSING_SUFFIX}"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "signals": list(self.signals),
            "extras": list(self.extras),
            "signals_schema_version": QualitySignals.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSchema:
        if int(d["version"]) != SCHEMA_VERSION:
            raise ValueError(
                f"feature schema version {d['version']} is not the current {SCHEMA_VERSION}"
            )
        if int(d.get("signals_schema_version", 1)) != QualitySignals.SCHEMA_VERSION:
            raise ValueError("QualitySignals schema version mismatch")
        return cls(
            kind=str(d["kind"]),
            version=int(d["version"]),
            signals=tuple(d["signals"]),
            extras=tuple(d["extras"]),
        )


def hypothesis_extras(table: pd.DataFrame) -> pd.DataFrame:
    """Add the Model H extras to a hypotheses table that carries ``rejection_reason`` and
    ``diameter``: ``rejected`` = 1 when the Refinement declined the candidate."""
    out = table.copy()
    reason = (
        out["rejection_reason"] if "rejection_reason" in out else pd.Series(None, index=out.index)
    )
    out["rejected"] = reason.map(lambda r: 0.0 if r is None or _is_nan(r) else 1.0).astype(float)
    return out


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


__all__ = [
    "DIAMETER_NORMALISED",
    "FUSED_EXTRAS",
    "HYPOTHESIS_EXTRAS",
    "IMPUTE_VALUE",
    "MISSING_SUFFIX",
    "SCHEMA_VERSION",
    "SIGNALS",
    "FeatureSchema",
    "hypothesis_extras",
    "signal_feature_name",
]
