"""ConfidenceModels (D11): Model H scores PoseHypotheses, Model F scores FusedPoses.

Both are the same object: a :class:`FeatureSchema`, a standardisation (mean / scale per feature),
a classifier fitted on held-out val scenes and an optional affine recalibration of its logit
(``calibration = (a, b)``: ``z' = a·z + b``, Platt scaling fitted on out-of-fold logits of the
same val rows, which takes out the over-confidence an in-sample fit on a handful of scenes has).
The default classifier is an L2-regularised logistic regression — a dozen coefficients anyone can
read; a one-hidden-layer MLP comparator exists for the "does logistic under-fit?" check D11 asks
for and is only kept if it clearly wins. The fitted model is a JSON file: no pickles, so a model
fitted today still loads after a scikit-learn upgrade and the stage cache can hash it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from binposert.confidence.schema import FeatureSchema

F64 = npt.NDArray[np.float64]
KINDS = ("logistic", "mlp")
MODEL_FORMAT_VERSION = 1


@dataclass
class ConfidenceModel:
    schema: FeatureSchema
    kind: str = "logistic"
    mean: F64 = field(default_factory=lambda: np.zeros(0))
    scale: F64 = field(default_factory=lambda: np.ones(0))
    # logistic: one (W, b); mlp: a list of (W, b) layers with ReLU between, sigmoid at the end
    layers: list[tuple[F64, F64]] = field(default_factory=list)
    calibration: tuple[float, float] = (1.0, 0.0)  # logit -> a * logit + b
    provenance: dict[str, Any] = field(default_factory=dict)  # what it was fitted on

    # ------------------------------------------------------------------ inference

    def raw_decision(self, X: F64) -> F64:
        """Logit of the classifier before recalibration, for features ``X``."""
        if X.shape[1] != self.schema.n_features:
            raise ValueError(f"expected {self.schema.n_features} features, got {X.shape[1]}")
        z = (X - self.mean) / self.scale
        for i, (W, b) in enumerate(self.layers):
            z = z @ W + b
            if i < len(self.layers) - 1:
                z = np.maximum(z, 0.0)
        return np.asarray(z[:, 0], dtype=np.float64)

    def decision(self, X: F64) -> F64:
        """Logit of the Confidence (recalibrated) for features ``X``."""
        a, b = self.calibration
        return a * self.raw_decision(X) + b

    def raw_logits(self, table: pd.DataFrame) -> F64:
        if len(table) == 0:
            return np.zeros(0, dtype=np.float64)
        return self.raw_decision(self.schema.featurize(table))

    def recalibrated(self, a: float, b: float) -> ConfidenceModel:
        """The same classifier with the affine logit recalibration ``(a, b)``."""
        return ConfidenceModel(
            schema=self.schema,
            kind=self.kind,
            mean=self.mean,
            scale=self.scale,
            layers=self.layers,
            calibration=(float(a), float(b)),
            provenance={**self.provenance, "calibration": [float(a), float(b)]},
        )

    def predict_proba(self, table: pd.DataFrame) -> F64:
        """Confidence of every row of a table carrying the schema's columns."""
        if len(table) == 0:
            return np.zeros(0, dtype=np.float64)
        return _sigmoid(self.decision(self.schema.featurize(table)))

    def coefficients(self) -> dict[str, float]:
        """Standardised logistic coefficients by feature name (empty for an MLP)."""
        if self.kind != "logistic":
            return {}
        W = self.layers[0][0][:, 0]
        return {n: float(w) for n, w in zip(self.schema.feature_names, W, strict=True)}

    @property
    def intercept(self) -> float:
        return float(self.layers[-1][1][0])

    # ------------------------------------------------------------------ fitting

    @classmethod
    def fit(
        cls,
        schema: FeatureSchema,
        table: pd.DataFrame,
        y: npt.ArrayLike,
        kind: str = "logistic",
        C: float = 1.0,
        hidden: int = 16,
        seed: int = 0,
        provenance: dict[str, Any] | None = None,
    ) -> ConfidenceModel:
        if kind not in KINDS:
            raise ValueError(f"unknown model kind {kind!r}; choose from {KINDS}")
        X = schema.featurize(table)
        yy = np.asarray(y, dtype=bool)
        if len(X) != len(yy):
            raise ValueError("table and labels differ in length")
        if yy.all() or not yy.any():
            raise ValueError("labels must contain both successes and failures")
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0  # constant feature: leave it centred at zero
        Z = (X - mean) / scale
        layers: list[tuple[F64, F64]]
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
            clf.fit(Z, yy)
            layers = [(np.asarray(clf.coef_.T, dtype=np.float64), np.asarray(clf.intercept_))]
        else:
            from sklearn.neural_network import MLPClassifier

            # sklearn adds 0.5 * alpha * ||W||^2 / n_samples to the *mean* log-loss, while
            # LogisticRegression(C) minimises 0.5 * ||w||^2 + C * sum(loss): the same penalty per
            # sample is alpha = 1 / C
            mlp = MLPClassifier(
                hidden_layer_sizes=(hidden,),
                alpha=1.0 / C,
                max_iter=2000,
                random_state=seed,
            )
            mlp.fit(Z, yy)
            layers = [
                (np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64))
                for W, b in zip(mlp.coefs_, mlp.intercepts_, strict=True)
            ]
        prov = {
            "n_rows": int(len(yy)),
            "n_success": int(yy.sum()),
            "base_rate": float(yy.mean()),
            "C": float(C),
            "hidden": int(hidden) if kind == "mlp" else 0,
            "seed": int(seed),
            **(provenance or {}),
        }
        return cls(schema=schema, kind=kind, mean=mean, scale=scale, layers=layers, provenance=prov)

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "kind": self.kind,
            "schema": self.schema.to_dict(),
            "feature_names": self.schema.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in self.layers],
            "calibration": list(self.calibration),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConfidenceModel:
        if int(d.get("format_version", 0)) != MODEL_FORMAT_VERSION:
            raise ValueError(f"unsupported confidence model format {d.get('format_version')}")
        schema = FeatureSchema.from_dict(d["schema"])
        if list(d["feature_names"]) != schema.feature_names:
            raise ValueError("feature names in the model file do not match the schema")
        return cls(
            schema=schema,
            kind=str(d["kind"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            layers=[
                (np.asarray(x["W"], dtype=np.float64), np.asarray(x["b"], dtype=np.float64))
                for x in d["layers"]
            ],
            calibration=(float(d["calibration"][0]), float(d["calibration"][1]))
            if "calibration" in d
            else (1.0, 0.0),
            provenance=dict(d.get("provenance", {})),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceModel:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def platt_scaling(logits: npt.ArrayLike, y: npt.ArrayLike) -> tuple[float, float]:
    """``(a, b)`` of the one-dimensional logistic fit ``P(success) = σ(a·logit + b)``; fitted on
    *out-of-fold* logits it corrects the over-confidence of an in-sample fit."""
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    yy = np.asarray(y, dtype=bool)
    if yy.all() or not yy.any():
        return 1.0, 0.0
    clf = LogisticRegression(C=1e6, max_iter=5000)
    clf.fit(z, yy)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def file_fingerprint(path: str | Path) -> str:
    """Short content hash of a model file, so a stage config that names the file changes its
    cache hash when the model is refitted."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return h[:16]


def _sigmoid(z: F64) -> F64:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


__all__ = ["KINDS", "ConfidenceModel", "file_fingerprint", "platt_scaling"]
