"""Verdict thresholds (D11): Confidence -> accept / reject / request_view.

``accept`` when the Confidence is at least ``tau_accept``, ``reject`` when it is at most
``tau_reject``, ``request_view`` in between. Both thresholds are chosen on the val rows for a
target precision: ``tau_accept`` is the lowest threshold at which the accepted poses are at least
``accept_precision`` successes (the widest acceptance the target allows), ``tau_reject`` the
highest threshold at which the rejected poses are at least ``reject_precision`` failures. If no
threshold reaches a target the corresponding band is empty (accept nothing / reject nothing) and
the thresholds say so; if the two bands overlap the acceptance band wins, the rejection band is
pushed below it and ``reject_satisfiable`` reports whether the pushed band still meets its target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from binposert.types import Verdict

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VerdictThresholds:
    tau_accept: float
    tau_reject: float
    accept_precision: float  # target
    reject_precision: float  # target
    min_band_rows: int = 20  # a band must hold this many val rows before its precision counts
    n_fit_rows: int = 0
    accept_satisfiable: bool = True
    reject_satisfiable: bool = True

    def verdict(self, confidence: float) -> Verdict:
        if confidence >= self.tau_accept:
            return Verdict.ACCEPT
        if confidence <= self.tau_reject:
            return Verdict.REJECT
        return Verdict.REQUEST_VIEW

    def verdicts(self, confidence: npt.ArrayLike) -> list[Verdict]:
        p = np.asarray(confidence, dtype=np.float64).ravel()
        out = np.full(len(p), Verdict.REQUEST_VIEW.value, dtype=object)
        out[p >= self.tau_accept] = Verdict.ACCEPT.value
        out[(p <= self.tau_reject) & (p < self.tau_accept)] = Verdict.REJECT.value
        return [Verdict(v) for v in out]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerdictThresholds:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> VerdictThresholds:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def choose_thresholds(
    y: npt.ArrayLike,
    confidence: npt.ArrayLike,
    accept_precision: float = 0.95,
    reject_precision: float = 0.90,
    min_band_rows: int = 20,
) -> VerdictThresholds:
    yy = np.asarray(y, dtype=bool).ravel()
    pp = np.asarray(confidence, dtype=np.float64).ravel()
    if yy.shape != pp.shape:
        raise ValueError("labels and confidences differ in length")
    n = len(pp)
    order = np.argsort(-pp, kind="stable")  # descending confidence
    succ_desc = np.cumsum(yy[order]).astype(np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    prec_accept = succ_desc / k  # precision when the top-k are accepted
    # the widest acceptance meeting the target: the largest k (lowest threshold) whose precision
    # holds; ties in confidence are resolved by taking whole ties (k must end a tie run)
    p_desc = pp[order]
    ends_run = np.ones(n, dtype=bool)
    ends_run[:-1] = p_desc[:-1] != p_desc[1:]
    ok = (prec_accept >= accept_precision) & (k >= min_band_rows) & ends_run
    if ok.any():
        k_acc = int(np.nonzero(ok)[0][-1])
        tau_accept = float(p_desc[k_acc])
        accept_ok = True
    else:
        tau_accept, accept_ok = float("inf"), False

    order_r = np.argsort(pp, kind="stable")  # ascending
    fail_asc = np.cumsum(~yy[order_r]).astype(np.float64)
    prec_reject = fail_asc / k
    p_asc = pp[order_r]
    ends_run_r = np.ones(n, dtype=bool)
    ends_run_r[:-1] = p_asc[:-1] != p_asc[1:]
    ok_r = (prec_reject >= reject_precision) & (k >= min_band_rows) & ends_run_r
    if ok_r.any():
        k_rej = int(np.nonzero(ok_r)[0][-1])
        tau_reject = float(p_asc[k_rej])
        reject_ok = True
    else:
        tau_reject, reject_ok = float("-inf"), False
    if reject_ok and accept_ok and tau_reject >= tau_accept:
        # the bands overlap: keep the acceptance band (its precision is the binding one) and
        # shrink the rejection band under it; the shrunk band is a shorter prefix of the
        # ascending order, so its precision is re-checked and the flag says whether it holds
        below = p_asc[p_asc < tau_accept]
        tau_reject = float(below.max()) if len(below) else float("-inf")
        band = pp <= tau_reject
        n_band = int(band.sum())
        reject_ok = n_band >= min_band_rows and float((~yy[band]).mean()) >= reject_precision
    return VerdictThresholds(
        tau_accept=tau_accept,
        tau_reject=tau_reject,
        accept_precision=accept_precision,
        reject_precision=reject_precision,
        min_band_rows=min_band_rows,
        n_fit_rows=n,
        accept_satisfiable=accept_ok,
        reject_satisfiable=reject_ok,
    )


def verdict_rates(
    y: npt.ArrayLike, confidence: npt.ArrayLike, thresholds: VerdictThresholds
) -> dict[str, Any]:
    """Share of each Verdict and the success rate inside each band."""
    yy = np.asarray(y, dtype=bool).ravel()
    pp = np.asarray(confidence, dtype=np.float64).ravel()
    v = np.asarray([x.value for x in thresholds.verdicts(pp)], dtype=object)
    out: dict[str, Any] = {"n": int(len(pp))}
    for name in (Verdict.ACCEPT, Verdict.REJECT, Verdict.REQUEST_VIEW):
        sel = v == name.value
        n = int(sel.sum())
        out[f"{name.value}_rate"] = float(n / len(pp)) if len(pp) else float("nan")
        out[f"{name.value}_n"] = n
        out[f"{name.value}_success_rate"] = float(yy[sel].mean()) if n else float("nan")
    out["accept_precision"] = out["accept_success_rate"]
    out["reject_precision"] = 1.0 - out["reject_success_rate"] if out["reject_n"] else float("nan")
    return out


__all__ = ["VerdictThresholds", "choose_thresholds", "verdict_rates"]
