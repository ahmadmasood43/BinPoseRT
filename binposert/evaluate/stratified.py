"""Stratified summaries of per-GT evaluation rows (DECISIONS.md §4: visibility bins)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VISIBILITY_BINS: list[tuple[float, float]] = [(0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]


def stratify_by_visibility(
    rows: pd.DataFrame, bins: list[tuple[float, float]] | None = None
) -> list[dict[str, Any]]:
    """Pooled per-GT recall fractions per visibility bin. The last bin is closed on the right."""
    bins = VISIBILITY_BINS if bins is None else bins
    out: list[dict[str, Any]] = []
    for k, (lo, hi) in enumerate(bins):
        if len(rows) == 0:
            sel = rows
        else:
            v = rows["visible_fraction"]
            last = k == len(bins) - 1
            sel = rows[(v >= lo) & ((v <= hi) if last else (v < hi))]
        n = int(len(sel))
        entry: dict[str, Any] = {"lo": lo, "hi": hi, "n_gt": n}
        for col in ("ar_vsd", "ar_mssd", "ar_mspd", "success_0.1d"):
            vals = sel[col].to_numpy(dtype=float) if n and col in sel else np.array([])
            vals = vals[np.isfinite(vals)]
            entry[col] = float(vals.mean()) if len(vals) else float("nan")
        parts = [entry["ar_vsd"], entry["ar_mssd"], entry["ar_mspd"]]
        finite = [p for p in parts if np.isfinite(p)]
        entry["ar"] = float(np.mean(finite)) if finite else float("nan")
        out.append(entry)
    return out
