"""Stratified reports over the per-GT rows of a LocalisationReport (DECISIONS.md §4).

Visibility bins default to [0.10, 0.30), [0.30, 0.60), [0.60, 1.00]; initial-error bins (Beta) are
[0, 5), [5, 10), [10, 20), [20, inf) mm. Recall in a bin is the *pooled* mean over its GT rows —
not the per-object average BOP uses for the headline AR — because bins are typically too small to
average per object.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

VISIBILITY_BINS: tuple[float, ...] = (0.10, 0.30, 0.60, 1.00)
INITIAL_ERROR_BINS_MM: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, float("inf"))


def stratify(
    rows: pd.DataFrame | Sequence[dict[str, Any]],
    by: str = "visible_fraction",
    edges: Sequence[float] = VISIBILITY_BINS,
    metrics: Sequence[str] = ("mssd_ar", "mspd_ar", "success_0.1d"),
) -> pd.DataFrame:
    """One row per bin: ``bin``, ``lo``, ``hi``, ``n_gt`` and the mean of each metric."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    edges = list(edges)
    out: list[dict[str, Any]] = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        last = k == len(edges) - 2
        if len(df):
            sel = (df[by] >= lo) & ((df[by] <= hi) if last else (df[by] < hi))
            part = df[sel]
        else:
            part = df
        row: dict[str, Any] = {
            "bin": f"[{lo:g}, {hi:g}{']' if last else ')'}",
            "lo": lo,
            "hi": hi,
            "n_gt": int(len(part)),
        }
        for m in metrics:
            row[m] = float(part[m].mean()) if len(part) and m in part else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(table: pd.DataFrame, title: str, metrics: Sequence[str] | None = None) -> str:
    metrics = list(metrics or [c for c in table.columns if c not in ("bin", "lo", "hi", "n_gt")])
    head = "| bin | n_gt | " + " | ".join(metrics) + " |"
    sep = "|---|---:|" + "|".join(["---:"] * len(metrics)) + "|"
    lines = [f"### {title}", "", head, sep]
    for r in table.to_dict("records"):
        vals = " | ".join("n/a" if np.isnan(r[m]) else f"{r[m]:.3f}" for m in metrics)
        lines.append(f"| {r['bin']} | {r['n_gt']} | {vals} |")
    return "\n".join(lines) + "\n"
