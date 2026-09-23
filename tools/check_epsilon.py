#!/usr/bin/env python
"""Verify that Epsilon's artefacts are complete and consistent before the milestone is closed.

    uv run python tools/check_epsilon.py --dataset xyzibd --groups 0 1 2 3

Checks, each printed as PASS / FAIL: every A9 row has a run manifest whose stages all carry
``_SUCCESS``; its nbv stage wrote the fused / tracks / hypotheses tables, ``groups.json`` and
``episodes.json``; every FusedPose carries a Confidence in [0, 1] and a Verdict; the projected
hypotheses are the FusedPoses times the reference Views; the reference Views of start group
``g`` are Gamma's strided group ``g``; the Views used respect the policy and the budget (fixed =
the strided order, random / nbv = start + distinct pool images, nested prefixes across budgets);
the ``fixed`` budget-4 row reproduces the A8 k = 4 row's per-GT errors on the same images (the
loop is the same pipeline); the Verdict-stopped rows stop for a recorded reason; the report,
curve figure and JSON exist and the report holds the NBV − random difference; the grasp file
covers every object of the dataset and the pick figure exists; no row log contains a
traceback. Exit status 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.active import strided_order  # noqa: E402
from binposert.viz.grasp import load_grasps  # noqa: E402


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))
        self.failures += 0 if ok else 1
        return ok


def manifest(outputs: Path, name: str, dataset: str) -> dict | None:
    p = outputs / "runs" / f"{name}_{dataset}" / "run_manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def check_row(c: Checker, outputs: Path, name: str, dataset: str, g: int) -> dict | None:
    m = manifest(outputs, name, dataset)
    if not c.check(m is not None, f"{name}: run manifest"):
        return None
    assert m is not None
    dirs = {k: Path(v["dir"]) for k, v in m["stages"].items()}
    c.check(all((d / "_SUCCESS").exists() for d in dirs.values()), f"{name}: every stage complete")
    d = dirs["nbv"]
    files = [
        "fused.parquet",
        "tracks.parquet",
        "hypotheses.parquet",
        "groups.json",
        "episodes.json",
    ]
    c.check(all((d / f).exists() for f in files), f"{name}: nbv artefacts")
    fused = pd.read_parquet(d / "fused.parquet")
    hyps = pd.read_parquet(d / "hypotheses.parquet")
    groups = json.loads((d / "groups.json").read_text())
    episodes = json.loads((d / "episodes.json").read_text())
    c.check(
        fused.confidence.between(0, 1).all()
        and fused.verdict.isin(["accept", "reject", "request_view"]).all(),
        f"{name}: every FusedPose has Confidence in [0, 1] and a Verdict",
        f"{len(fused)} FusedPoses",
    )
    n_ref = sum(len(gs[0]) for gs in groups.values())
    expected = int(
        sum(len(fused[fused.scene_id == int(s)]) * len(gs[0]) for s, gs in groups.items())
    )
    c.check(
        len(hyps) == expected,
        f"{name}: hypotheses = FusedPoses x reference Views",
        f"{len(hyps)} vs {expected}",
    )
    c.check(hyps.confidence.notna().all(), f"{name}: projected hypotheses carry the Confidence")
    ok_ref = all(e["reference"] == strided_order(e["pool"], g, 4)[:4] for e in episodes)
    c.check(ok_ref, f"{name}: reference Views are Gamma's strided group {g}")
    policy = m["config"]["nbv"]["params"]["policy"]
    budget = int(m["config"]["nbv"]["params"]["budget"])
    stop = m["config"]["nbv"]["params"]["stop"]
    ok_used = True
    for e in episodes:
        used = e["views_used"]
        cap = len(e["pool"]) if budget <= 0 else min(budget, len(e["pool"]))
        ok_used &= (
            used[0] == e["start"] and len(set(used)) == len(used) and set(used) <= set(e["pool"])
        )
        ok_used &= len(used) <= cap
        if stop == "budget":
            ok_used &= len(used) == cap
        if policy == "fixed":
            ok_used &= used == strided_order(e["pool"], g, 4)[: len(used)]
        if policy in ("nbv", "oracle"):
            ok_used &= all(
                s["reason"] in (policy, "fallback") for s in e["steps"] if s["chosen"] is not None
            )
    c.check(ok_used, f"{name}: Views used follow policy={policy}, budget={budget}, stop={stop}")
    c.check(
        all(e["stop"].startswith("stop:") for e in episodes),
        f"{name}: every episode stopped for a reason",
    )
    log = outputs / "runs" / f"{name}_{dataset}.log"
    c.check(log.exists() and "Traceback" not in log.read_text(), f"{name}: log without traceback")
    return {"manifest": m, "dirs": dirs, "episodes": episodes, "fused": fused, "n_ref": n_ref}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="xyzibd")
    ap.add_argument("--groups", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    args = ap.parse_args()
    outputs = Path(args.outputs)
    c = Checker()
    ds = args.dataset

    rows: dict[str, dict] = {}
    for g in args.groups:
        names = [f"A9_fixed_b1_g{g}", f"A9_fixed_b0_g{g}"]
        names += [f"A9_{p}_b{b}_g{g}" for b in args.budgets for p in ("fixed", "random", "nbv")]
        names += [f"A9_{p}_verdict_g{g}" for p in ("nbv", "random")]
        # the ablation and the oracle are optional rows: checked when present
        optional = [f"A9_{p}_b{b}_g{g}" for b in args.budgets for p in ("nbve", "oracle")]
        for name in names + [n for n in optional if manifest(outputs, n, ds) is not None]:
            r = check_row(c, outputs, name, ds, g)
            if r is not None:
                rows[name] = r

    # nested prefixes across budgets for the seeded / deterministic policies
    for g in args.groups:
        for p in ("random", "nbv", "fixed", "nbve", "oracle"):
            chain = [rows.get(f"A9_{p}_b{b}_g{g}") for b in sorted(args.budgets)]
            if any(r is None for r in chain):
                continue
            ok = True
            for a, b in zip(chain[:-1], chain[1:], strict=True):
                ua = {e["scene_id"]: e["views_used"] for e in a["episodes"]}
                ub = {e["scene_id"]: e["views_used"] for e in b["episodes"]}
                ok &= all(ub[s][: len(ua[s])] == ua[s] for s in ua)
            c.check(ok, f"{p} g{g}: budget rows are nested prefixes")

    # the fixed budget-4 row is the A8 k = 4 pipeline on the same images: identical per-GT errors
    for g in args.groups:
        r = rows.get(f"A9_fixed_b4_g{g}")
        a8 = manifest(outputs, "A8_k4", ds)
        if r is None or a8 is None:
            continue
        mine = pd.read_parquet(r["dirs"]["evaluate"] / "gt_rows.parquet")
        theirs = pd.read_parquet(Path(a8["stages"]["evaluate"]["dir"]) / "gt_rows.parquet")
        keys = ["scene_id", "image_id", "object_id", "gt_index"]
        merged = mine.merge(theirs, on=keys, suffixes=("_a9", "_a8"))
        same_rows = len(merged) == len(mine)
        close = same_rows and bool(
            np.allclose(merged.mssd_mm_a9.fillna(-1), merged.mssd_mm_a8.fillna(-1), atol=1e-6)
            and np.allclose(merged.ar_vsd_a9, merged.ar_vsd_a8, atol=1e-6)
        )
        c.check(
            close,
            f"fixed b4 g{g} reproduces A8_k4 per-GT errors on its images",
            f"{len(merged)} / {len(mine)} GT rows matched",
        )

    # the paired difference NBV - random is what the exit criterion is about
    curve = outputs / f"{ds}_epsilon_curve.json"
    if c.check(curve.exists(), "curve JSON exists"):
        cj = json.loads(curve.read_text())
        diff = cj.get("nbv_minus_random", {})
        c.check(
            bool(diff),
            "curve JSON holds the NBV - random paired difference",
            ", ".join(sorted(diff)),
        )
        for label, table in (("random", diff), ("fixed ", cj.get("nbv_minus_fixed", {}))):
            for k, d in sorted(table.items()):
                print(
                    f"      {k}: NBV - {label} = {100 * d['mean_diff']:+.2f} pt "
                    f"[{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}], wins {100 * d['wins']:.0f} %"
                )
    c.check((outputs / f"{ds}_epsilon_report.md").exists(), "Epsilon report exists")
    c.check((outputs / f"{ds}_epsilon_curve.png").exists(), "AR-vs-views figure exists")
    c.check(
        (REPO / "docs" / "figures" / f"epsilon_{ds}_curve.png").exists(),
        "figure copied to docs/figures",
    )

    grasp_path = REPO / "models" / "grasps" / f"{ds}.json"
    if c.check(grasp_path.exists(), "grasp file exists"):
        grasps = load_grasps(grasp_path)
        any_row = next(iter(rows.values()), None)
        if any_row is not None:
            info = any_row["manifest"]["config"]["dataset"]
            models_info = json.loads(
                (REPO / info["root"] / info["models_dir"] / "models_info.json").read_text()
            )
            c.check(
                set(grasps) == {int(k) for k in models_info},
                "one Grasp Pose per ObjectModel",
                f"{len(grasps)} grasps, {len(models_info)} models",
            )
            c.check(
                all(
                    np.allclose(
                        g.T_object_gripper[:3, :3] @ g.T_object_gripper[:3, :3].T,
                        np.eye(3),
                        atol=1e-6,
                    )
                    for g in grasps.values()
                ),
                "every T_object_gripper is rigid",
            )
    picks = list((outputs / f"{ds}_epsilon_pick").glob("pick_scene*.png"))
    c.check(bool(picks), "pick figure exists", ", ".join(p.name for p in picks))
    c.check(
        bool(list((REPO / "docs" / "figures").glob(f"epsilon_{ds}_pick*.png"))),
        "pick figure copied to docs/figures",
    )

    print(f"\n{c.failures} failure(s)")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
