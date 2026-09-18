#!/usr/bin/env python
"""Verify that Delta's artefacts are complete and consistent before the milestone is closed.

    uv run python tools/check_delta.py --datasets tless xyzibd --tag v1

Checks, each printed as PASS / FAIL: the labelled tables of every dataset exist and carry no NaN
label; the single-view fused table agrees with the hypotheses table; the fitted model files,
card, calibration report and figures exist and the schema versions match; the report holds the
per-dataset, cross-dataset, MLP and ablation rows; every A8 / A8w row has a run manifest whose
stages all carry ``_SUCCESS``, a scored ``fused.parquet`` without NaN, the same fuse stage as the
Gamma mean row and as many predictions (A8) or a different association with a similar number of
predictions (A8w), and an evaluate report scoring the same GT and images as the Gamma row; the
Delta reports, galleries and the update-path profile exist; no row log contains a traceback.
Exit status 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binposert.confidence import ConfidenceModel, VerdictThresholds  # noqa: E402
from binposert.confidence.schema import SCHEMA_VERSION, SIGNALS  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datasets", nargs="+", default=["tless", "xyzibd"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    args = ap.parse_args()
    c = Checker()
    out = args.outputs

    # labelled tables
    for ds in args.datasets:
        d = out / "confidence" / ds
        ok = all(
            (d / f).exists()
            for f in ("build.json", "hypotheses.parquet", "fused.parquet", "members.parquet")
        )
        if not c.check(ok, f"{ds}: labelled tables"):
            continue
        h = pd.read_parquet(d / "hypotheses.parquet")
        f = pd.read_parquet(d / "fused.parquet")
        m = pd.read_parquet(d / "members.parquet")
        c.check(
            len(h) > 1000 and len(f) > len(h),
            f"{ds}: table sizes",
            f"{len(h)} hyps, {len(f)} fused",
        )
        c.check(
            h["success"].notna().all()
            and f["success"].notna().all()
            and h["mssd_mm"].notna().all(),
            f"{ds}: labels carry no NaN",
        )
        c.check(
            0.1 < h["success"].mean() < 0.9,
            f"{ds}: both classes present",
            f"{h.success.mean():.3f}",
        )
        k1 = f[f.k == 1]
        c.check(
            len(k1) == len(h) and abs(k1["success"].mean() - h["success"].mean()) < 1e-9,
            f"{ds}: k=1 fused table equals the hypotheses table",
        )
        keys = ["scene_id", "image_id", "object_id", "detection_id", "hypothesis_id"]
        joined = m.merge(h[keys], on=keys, how="left", indicator=True)
        c.check((joined["_merge"] == "both").all(), f"{ds}: every member links to a hypothesis")

    # models and calibration report
    md = REPO / "models" / "confidence" / args.tag
    rd = out / "confidence" / args.tag
    ok = all(
        (md / f).exists() for f in ("model_h.json", "model_f.json", "thresholds.json", "card.md")
    )
    if c.check(ok, f"models/confidence/{args.tag}: model files + card"):
        mh, mf = (
            ConfidenceModel.load(md / "model_h.json"),
            ConfidenceModel.load(md / "model_f.json"),
        )
        c.check(
            mh.schema.version == SCHEMA_VERSION and mf.schema.version == SCHEMA_VERSION,
            "model schema versions are current",
        )
        c.check(mh.schema.kind == "hypothesis" and mf.schema.kind == "fused", "model kinds")
        thr = VerdictThresholds.load(md / "thresholds.json")
        c.check(
            thr.tau_reject < thr.tau_accept,
            "thresholds ordered",
            f"{thr.tau_reject:.3f} < {thr.tau_accept:.3f}",
        )
    ok = (rd / "report.json").exists() and (rd / "report.md").exists()
    if c.check(ok, f"outputs/confidence/{args.tag}: calibration report"):
        rep = json.loads((rd / "report.json").read_text())
        for fig in ("reliability_h", "reliability_f", "risk_coverage", "risk_coverage_k"):
            c.check(Path(rep["figures"][fig]).exists(), f"figure {fig}")
        for model in ("h", "f"):
            ev = rep[f"{model}_eval"]
            c.check(
                "pooled" in ev and all(f"dataset={ds}" in ev for ds in args.datasets),
                f"Model {model.upper()}: per-dataset eval rows",
            )
            c.check(
                ev["pooled"]["roc_auc"] > 0.7,
                f"Model {model.upper()}: eval ROC-AUC > 0.7",
                f"{ev['pooled']['roc_auc']:.3f}",
            )
            c.check(
                ev["pooled"]["ece"] < 0.1,
                f"Model {model.upper()}: eval ECE < 10 %",
                f"{100 * ev['pooled']['ece']:.1f} %",
            )
        c.check(len(rep["cross"]) == len(args.datasets), "cross-dataset rows")
        c.check("decision" in rep["mlp"], "MLP comparator decision recorded")
        c.check(
            {r["signal"] for r in rep["ablation_h"]} == set(SIGNALS)
            and {r["signal"] for r in rep["ablation_f"]} == set(SIGNALS),
            "ablate-one-signal covers every signal",
        )
        v = rep["verdicts"]["eval pooled"]
        c.check(
            abs(v["accept_rate"] + v["reject_rate"] + v["request_view_rate"] - 1) < 1e-9,
            "verdict rates sum to one",
        )

    # A8 rows
    for ds in args.datasets:
        for k in args.ks:
            base = manifest(out, f"A6_k{k}_mean", ds)
            for name in [f"A8_k{k}"] + ([f"A8w_k{k}"] if k > 1 else []):
                m = manifest(out, name, ds)
                if not c.check(m is not None, f"{name} {ds}: run manifest"):
                    continue
                assert m is not None
                dirs = {s: Path(v["dir"]) for s, v in m["stages"].items()}
                c.check(
                    all((d / "_SUCCESS").exists() for d in dirs.values()),
                    f"{name} {ds}: every stage has _SUCCESS",
                )
                cd = dirs["confidence"]
                fused = pd.read_parquet(cd / "fused.parquet")
                hyps = pd.read_parquet(cd / "hypotheses.parquet")
                c.check(
                    len(fused) > 0
                    and fused["confidence"].between(0, 1).all()
                    and fused["verdict"].notna().all(),
                    f"{name} {ds}: every FusedPose has Confidence + Verdict",
                    f"{len(fused)} rows",
                )
                c.check(
                    hyps["confidence"].notna().all(),
                    f"{name} {ds}: projected hypotheses carry Confidence",
                )
                rep = json.loads((dirs["evaluate"] / "report.json").read_text())
                if base is not None:
                    brep = json.loads(
                        (Path(base["stages"]["evaluate"]["dir"]) / "report.json").read_text()
                    )
                    c.check(
                        rep["n_gt"] == brep["n_gt"] and rep["n_images"] == brep["n_images"],
                        f"{name} {ds}: scores the same GT / images as A6_k{k}_mean",
                    )
                    if name.startswith("A8_"):
                        # the same fused poses, only ranked by Confidence
                        c.check(
                            m["stages"]["fuse"]["hash"] == base["stages"]["fuse"]["hash"]
                            and rep["n_predictions"] == brep["n_predictions"],
                            f"{name} {ds}: shares the fuse stage / predictions with A6_k{k}_mean",
                        )
                    else:
                        # Model-H weights re-associate: a different, similar-sized track set
                        c.check(
                            m["stages"]["associate"]["hash"] != base["stages"]["associate"]["hash"]
                            and abs(rep["n_predictions"] - brep["n_predictions"])
                            <= 0.02 * brep["n_predictions"],
                            f"{name} {ds}: Model-H weights change the association (± 2 % preds)",
                        )
                log = out / "runs" / f"{name}_{ds}.log"
                c.check(
                    log.exists() and "Traceback" not in log.read_text(),
                    f"{name} {ds}: log without traceback",
                )
        c.check((out / f"{ds}_delta_report.md").exists(), f"{ds}: Delta report")
        for k in (1, max(args.ks)):
            gal = out / f"{ds}_delta_gallery_A8_k{k}" / "confident_failures.png"
            c.check(gal.exists(), f"{ds}: confident-failure gallery k={k}")
    c.check((out / "tless_update_path_profile.json").exists(), "update-path profile")
    print(f"\n{c.failures} failure(s)")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
