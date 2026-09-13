"""Builds the stage plan from a resolved config, runs local stages, skips cached ones and stops
with instructions when a GPU artefact is missing (D12, ADR-0002)."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from binposert import artefacts
from binposert.data import BopDataset
from binposert.pipeline import manifest as mf
from binposert.pipeline.cache import StagePlan, stage_hash
from binposert.pipeline.selection import Selection
from binposert.pipeline.stages import ADAPTER_REQUEST_FILE, STAGES, StageContext

log = logging.getLogger("binposert.pipeline")

DATASET_IDENTITY_KEYS = (
    "name",
    "split",
    "models_dir",
    "targets",
    "scene_ids",
    "image_ids",
    "max_images_per_scene",
)


class MissingGpuArtefact(RuntimeError):
    """A GPU stage's output directory is not there yet. ``.plan`` tells which one."""

    def __init__(self, plan: StagePlan, message: str) -> None:
        super().__init__(message)
        self.plan = plan


@dataclass
class RunResult:
    run_dir: Path
    manifest_path: Path
    stages: list[StagePlan]
    status: dict[str, str]  # stage -> ran | cached | missing | skipped
    report: dict[str, Any] | None = None  # evaluate/report.json when the run got that far
    results_csv: Path | None = None
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return all(s in ("ran", "cached") for s in self.status.values())


class Pipeline:
    def __init__(self, config: dict[str, Any], outputs_root: str | Path | None = None) -> None:
        self.config = config
        ds_cfg = config["dataset"]
        self.dataset = BopDataset(
            ds_cfg["root"], split=str(ds_cfg["split"]), models_dir=ds_cfg.get("models_dir")
        )
        self.selection = Selection.from_config(self.dataset, ds_cfg)
        root = Path(outputs_root or config.get("outputs_root", "outputs"))
        self.split_root = root / str(ds_cfg["name"]) / str(ds_cfg["split"])
        self.stage_names: list[str] = [str(s) for s in config["pipeline"]["stages"]]
        self.seed = int(config.get("seed", 0))
        self.plan: list[StagePlan] = self._build_plan()

    # ------------------------------------------------------------------ planning

    def _dataset_identity(self) -> dict[str, Any]:
        ds = self.config["dataset"]
        return {k: ds.get(k) for k in DATASET_IDENTITY_KEYS}

    def _build_plan(self) -> list[StagePlan]:
        plans: list[StagePlan] = []
        latest_producer: dict[str, StagePlan] = {}
        identity = self._dataset_identity()
        for name in self.stage_names:
            spec = STAGES[name]
            cfg = dict(self.config.get(spec.config_key, {}))
            inputs: dict[str, Path] = {}
            input_hashes: dict[str, str] = {}
            for kind in spec.consumes:
                if kind not in latest_producer:
                    raise KeyError(
                        f"stage {name!r} needs {kind!r} but no earlier stage produces it"
                    )
                up = latest_producer[kind]
                inputs[kind] = up.directory
                input_hashes[up.name] = up.hash
            h = stage_hash(name, spec.version(cfg), cfg, identity, input_hashes)
            plan = StagePlan(
                name=name,
                impl=spec.impl(cfg),
                version=spec.version(cfg),
                hash=h,
                directory=self.split_root / name / h,
                config=cfg,
                inputs=inputs,
                input_hashes=input_hashes,
                produces=spec.produces,
            )
            plans.append(plan)
            latest_producer[spec.produces] = plan
        return plans

    def describe(self) -> str:
        lines = [f"dataset {self.dataset.name}/{self.dataset.split}: {len(self.selection)} images"]
        for p in self.plan:
            state = "done" if p.is_complete else ("gpu: missing" if p.impl == "gpu" else "to run")
            lines.append(
                f"  {p.name:<12} {p.config.get('name', ''):<12} {p.hash}  [{state}]  {p.directory}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ execution

    def run(self, stop_after: str | None = None, experiment: str | None = None) -> RunResult:
        """Run every stage in order; both arguments default to ``pipeline.stop_after`` and
        ``experiment`` from the config."""
        stop_after = stop_after or self.config.get("pipeline", {}).get("stop_after")
        experiment = experiment or self.config.get("experiment")
        t_run = time.perf_counter()
        status: dict[str, str] = {}
        timings: dict[str, float] = {}
        run_dir = self._run_dir(experiment)
        failure: MissingGpuArtefact | None = None
        for plan in self.plan:
            if plan.is_complete:
                status[plan.name] = "cached"
                log.info("%s: cached at %s", plan.name, plan.directory)
            elif plan.impl == "gpu":
                self._write_adapter_request(plan)
                status[plan.name] = "missing"
                failure = MissingGpuArtefact(plan, self._gpu_instructions(plan))
                break
            else:
                t0 = time.perf_counter()
                self._run_local(plan)
                timings[plan.name] = time.perf_counter() - t0
                status[plan.name] = "ran"
            if stop_after is not None and plan.name == stop_after:
                for later in self.plan[self.plan.index(plan) + 1 :]:
                    status[later.name] = "skipped"
                break
        timings["total"] = time.perf_counter() - t_run

        result = RunResult(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            stages=self.plan,
            status=status,
            timings_s=timings,
        )
        eval_plan = next((p for p in self.plan if p.name == "evaluate"), None)
        if eval_plan is not None and status.get("evaluate") in ("ran", "cached"):
            with open(eval_plan.directory / "report.json") as f:
                result.report = json.load(f)
            result.results_csv = self._publish_results(eval_plan, run_dir, experiment)
        self._write_manifest(result, experiment)
        if failure is not None:
            raise failure
        return result

    def _run_local(self, plan: StagePlan) -> None:
        if plan.directory.exists():  # a previous attempt without _SUCCESS: start clean
            shutil.rmtree(plan.directory)
        plan.directory.mkdir(parents=True)
        for kind, d in plan.inputs.items():
            if not artefacts.is_complete(d):
                raise artefacts.ArtefactError(f"{plan.name}: input {kind} at {d} is incomplete")
        ctx = StageContext(
            dataset=self.dataset,
            selection=self.selection,
            config=plan.config,
            inputs=plan.inputs,
            out_dir=plan.directory,
            run_config=self.config,
            seed=self.seed,
            info={"hash": plan.hash, "inputs": {k: str(v) for k, v in plan.inputs.items()}},
        )
        log.info("%s: running %s → %s", plan.name, plan.config.get("name", ""), plan.directory)
        STAGES[plan.name].run(ctx)
        artefacts.mark_success(plan.directory)

    # ------------------------------------------------------------------ GPU handshake

    def _write_adapter_request(self, plan: StagePlan) -> None:
        """Leave everything an adapter needs next to where its output must land."""
        plan.directory.mkdir(parents=True, exist_ok=True)
        request = {
            "stage": plan.name,
            "hash": plan.hash,
            "version": plan.version,
            "config": plan.config,
            "dataset": {**self._dataset_identity(), "root": str(self.config["dataset"]["root"])},
            "inputs": {k: str(v) for k, v in plan.inputs.items()},
            "image_keys": [list(k) for k in self.selection.image_keys],
            "objects": (
                None
                if self.selection.objects is None
                else {f"{s}/{i}": o for (s, i), o in self.selection.objects.items()}
            ),
            "output_dir": str(plan.directory),
        }
        with open(plan.directory / ADAPTER_REQUEST_FILE, "w") as f:
            json.dump(request, f, indent=2, default=str)

    def _gpu_instructions(self, plan: StagePlan) -> str:
        adapter = {
            "segment": "cnos_cli.py",
            "coarse_pose": f"{plan.config.get('name')}_cli.py",
        }.get(plan.name, f"{plan.config.get('name')}_cli.py")
        return (
            f"stage {plan.name!r} ({plan.config.get('name')}) is a GPU stage and its output is not "
            f"cached.\n  expected: {plan.directory}\n"
            f"  request:  {plan.directory / ADAPTER_REQUEST_FILE}\n"
            f"On the GPU machine run (inside docker/{plan.config.get('name')}/):\n"
            f"  python adapters/{adapter} run --request {plan.directory / ADAPTER_REQUEST_FILE}\n"
            f"then rsync outputs/ back and re-run this command."
        )

    # ------------------------------------------------------------------ run directory / manifest

    def _run_dir(self, experiment: str | None) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.split_root / "runs" / (experiment or "adhoc") / stamp

    def _publish_results(self, eval_plan: StagePlan, run_dir: Path, experiment: str | None) -> Path:
        """Copy the BOP CSV under the bop_toolkit naming rule ``<method>_<dataset>-<split>.csv``
        (no underscores inside ``method``) next to the manifest."""
        run_dir.mkdir(parents=True, exist_ok=True)
        est = str(self.config.get("estimator", {}).get("name", ""))
        seg = str(self.config.get("segmenter", {}).get("name", ""))
        method = "-".join(x for x in (experiment or "adhoc", seg, est) if x).replace("_", "-")
        split = self.dataset.split.split("_")[0]
        dst = run_dir / f"{method}_{self.dataset.name}-{split}.csv"
        shutil.copy(eval_plan.directory / "results.csv", dst)
        shutil.copy(eval_plan.directory / "report.json", run_dir / "report.json")
        shutil.copy(eval_plan.directory / "report.md", run_dir / "report.md")
        return dst

    def _write_manifest(self, result: RunResult, experiment: str | None) -> None:
        manifest = {
            "experiment": experiment,
            "command": " ".join(sys.argv),
            "git": mf.git_info(),
            "config": self.config,
            "config_hash": mf.config_hash(self.config),
            "dataset": mf.dataset_fingerprint(self.dataset, list(self.selection.image_keys)),
            "seed": self.seed,
            "hardware": mf.hardware_info(),
            "packages": mf.package_versions(),
            "stages": [
                {
                    "name": p.name,
                    "impl": p.impl,
                    "version": p.version,
                    "hash": p.hash,
                    "directory": str(p.directory),
                    "inputs": {k: str(v) for k, v in p.inputs.items()},
                    "checkpoints": p.config.get("checkpoints"),
                    "status": result.status.get(p.name, "not reached"),
                    "seconds": result.timings_s.get(p.name),
                }
                for p in self.plan
            ],
            "timings_s": result.timings_s,
            "peak_rss_mb": mf.peak_rss_mb(),
            "results_csv": str(result.results_csv) if result.results_csv else None,
            "summary": (
                {k: result.report[k] for k in ("ar", "ar_vsd", "ar_mssd", "ar_mspd", "n_gt")}
                if result.report
                else None
            ),
        }
        mf.write_manifest(result.manifest_path, manifest)
