#!/usr/bin/env python
"""CNOS adapter: fills a ``segment`` stage directory with a Detections artefact (D4, D12).

Two variants, chosen by ``segmenter.variant`` in the request:

``bop23_default``  (no GPU) — import the official BOP'23 default detections
    (``cnos-fastsam_<dataset>-test.json``, the file FoundPose and most zero-shot methods use). The
    JSON is looked up under ``<dataset_root>/detections/cnos-fastsam/``, or downloaded from
    ``upstream.detections_url`` when missing, or given explicitly with ``--detections-json``.

``run``  (GPU, inside docker/cnos/) — run the pinned CNOS repo's ``run_inference.py`` and import the
    JSON it writes. The output format of both variants is identical (BOP/COCO RLE).

Usage::

    python adapters/cnos_cli.py run --request <stage dir>/adapter_request.json
    python adapters/cnos_cli.py import --json cnos-fastsam_tless-test.json --request …
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters._common import (  # noqa: E402
    Request,
    clean_output_dir,
    read_json,
    run_command,
    setup_logging,
)
from binposert import artefacts  # noqa: E402

log = logging.getLogger("adapter.cnos")

CNOS_REPO = Path(os.environ.get("CNOS_REPO", "/opt/cnos"))


# ----------------------------------------------------------------------------- import


def import_bop_detections(req: Request, json_path: Path, source: str) -> int:
    """Convert a BOP-format detection JSON into the Detections artefact for the request's images."""
    raw: list[dict[str, Any]] = read_json(json_path)
    wanted = set(req.image_keys)
    per_image: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for d in raw:
        key = (int(d["scene_id"]), int(d["image_id"]))
        if key not in wanted:
            continue
        objs = req.object_ids(*key)
        if objs is not None and int(d["category_id"]) not in objs:
            continue
        per_image[key].append(d)

    writer = artefacts.DetectionsWriter(req.output_dir, source=source)
    n = 0
    for key in req.image_keys:
        dets = sorted(
            per_image.get(key, []), key=lambda d: -float(d["score"])
        )  # stable: ties keep order
        for det_id, d in enumerate(dets):
            mask = artefacts.rle_to_mask(d["segmentation"])
            writer.add_mask(
                scene_id=key[0],
                image_id=key[1],
                object_id=int(d["category_id"]),
                detection_id=det_id,
                mask=mask,
                score=float(d["score"]),
                time_s=float(d.get("time", float("nan"))),
            )
            n += 1
    writer.finish(
        stage="segment",
        config=req.config,
        hash=req.hash,
        n_images=len(req.image_keys),
        detections_json=str(json_path),
        upstream=req.config.get("upstream"),
    )
    return n


def locate_default_detections(req: Request, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    name = f"cnos-fastsam_{req.dataset_name}-test.json"
    det_dir = req.dataset_root / "detections" / "cnos-fastsam"
    path = det_dir / name
    if path.exists():
        return path
    url = str(req.config.get("upstream", {}).get("detections_url", ""))
    if not url:
        raise FileNotFoundError(f"{path} missing and no upstream.detections_url to fetch it from")
    zip_path = req.dataset_root / "detections" / Path(url).name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        # inside the BOP'23 zip the files carry a UUID suffix (cnos-fastsam_tless-test_<uuid>.json)
        # next to macOS "._" resource forks; take the real one and store it under the plain name
        stem = name[: -len(".json")]
        members = [
            m
            for m in z.namelist()
            if Path(m).name.startswith(stem)
            and m.endswith(".json")
            and not Path(m).name.startswith("._")
            and "__MACOSX" not in m
        ]
        if not members:
            raise FileNotFoundError(f"{stem}*.json not inside {zip_path}")
        det_dir.mkdir(parents=True, exist_ok=True)
        with z.open(members[0]) as src, open(path, "wb") as dst:
            dst.write(src.read())
    return path


# ----------------------------------------------------------------------------- run upstream


def run_cnos(req: Request) -> Path:
    """Run the pinned CNOS repo on the dataset and return the JSON it produced.

    CNOS expects ``<root_dir>/datasets/<dataset>`` (configs/user/default.yaml) and writes to
    ``<root_dir>/results/<name_exp>/``; we point ``root_dir`` at a scratch directory whose
    ``datasets/<name>`` is a symlink to our BOP root, and rendered templates are reused across runs.
    """
    run_cfg = req.config.get("run", {})
    model = str(run_cfg.get("model", "cnos_fast"))
    rendering = str(run_cfg.get("rendering_type", "pyrender"))
    level = int(run_cfg.get("level_templates", 0))
    scratch = Path(os.environ.get("CNOS_SCRATCH", "/scratch/cnos"))
    (scratch / "datasets").mkdir(parents=True, exist_ok=True)
    link = scratch / "datasets" / req.dataset_name
    if not link.exists():
        link.symlink_to(req.dataset_root.resolve())
    templates = scratch / "datasets" / "templates_pyrender" / req.dataset_name
    common = [f"dataset_name={req.dataset_name}", f"user.local_root_dir={scratch}"]
    if not templates.exists():
        run_command(
            [sys.executable, "-m", "src.scripts.render_template_with_pyrender", *common],
            cwd=CNOS_REPO,
        )
    name_exp = f"binposert_{req.hash}"
    run_command(
        [
            sys.executable,
            "run_inference.py",
            *common,
            f"model={model}",
            f"model.onboarding_config.rendering_type={rendering}",
            f"model.onboarding_config.level_templates={level}",
            f"name_exp={name_exp}",
        ],
        cwd=CNOS_REPO,
    )
    results = scratch / "results" / name_exp
    candidates = sorted(p for p in results.glob("*.json") if "score_distribution" not in p.name)
    if not candidates:
        raise FileNotFoundError(f"CNOS wrote no prediction JSON under {results}")
    return candidates[-1]


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="fill the stage directory according to the request")
    p_run.add_argument("--request", required=True, type=Path)
    p_imp = sub.add_parser("import", help="import a given BOP-format detection JSON")
    p_imp.add_argument("--request", required=True, type=Path)
    p_imp.add_argument("--json", required=True, type=Path)
    args = ap.parse_args(argv)

    req = Request.load(args.request)
    if req.stage != "segment":
        raise SystemExit(f"request is for stage {req.stage!r}, this adapter fills 'segment'")
    clean_output_dir(req)
    variant = str(req.config.get("variant", "bop23_default"))
    if args.cmd == "import":
        json_path, source = Path(args.json), "cnos-fastsam"
    elif variant == "bop23_default":
        json_path, source = locate_default_detections(req, None), "cnos-fastsam"
    elif variant == "run":
        json_path = run_cnos(req)
        source = "cnos-" + str(req.config.get("run", {}).get("model", "cnos_fast"))
    else:
        raise SystemExit(f"unknown segmenter.variant {variant!r}")
    n = import_bop_detections(req, json_path, source)
    artefacts.mark_success(req.output_dir)
    log.info("wrote %d detections for %d images → %s", n, len(req.image_keys), req.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
