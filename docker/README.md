# Estimator containers (D15)

One image per external estimator, each pinning an upstream commit (see `configs/segmenter/cnos.yaml`,
`configs/estimator/{foundpose,megapose}.yaml` — the Dockerfile `ARG` must match). The core package is
installed into every image with `pip install --no-deps` so the adapters can write the frozen artefact
schema (`binposert/artefacts.py`) without dragging Open3D / Hydra into the estimator environments.

These images are built and smoke-tested on the GPU machine as the first task of Alpha; nothing here
runs on the laptop.

```bash
# on the GPU machine, from the repository root
docker build -t binposert/foundpose -f docker/foundpose/Dockerfile .
docker build -t binposert/megapose  -f docker/megapose/Dockerfile .
docker build -t binposert/cnos      -f docker/cnos/Dockerfile .      # only for segmenter.variant=run
```

## Handshake with `tools/run.py`

1. `uv run python tools/run.py experiment=A0 dataset=tless` runs the local stages, then stops at the
   first GPU stage whose output is not cached and writes
   `outputs/tless/test_primesense/<stage>/<hash>/adapter_request.json`.
2. Run the named adapter inside the matching container with that request. The adapter fills the
   directory next to the request and writes `_SUCCESS` last.
3. `rsync -a outputs/ laptop:BinPoseRT/outputs/` and re-run the same `tools/run.py` command.

```bash
BOP=/data/bop            # parent of tless/, xyzibd/, …
REPO=$PWD
docker run --gpus all --rm -it \
  -v $BOP:/data/bop -v $REPO:/workspace -v /scratch/binposert:/scratch \
  -e BOP_PATH=/data/bop -e BINPOSERT_DATASET_ROOT=/data/bop/tless \
  binposert/foundpose \
  python adapters/foundpose_cli.py run --request outputs/tless/test_primesense/coarse_pose/<hash>/adapter_request.json
```

`BINPOSERT_DATASET_ROOT` overrides the dataset path recorded in the request (the laptop's
`data/bop/tless`) with the container's mount. `/scratch` keeps templates / object representations /
checkpoints between runs.

## A1 without a GPU

`segmenter.variant=bop23_default` (the default) imports the official BOP'23 CNOS-FastSAM detections
instead of running CNOS; `adapters/cnos_cli.py run --request …` works on the laptop in the core `uv`
environment and downloads the zip on first use.

## Official evaluation

`tools/bop_eval.sh <results.csv>` runs `bop_toolkit` in its own environment on any BOP CSV under
`outputs/…/runs/` (D15). Core metrics and toolkit numbers must agree within tolerance or the
discrepancy is explained in the run directory.
