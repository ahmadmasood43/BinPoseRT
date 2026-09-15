# Estimator environments (D15)

One directory per external estimator. Each holds the *single source of truth* for that estimator's
environment, used both by the Docker image and by the host-side venv:

| file | purpose |
|---|---|
| `Dockerfile` | CUDA 11.8 runtime + Python 3.10 venv + pinned upstream checkout |
| `requirements.txt` | exact pip pins (torch 2.5.1+cu118 — Pascal-capable) |
| `upstream.env` | upstream repository URL and commit; the stage `version` in `configs/` follows it |
| `patch.sh` | minimal, idempotent source patches (only where unavoidable) |

The adapters in `adapters/` are the only code that runs inside these environments. They import
`binposert.types` and `binposert.pipeline.artefacts` from the mounted repository (numpy / pandas /
imageio only) and write D12 artefacts.

## Docker

```bash
docker build -f docker/cnos/Dockerfile -t binposert/cnos .
PARAMS=$(uv run python -c "import json,yaml;print(json.dumps(yaml.safe_load(open('configs/segmenter/cnos.yaml'))['params']))")
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$PWD":/work binposert/cnos --dataset-root data/bop/tless --split test_primesense \
  --targets data/bop/tless/test_targets_bop19.json \
  --out outputs/tless/test_primesense/segment/<hash> --params "$PARAMS"
```

`pyrender` (CNOS templates, FoundPose templates) and `panda3d` (MegaPose) render through EGL; the
container therefore needs the NVIDIA Container Toolkit with the `graphics` driver capability.

## Host venv (no container toolkit)

```bash
tools/setup_estimator_env.sh cnos        # envs/cnos/.venv + third_party/cnos @ pinned commit
tools/setup_estimator_env.sh foundpose
```

`configs/segmenter/cnos.yaml` and `configs/estimator/*.yaml` carry the host command lines, so on a
GPU machine the whole row runs as one command and the stage hash directory is chosen for you:

```bash
uv run python tools/run.py experiment=A1 dataset=tless run_external=true
```

## Checkpoints

Downloaded once into `data/checkpoints/` (git-ignored):

| file | used by | source |
|---|---|---|
| `sam_vit_h_4b8939.pth` | CNOS | https://dl.fbaipublicfiles.com/segment_anything/ |
| `dinov2_vitl14_pretrain.pth` | CNOS, FoundPose | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/ |
| `megapose-models/` | MegaPose | https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/ |

The adapters symlink them into `$TORCH_HOME/hub/checkpoints` so nothing is downloaded at run time.
