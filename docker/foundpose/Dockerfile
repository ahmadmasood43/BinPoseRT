# FoundPose (coarse pipeline) — https://github.com/facebookresearch/foundpose, pinned commit below.
# Build from the repository root:  docker build -t binposert/foundpose -f docker/foundpose/Dockerfile .
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG FOUNDPOSE_COMMIT=3103473bfa3985fdf563bb31cefc0cae8f979ae0
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgl1 libglib2.0-0 libegl1 libosmesa6 \
        libxrender1 libxext6 freeglut3-dev && rm -rf /var/lib/apt/lists/*

# conda (miniforge) — FoundPose ships a conda environment file.
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

RUN git clone --recurse-submodules https://github.com/facebookresearch/foundpose /opt/foundpose \
    && cd /opt/foundpose && git checkout ${FOUNDPOSE_COMMIT} && git submodule update --init --recursive
WORKDIR /opt/foundpose
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true \
    && conda env create -f conda_foundpose_gpu.yaml && conda clean -afy
SHELL ["conda", "run", "-n", "foundpose_gpu", "/bin/bash", "-c"]

# bop_toolkit paths come from the environment instead of editing config.py by hand (README step 3).
RUN python - <<'PY'
import re, pathlib
p = pathlib.Path("external/bop_toolkit/bop_toolkit_lib/config.py")
s = p.read_text()
if "import os" not in s:
    s = "import os\n" + s
s = re.sub(r"^datasets_path = .*$", "datasets_path = os.environ.get('BOP_PATH', '/data/bop')", s, flags=re.M)
s = re.sub(r"^output_path = .*$", "output_path = os.environ.get('FOUNDPOSE_OUTPUT_PATH', '/scratch/foundpose')", s, flags=re.M)
p.write_text(s)
PY

# The core package, dependency-free, for adapters/*.py (pandas/pyarrow/imageio are all it needs).
COPY binposert /workspace/binposert
COPY adapters /workspace/adapters
# The core is used straight from /workspace via PYTHONPATH (its metadata says python>=3.10,
# the estimator envs are 3.9; binposert.artefacts / binposert.types themselves run on 3.9).
RUN pip install pandas pyarrow imageio

ENV REPO_PATH=/opt/foundpose \
    BOP_PATH=/data/bop \
    FOUNDPOSE_REPO=/opt/foundpose \
    FOUNDPOSE_OUTPUT_PATH=/scratch/foundpose \
    PYTHONPATH=/opt/foundpose:/opt/foundpose/external/bop_toolkit:/opt/foundpose/external/dinov2:/workspace
WORKDIR /workspace
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "foundpose_gpu"]
CMD ["python", "adapters/foundpose_cli.py", "--help"]
