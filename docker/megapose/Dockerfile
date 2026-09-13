# MegaPose — https://github.com/megapose6d/megapose6d, pinned commit below.
# Build from the repository root:  docker build -t binposert/megapose -f docker/megapose/Dockerfile .
# Checkpoints: run once with the data dir mounted:
#   python -m megapose.scripts.download --megapose_models   (→ $MEGAPOSE_DATA_DIR/megapose-models)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG MEGAPOSE_COMMIT=f3b8e1247f133f3d098833a251b8f2d744c03e1f
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgl1 libglib2.0-0 libegl1 libosmesa6 \
        libxrender1 libxext6 xvfb && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/megapose6d/megapose6d /opt/megapose \
    && cd /opt/megapose && git checkout ${MEGAPOSE_COMMIT}
WORKDIR /opt/megapose
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true \
    && conda env create -f conda/environment_full.yaml && conda clean -afy
SHELL ["conda", "run", "-n", "megapose", "/bin/bash", "-c"]
RUN pip install -e .

COPY binposert /workspace/binposert
COPY adapters /workspace/adapters
# The core is used straight from /workspace via PYTHONPATH (its metadata says python>=3.10,
# the estimator envs are 3.9; binposert.artefacts / binposert.types themselves run on 3.9).
RUN pip install pandas pyarrow imageio

ENV MEGAPOSE_DATA_DIR=/scratch/megapose PYTHONPATH=/workspace
WORKDIR /workspace
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "megapose"]
CMD ["python", "adapters/megapose_cli.py", "--help"]
