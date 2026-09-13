# CNOS — https://github.com/nv-nguyen/cnos, pinned commit below. Only needed for
# segmenter.variant=run; the default variant imports the official BOP'23 detections without a GPU.
# Build from the repository root:  docker build -t binposert/cnos -f docker/cnos/Dockerfile .
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG CNOS_COMMIT=298d1f3366171464ca271659f0e2f7a6eb8e39b4
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgl1 libglib2.0-0 libegl1 libosmesa6 \
        libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/nv-nguyen/cnos /opt/cnos && cd /opt/cnos && git checkout ${CNOS_COMMIT}
WORKDIR /opt/cnos
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true \
    && conda env create -f environment.yml && conda clean -afy
SHELL ["conda", "run", "-n", "cnos", "/bin/bash", "-c"]
RUN pip install git+https://github.com/facebookresearch/segment-anything.git ultralytics==8.0.135
# Weights (SAM / FastSAM / DINOv2) are fetched by CNOS's own download scripts into the scratch mount:
#   python -m src.scripts.download_sam && python -m src.scripts.download_fastsam

COPY binposert /workspace/binposert
COPY adapters /workspace/adapters
# The core is used straight from /workspace via PYTHONPATH (its metadata says python>=3.10,
# the estimator envs are 3.9; binposert.artefacts / binposert.types themselves run on 3.9).
RUN pip install pandas pyarrow imageio

ENV CNOS_REPO=/opt/cnos CNOS_SCRATCH=/scratch/cnos PYTHONPATH=/opt/cnos:/workspace
WORKDIR /workspace
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cnos"]
CMD ["python", "adapters/cnos_cli.py", "--help"]
