# CNOS estimator image (D15). Build from the repository root:
#   docker build -f docker/cnos/Dockerfile -t binposert/cnos .
# Run (needs the NVIDIA Container Toolkit; EGL for pyrender templates needs the graphics capability):
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
#     -v $PWD:/work -w /work binposert/cnos \
#     python adapters/cnos_cli.py --dataset-root data/bop/tless --split test_primesense \
#       --targets data/bop/tless/test_targets_bop19.json --out <stage_dir> --params '{...}'
# The same pins drive the host-side alternative: tools/setup_estimator_env.sh cnos
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=egl TORCH_HOME=/work/data/checkpoints/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git wget ca-certificates \
        libgl1 libglib2.0-0 libegl1 libgles2 libglvnd0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
# Expose the host driver's EGL vendor library to GLVND inside the container.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/cnos/requirements.txt docker/cnos/upstream.env /opt/spec/
RUN python3.10 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r /opt/spec/requirements.txt
ENV PATH=/opt/venv/bin:$PATH

# Upstream checkout at the pinned commit (see upstream.env).
RUN . /opt/spec/upstream.env && git clone "$UPSTREAM_URL" /opt/cnos \
    && git -C /opt/cnos checkout "$UPSTREAM_COMMIT"

# The repository is mounted at /work (adapters/, binposert/, data/); PYTHONPATH makes both importable.
ENV PYTHONPATH=/work:/opt/cnos
WORKDIR /work
ENTRYPOINT ["python", "adapters/cnos_cli.py", "--upstream", "/opt/cnos"]
