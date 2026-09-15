# FoundPose estimator image (D15). Build from the repository root:
#   docker build -f docker/foundpose/Dockerfile -t binposert/foundpose .
# Run:
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
#     -v $PWD:/work -w /work binposert/foundpose \
#     python adapters/foundpose_cli.py --dataset-root data/bop/tless --split test_primesense \
#       --targets data/bop/tless/test_targets_bop19.json --detections <segment_dir> \
#       --out <stage_dir> --params '{...}'
# Host-side alternative with the same pins: tools/setup_estimator_env.sh foundpose
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=egl TORCH_HOME=/work/data/checkpoints/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git wget ca-certificates \
        libgl1 libglib2.0-0 libegl1 libgles2 libglvnd0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/foundpose/requirements.txt docker/foundpose/upstream.env docker/foundpose/patch.sh /opt/spec/
RUN python3.10 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r /opt/spec/requirements.txt
ENV PATH=/opt/venv/bin:$PATH

RUN . /opt/spec/upstream.env && git clone --recurse-submodules "$UPSTREAM_URL" /opt/foundpose \
    && git -C /opt/foundpose checkout "$UPSTREAM_COMMIT" \
    && git -C /opt/foundpose submodule update --init --recursive \
    && bash /opt/spec/patch.sh /opt/foundpose

ENV PYTHONPATH=/work:/opt/foundpose:/opt/foundpose/external/bop_toolkit:/opt/foundpose/external/dinov2
WORKDIR /work
ENTRYPOINT ["python", "adapters/foundpose_cli.py", "--upstream", "/opt/foundpose"]
