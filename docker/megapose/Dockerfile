# MegaPose estimator image (D15). Build from the repository root:
#   docker build -f docker/megapose/Dockerfile -t binposert/megapose .
# Run (panda3d renders through EGL -> graphics capability):
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
#     -v $PWD:/work binposert/megapose --dataset-root data/bop/tless --split test_primesense \
#       --targets data/bop/tless/test_targets_bop19.json --detections <segment_dir> \
#       --out <stage_dir> --params '{...}'
# Host-side alternative with the same pins: tools/setup_estimator_env.sh megapose
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MEGAPOSE_DATA_DIR=/work/data/checkpoints/megapose
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git wget ca-certificates \
        libgl1 libglib2.0-0 libegl1 libgles2 libglvnd0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/megapose/requirements.txt docker/megapose/upstream.env docker/megapose/patch.sh /opt/spec/
RUN python3.10 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r /opt/spec/requirements.txt
ENV PATH=/opt/venv/bin:$PATH

RUN . /opt/spec/upstream.env && git clone "$UPSTREAM_URL" /opt/megapose \
    && git -C /opt/megapose checkout "$UPSTREAM_COMMIT" && bash /opt/spec/patch.sh /opt/megapose

ENV PYTHONPATH=/work:/opt/megapose/src
WORKDIR /work
ENTRYPOINT ["python", "adapters/megapose_cli.py", "--upstream", "/opt/megapose"]
