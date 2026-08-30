#!/bin/bash
# Rebuild the vllm-xpu-kernels wheel from the patched tree at /root/build/vxk
# (which now carries the GDN spec ragged-batch fix), replicating
# vllm/docker/Dockerfile stage 1c inside the omix devel builder.
# Uses uv (like the Dockerfile) since the raw container lacks python3-venv.
set -eo pipefail

WHEELS=/root/build/wheels
rm -rf "$WHEELS"
mkdir -p "$WHEELS"

docker run --rm \
  -v /root/build/vxk:/src/vllm-xpu-kernels \
  -v "$WHEELS":/wheels \
  -e KERNELS_MAX_JOBS=16 \
  intel/omix:0.1.0-devel-ubuntu24.04 \
  bash -c '
    set -exo pipefail
    source /opt/intel/oneapi/setvars.sh --force
    apt-get update -y && apt-get install -y --no-install-recommends \
        git curl ca-certificates python3-dev ninja-build cmake pkg-config numactl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    uv venv /opt/venv
    export VIRTUAL_ENV=/opt/venv
    export PATH=/opt/venv/bin:$PATH
    uv pip install pip wheel setuptools
    uv pip install torch==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu
    cd /src/vllm-xpu-kernels
    rm -rf build vllm_xpu_kernels.egg-info dist
    uv pip install -r requirements.txt
    export CMAKE_PREFIX_PATH="$(python3 -c "import site; print(site.getsitepackages()[0])"):${CMAKE_PREFIX_PATH:-}"
    MAX_JOBS=$KERNELS_MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$KERNELS_MAX_JOBS \
      pip wheel --no-build-isolation --no-deps . -w /wheels
    ls -la /wheels/
  '

echo "BUILD-OK: $(ls "$WHEELS")"
