#!/bin/bash
# v88b — apply the int64 kernel fix to /root/build/vxk and rebuild the
# vllm-xpu-kernels wheel with KERNELS_MAX_JOBS=52 (user directive; the old
# v26 script used 16). Backup .v88bak next to each patched header.
set -u
TS=$(date +%H%M%S)
L=/root/build/lce1/v88b_$TS.out
WHEELS=/root/build/wheels_v88
echo "v88b fix+build start $TS" | tee -a $L

# 1. apply the kernel patch (idempotent, self-verifying)
python3 /root/build/patch_v88_int64fix.py 2>&1 | tee -a $L
grep -c 'V88FIX' /root/build/vxk/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp /root/build/vxk/csrc/xpu/gdn_attn/causal_conv1d.hpp | xargs echo marks: | tee -a $L

# 2. rebuild the wheel in the omix devel builder, jobs=52
rm -rf "$WHEELS"; mkdir -p "$WHEELS"
docker run --rm \
  -v /root/build/vxk:/src/vllm-xpu-kernels \
  -v "$WHEELS":/wheels \
  -e KERNELS_MAX_JOBS=52 \
  intel/omix:0.1.0-devel-ubuntu24.04 \
  bash -c '
    set -eo pipefail
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
  ' 2>&1 | tail -40 | tee -a $L

ls "$WHEELS"/*.whl >/dev/null 2>&1 && echo "BUILD-OK: $(ls $WHEELS)" | tee -a $L || { echo BUILD_FAILED | tee -a $L; exit 1; }
echo V88B_DONE | tee -a $L
