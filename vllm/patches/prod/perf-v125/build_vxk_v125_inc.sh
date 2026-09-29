#!/bin/bash
# build_vxk_v125_inc.sh — v125 P22B incremental _xpu_C rebuild after the
# fp8-native DISPATCH_STATE_DTYPE patch (p22b_kernel_patch.py, markers
# 'v125 P22B', applied in place to /root/build/vxk — no source cp here).
# Reuses the persisted cmake build dir in the warm vxk-inc builder, then
# swaps the relinked _xpu_C.abi3.so into lsv-test (backup .v125bak).
# Verification beyond the v78 pattern: the compiled reject-message literal
# must name both fp8 dtypes (strings on the .so) — binary-level proof the
# patched dispatch is in the artifact.
set -uo pipefail

grep -q 'v125 P22B' /root/build/vxk/csrc/xpu/gdn_attn/gated_delta_rule.hpp \
  || { echo ABORT_PATCH_NOT_PRESENT; exit 1; }
echo "patch marker present in source tree"

docker inspect vxk-inc >/dev/null 2>&1 || docker run -d --name vxk-inc \
  -v /root/build/vxk:/src/vllm-xpu-kernels \
  intel/omix:0.1.0-devel-ubuntu24.04 sleep infinity >/dev/null
docker start vxk-inc >/dev/null 2>&1 || true
echo "builder up"

# env identical to build_vxk_v78_inc.sh (torch==2.11.0+xpu at the exact
# paths build.ninja recorded; guarded steps are no-ops when already done)
docker exec vxk-inc bash -c '
  set -eo pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq --no-install-recommends curl ca-certificates \
    ninja-build python3-dev pkg-config numactl patchelf >/dev/null 2>&1
  export PATH=/root/.local/bin:$PATH
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH=/root/.local/bin:$PATH
  [ -x /opt/venv/bin/python ] || uv venv /opt/venv >/dev/null
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  if ! /opt/venv/bin/python -c "import torch" >/dev/null 2>&1; then
    uv pip install -q pip wheel setuptools
    uv pip install -q torch==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu
  fi
  /opt/venv/bin/python -c "import cmake" >/dev/null 2>&1 || \
    uv pip install -q -r /src/vllm-xpu-kernels/requirements.txt
  /opt/venv/bin/python -c "import torch; print(\"ENV_READY torch\", torch.__version__)"
'

# incremental ninja (KERNELS_MAX_JOBS=52 standing directive): the header
# edit recompiles every TU including gated_delta_rule, then relinks.
docker exec vxk-inc bash -c '
  set -eo pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  cd /src/vllm-xpu-kernels/build/temp
  ninja -j 52 _xpu_C 2>&1 | tail -8
  echo NINJA_RC=$?
' || { echo NINJA_FAILED; exit 3; }

SO=/root/build/vxk/build/temp/_xpu_C.abi3.so
CSO=/src/vllm-xpu-kernels/build/temp/_xpu_C.abi3.so
[ -f "$SO" ] || { echo NO_SO_FOUND; exit 4; }

docker exec vxk-inc patchelf --set-rpath '$ORIGIN' "$CSO"
docker exec vxk-inc readelf -d "$CSO" | grep RUNPATH

# v125-specific: the patched dispatch's reject message is a compiled string
# literal — both fp8 names present = the patched header is in the binary.
STR_E4M3=$(docker exec vxk-inc bash -c "strings '$CSO' | grep -c 'fp8_e4m3fn'" || true)
STR_E5M2=$(docker exec vxk-inc bash -c "strings '$CSO' | grep -c 'fp8_e5m2'" || true)
echo "so_string_hits e4m3=$STR_E4M3 e5m2=$STR_E5M2 (need >=1 each)"
[ "$STR_E4M3" -ge 1 ] 2>/dev/null || { echo ABORT_SO_MISSING_E4M3_LITERAL; exit 5; }
[ "$STR_E5M2" -ge 1 ] 2>/dev/null || { echo ABORT_SO_MISSING_E5M2_LITERAL; exit 5; }

# swap into lsv-test (backup current first)
SP=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
docker exec lsv-test cp -a $SP/_xpu_C.abi3.so $SP/_xpu_C.abi3.so.v125bak 2>/dev/null || true
docker cp "$SO" lsv-test:$SP/_xpu_C.abi3.so
docker exec lsv-test /opt/venv/bin/python -c "
import os
import torch
import vllm_xpu_kernels
so = os.path.join(os.path.dirname(vllm_xpu_kernels.__file__), '_xpu_C.abi3.so')
torch.ops.load_library(so)
_ = torch.ops._xpu_C.gdn_attention
print('INCREMENTAL_DONE_V125')
"
