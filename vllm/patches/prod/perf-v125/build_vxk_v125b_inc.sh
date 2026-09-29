#!/bin/bash
# build_vxk_v125b_inc.sh — v125 P22B addendum: incremental _xpu_C rebuild
# after the xe2 PREFILL fp8-native dispatch patch (p22b_xe2_patch.py,
# markers 'v125 P22B' in chunk_gated_delta_rule_kernels_xe2.hpp).
# Same builder/env as build_vxk_v125_inc.sh; the only deltas:
#   - patch-present check keys on the xe2 header
#   - binary proof: strings 'fp8_e5m2' count must STRICTLY INCREASE vs the
#     pre-build .so (the v125 decode patch already baked one literal; the
#     xe2 reject message adds another — a count delta is the discriminator)
#   - pre-xe2 .so preserved as .v125b2bak (rollback to the decode-patch-only
#     artifact if the xe2 leg misbehaves; .v125bak stays = pre-P22B wheel)
set -uo pipefail

XE2=/root/build/vxk/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp
grep -q 'v125 P22B' "$XE2" || { echo ABORT_PATCH_NOT_PRESENT; exit 1; }
grep -q 'at::kFloat8_e4m3fn' "$XE2" || { echo ABORT_BRANCHES_MISSING; exit 1; }
echo "xe2 patch marker present in source tree"

docker inspect vxk-inc >/dev/null 2>&1 || docker run -d --name vxk-inc \
  -v /root/build/vxk:/src/vllm-xpu-kernels \
  intel/omix:0.1.0-devel-ubuntu24.04 sleep infinity >/dev/null
docker start vxk-inc >/dev/null 2>&1 || true
echo "builder up"

# baseline string count from the CURRENT lsv-test .so (decode-patch build)
SP=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
BASE_E5M2=$(docker exec lsv-test bash -c "strings $SP/_xpu_C.abi3.so | grep -c 'fp8_e5m2'" || true)
echo "baseline so fp8_e5m2 literals: ${BASE_E5M2:-0}"

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

# incremental ninja (KERNELS_MAX_JOBS=52 standing directive); grep filter so
# an error line can never hide under a 'tail' window (the v125 ninja lesson)
docker exec vxk-inc bash -c '
  set -o pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  cd /src/vllm-xpu-kernels/build/temp
  ninja -j 52 _xpu_C 2>&1 | grep -E "error|Error|FAILED|warning: #include|ninja: build stopped" || true
  ninja -j 52 _xpu_C >/dev/null 2>&1 && echo NINJA_OK || echo NINJA_FAILED
' | tee /tmp/v125b_ninja.out
grep -q NINJA_OK /tmp/v125b_ninja.out || { echo ABORT_NINJA; exit 3; }

SO=/root/build/vxk/build/temp/_xpu_C.abi3.so
CSO=/src/vllm-xpu-kernels/build/temp/_xpu_C.abi3.so
[ -f "$SO" ] || { echo NO_SO_FOUND; exit 4; }

docker exec vxk-inc patchelf --set-rpath '$ORIGIN' "$CSO"
docker exec vxk-inc readelf -d "$CSO" | grep RUNPATH

# xe2 patch is IN the binary: fp8_e5m2 literal count strictly above baseline
NEW_E5M2=$(docker exec vxk-inc bash -c "strings '$CSO' | grep -c 'fp8_e5m2'" || true)
echo "new so fp8_e5m2 literals: ${NEW_E5M2:-0} (baseline ${BASE_E5M2:-0}, need > baseline)"
[ "${NEW_E5M2:-0}" -gt "${BASE_E5M2:-0}" ] 2>/dev/null || { echo ABORT_NO_NEW_FP8_LITERAL; exit 5; }

# swap into lsv-test: preserve the current (decode-patch) .so as .v125b2bak
docker exec lsv-test cp -a $SP/_xpu_C.abi3.so $SP/_xpu_C.abi3.so.v125b2bak 2>/dev/null || true
docker cp "$SO" lsv-test:$SP/_xpu_C.abi3.so
docker exec lsv-test /opt/venv/bin/python -c "
import os
import torch
import vllm_xpu_kernels
so = os.path.join(os.path.dirname(vllm_xpu_kernels.__file__), '_xpu_C.abi3.so')
torch.ops.load_library(so)
_ = torch.ops._xpu_C.gdn_attention
print('INCREMENTAL_DONE_V125B')
"
