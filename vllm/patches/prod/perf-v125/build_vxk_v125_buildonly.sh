#!/bin/bash
# build_vxk_v125_buildonly.sh — v125 P22B wheel BUILD-ONLY leg.
# Identical to build_vxk_v125_inc.sh (marker check -> warm vxk-inc builder
# env -> ninja -j 52 _xpu_C -> patchelf -> binary strings proof) but STOPS
# BEFORE the lsv-test swap: the swap must not run while a diag chain owns
# lsv-test (a swapped wheel would put the certified serve on a
# non-certified kernel). Swap is a separate step once the lane is free.
# KERNELS_MAX_JOBS=52 standing directive honored (ninja -j 52).
set -uo pipefail
L=/root/build/lce1/v125_buildonly.log
{
echo "=== v125 BUILDONLY start $(date +%F' '%T) ==="

grep -q 'v125 P22B' /root/build/vxk/csrc/xpu/gdn_attn/gated_delta_rule.hpp \
  || { echo ABORT_PATCH_NOT_PRESENT; exit 1; }
echo "patch marker present in source tree"

docker inspect vxk-inc >/dev/null 2>&1 || docker run -d --name vxk-inc \
  -v /root/build/vxk:/src/vllm-xpu-kernels \
  intel/omix:0.1.0-devel-ubuntu24.04 sleep infinity >/dev/null
docker start vxk-inc >/dev/null 2>&1 || true
echo "builder up"

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
' || { echo ABORT_ENV; exit 2; }

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

STR_E4M3=$(docker exec vxk-inc bash -c "strings '$CSO' | grep -c 'fp8_e4m3fn'" || true)
STR_E5M2=$(docker exec vxk-inc bash -c "strings '$CSO' | grep -c 'fp8_e5m2'" || true)
echo "so_string_hits e4m3=$STR_E4M3 e5m2=$STR_E5M2 (need >=1 each)"
[ "$STR_E4M3" -ge 1 ] 2>/dev/null || { echo ABORT_SO_MISSING_E4M3_LITERAL; exit 5; }
[ "$STR_E5M2" -ge 1 ] 2>/dev/null || { echo ABORT_SO_MISSING_E5M2_LITERAL; exit 5; }

md5sum "$SO"
echo "=== v125 BUILDONLY DONE $(date +%F' '%T) — .so ready at $SO; lsv-test swap NOT performed ==="
echo V125_BUILDONLY_DONE
} > "$L" 2>&1
tail -25 "$L"
echo V125_BUILDONLY_SCRIPT_DONE
