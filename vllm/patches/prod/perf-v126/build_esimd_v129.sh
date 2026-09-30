#!/bin/bash
# build_esimd_v129.sh — v126 P29A: ESIMD lgrf rebuild with the quantization
# roundtrip removals (p29a_kernel_patch.py): single-conversion e4m3 load
# (c10-parity NaN semantics) + fp8 SSM register chain in the spec kernel
# (W-1 pool loads removed, zero requant feedback; fp16 stays bit-identical
# via constexpr kFp8Chain = sizeof(StateT)==1).
#   1. host-side: backup 2 targets + apply p29a_kernel_patch.py (idempotent)
#   2. esimd-inc builder (intel/omix:0.1.0-devel-ubuntu24.04) with the tree
#      mounted at /src/esimd — NEVER touches lsv-test
#   3. build via setup_gdn_only.py (SyclExtension, -Xs -device bmg),
#      MAX_JOBS=52 KERNELS_MAX_JOBS=52 (standing directive)
#   4. symbol + freshness proof; .so copied out to /root/build/v129_so/
#      The swap into lsv-test is a SEPARATE gated step (never swap a .so
#      under a running experimental leg).
# Rollback: *.v129prebak copies in the tree + inverse edits.
set -uo pipefail
TREE=/root/llm-scaler/vllm/custom-esimd-kernels-vllm
L=/root/build/lce1/p129a_build.log
{
echo "=== P29A ESIMD build start $(date +%F' '%T) ==="

echo "--- 1. host-side patch (backup first) ---"
mkdir -p /root/build/p129a_prebak
for f in \
  csrc/xpu/esimd_kernels/utils.h \
  csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec.h; do
  [ -f "/root/build/p129a_prebak/$(basename $f)" ] || cp -a "$TREE/$f" "/root/build/p129a_prebak/$(basename $f)"
done
ls -la /root/build/p129a_prebak/
python3 /root/build/p29a_kernel_patch.py
PATCHRC=$?
echo "patch rc=$PATCHRC"
[ $PATCHRC -eq 0 ] || { echo ABORT_PATCH; exit 1; }
grep -c "v126 P29A" "$TREE/csrc/xpu/esimd_kernels/utils.h"
grep -c "v126 P29A" "$TREE/csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec.h"

echo "--- 2. builder container ---"
docker inspect esimd-inc >/dev/null 2>&1 || docker run -d --name esimd-inc \
  -v "$TREE":/src/esimd \
  intel/omix:0.1.0-devel-ubuntu24.04 sleep infinity >/dev/null
docker start esimd-inc >/dev/null 2>&1 || true
echo "esimd-inc up"

echo "--- 3. env (same recipe as vxk-inc, idempotent) ---"
docker exec esimd-inc bash -c '
  set -eo pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq --no-install-recommends curl ca-certificates \
    ninja-build python3-dev pkg-config numactl patchelf file >/dev/null 2>&1
  export PATH=/root/.local/bin:$PATH
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH=/root/.local/bin:$PATH
  [ -x /opt/venv/bin/python ] || uv venv /opt/venv >/dev/null
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  if ! /opt/venv/bin/python -c "import torch" >/dev/null 2>&1; then
    uv pip install -q pip wheel setuptools
    uv pip install -q torch==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu
  fi
  /opt/venv/bin/python -c "import torch; print(\"ENV_READY torch\", torch.__version__)"
  command -v icpx && icpx --version | head -1
'

echo "--- 4. build (MAX_JOBS=52 KERNELS_MAX_JOBS=52) ---"
touch /src/esimd/.p129a_stamp 2>/dev/null || docker exec esimd-inc touch /src/esimd/.p129a_stamp
BUILD_OUT=$(docker exec esimd-inc bash -c '
  set -o pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  export MAX_JOBS=52 KERNELS_MAX_JOBS=52
  export CC=icx CXX=icpx
  cd /src/esimd
  rm -rf build
  /opt/venv/bin/python setup_gdn_only.py build_ext --inplace 2>&1 | tail -40
')
echo "$BUILD_OUT"
echo "$BUILD_OUT" | grep -qE "error:|FAILED|Error building extension" && { echo NINJA_OR_SETUP_FAILED; exit 3; }

echo "--- 5. locate + prove the .so ---"
SO=$(docker exec esimd-inc bash -c "find /src/esimd -name 'custom_esimd_kernels_lgrf*.so' -newer /src/esimd/.p129a_stamp 2>/dev/null | head -1")
echo "built so: $SO"
[ -n "$SO" ] || { echo ABORT_NO_SO; exit 4; }
docker exec esimd-inc bash -c "ls -la '$SO'; file '$SO'"
docker exec esimd-inc bash -c "nm -D '$SO' | grep -c 'esimd_gdn_conv_fused'" || true
docker exec esimd-inc bash -c "sha256sum '$SO'"
docker exec esimd-inc bash -c "strings '$SO' | grep -ciE 'fp8|e4m3|e5m2'" || true

echo "--- 6. stage out to /root/build/v129_so ---"
mkdir -p /root/build/v129_so
# $SO is the in-container path; stage from the HOST tree (bind mount twin)
SO_HOST=$(find "$TREE" -name 'custom_esimd_kernels_lgrf*.so' -newer "$TREE/.p129a_stamp" 2>/dev/null | head -1)
echo "host so: $SO_HOST"
[ -n "$SO_HOST" ] || { echo ABORT_NO_HOST_SO; exit 5; }
cp -a "$SO_HOST" /root/build/v129_so/custom_esimd_kernels_lgrf.so
ls -la /root/build/v129_so/
sha256sum /root/build/v129_so/custom_esimd_kernels_lgrf.so
echo "P29A_BUILD_DONE $(date +%T)"
} > "$L" 2>&1
tail -80 "$L"
echo P29A_BUILD_SCRIPT_DONE
