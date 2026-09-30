#!/bin/bash
# build_esimd_v130.sh — v126 P29B-DIAG: ESIMD rebuild carrying the
# instrumented spec-kernel variant (p29dbg_install.py) on top of the
# current v129 tree state (p29a quantization-roundtrip removals stay in).
#   1. host-side: copy dbg header + idempotent integrator (marker v126
#      P29B-DIAG); production kernels byte-untouched
#   2. esimd-inc builder (intel/omix:0.1.0-devel-ubuntu24.04), tree at
#      /src/esimd — NEVER touches lsv-test
#   3. setup_gdn_only.py build, MAX_JOBS=52 KERNELS_MAX_JOBS=52 (standing
#      directive)
#   4. symbol + freshness proof; .so staged to /root/build/v130_so/
#      Swap into lsv-test is a SEPARATE manual step (diagnostics only).
set -uo pipefail
TREE=/root/llm-scaler/vllm/custom-esimd-kernels-vllm
L=/root/build/lce1/p129dbg_build.log
{
echo "=== P29B-DIAG build start $(date +%F' '%T) ==="

echo "--- 1. host-side install ---"
python3 /root/build/p29dbg_install.py
IRC=$?
echo "install rc=$IRC"
[ $IRC -eq 0 ] || { echo ABORT_INSTALL; exit 1; }
grep -c "P29B-DIAG" "$TREE/csrc/xpu/esimd_kernel_lgrf.sycl"
grep -c "P29B-DIAG" "$TREE/csrc/xpu/torch_extension_lgrf.cc"

echo "--- 2. builder container ---"
docker inspect esimd-inc >/dev/null 2>&1 || docker run -d --name esimd-inc \
  -v "$TREE":/src/esimd \
  intel/omix:0.1.0-devel-ubuntu24.04 sleep infinity >/dev/null
docker start esimd-inc >/dev/null 2>&1 || true
echo "esimd-inc up"

echo "--- 3. env (idempotent) ---"
docker exec esimd-inc bash -c '
  set -eo pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export PATH=/root/.local/bin:$PATH
  [ -x /opt/venv/bin/python ] || { echo NO_VENV; exit 9; }
  /opt/venv/bin/python -c "import torch; print(\"ENV_READY torch\", torch.__version__)"
  command -v icpx >/dev/null && icpx --version | head -1
' || { echo ENV_FAIL; exit 2; }

echo "--- 4. build (MAX_JOBS=52 KERNELS_MAX_JOBS=52) ---"
touch /src/esimd/.p129dbg_stamp 2>/dev/null || docker exec esimd-inc touch /src/esimd/.p129dbg_stamp
BUILD_OUT=$(docker exec esimd-inc bash -c '
  set -o pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  export MAX_JOBS=52 KERNELS_MAX_JOBS=52
  export CC=icx CXX=icpx
  cd /src/esimd
  rm -rf build
  /opt/venv/bin/python setup_gdn_only.py build_ext --inplace 2>&1 | tail -60
')
echo "$BUILD_OUT"
echo "$BUILD_OUT" | grep -qE "error:|FAILED|Error building extension" && { echo NINJA_OR_SETUP_FAILED; exit 3; }

echo "--- 5. locate + prove the .so ---"
SO=$(docker exec esimd-inc bash -c "find /src/esimd -name 'custom_esimd_kernels_lgrf*.so' -newer /src/esimd/.p129dbg_stamp 2>/dev/null | head -1")
echo "built so: $SO"
[ -n "$SO" ] || { echo ABORT_NO_SO; exit 4; }
docker exec esimd-inc bash -c "ls -la '$SO'; file '$SO'"
docker exec esimd-inc bash -c "nm -D '$SO' | grep -c 'esimd_gdn_conv_fused_seq_spec_dbg'" || true
docker exec esimd-inc bash -c "sha256sum '$SO'"

echo "--- 6. stage out to /root/build/v130_so ---"
mkdir -p /root/build/v130_so
cp -a "$SO" /root/build/v130_so/
SO2=$(find "$TREE" -name 'custom_esimd_kernels_lgrf*.so' -newer "$TREE/.p129dbg_stamp" 2>/dev/null | head -1)
[ -n "$SO2" ] && cp -a "$SO2" /root/build/v130_so/inplace.so 2>/dev/null || true
ls -la /root/build/v130_so/
sha256sum /root/build/v130_so/custom_esimd_kernels_lgrf.so
echo "P29B_DIAG_BUILD_DONE $(date +%T)"
} > "$L" 2>&1
tail -60 "$L"
echo P29B_DIAG_BUILD_SCRIPT_DONE
