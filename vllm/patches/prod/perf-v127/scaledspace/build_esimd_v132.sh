#!/bin/bash
# build_esimd_v132.sh — v127 M2: ESIMD rebuild carrying the scaled-space
# e4m3 spec kernel (m2_install.py) on top of the current tree state
# (v129 p29a removals + v130 p29dbg diagnostics stay in).
#   1. host-side: copy scaled header + idempotent integrator (marker
#      'v127 M2'); production kernels/entries byte-untouched
#   2. esimd-inc builder (intel/omix:0.1.0-devel-ubuntu24.04), tree at
#      /src/esimd — NEVER touches lsv-test (no-stop directive compliant)
#   3. setup_gdn_only.py build, MAX_JOBS=52 KERNELS_MAX_JOBS=52 (standing
#      directive)
#   4. symbol + freshness proof; .so staged to /root/build/v132_so/
#      Swap into any lane is a SEPARATE manual step (M2 validation only).
set -uo pipefail
TREE=/root/llm-scaler/vllm/custom-esimd-kernels-vllm
STAGE=/root/build/v127_stage/scaledspace
L=/root/build/lce1/m2_v132_build.log
{
echo "=== v127 M2 v132 build start $(date +%F' '%T) ==="

echo "--- 1. host-side install ---"
python3 "$STAGE/m2_install.py" "$TREE"
IRC=$?
echo "install rc=$IRC"
[ $IRC -eq 0 ] || { echo ABORT_INSTALL; exit 1; }
grep -c "v127 M2" "$TREE/csrc/xpu/esimd_kernel_lgrf.sycl"
grep -c "v127 M2" "$TREE/csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_scaled.h"

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
touch /src/esimd/.m2_stamp 2>/dev/null || docker exec esimd-inc touch /src/esimd/.m2_stamp
BUILD_OUT=$(docker exec esimd-inc bash -c '
  set -o pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  export MAX_JOBS=52 KERNELS_MAX_JOBS=52
  export CC=icx CXX=icpx
  cd /src/esimd
  rm -rf build
  /opt/venv/bin/python setup_gdn_only.py build_ext --inplace 2>&1 | tail -300
')
echo "$BUILD_OUT"
echo "--- first error lines (if any) ---"
echo "$BUILD_OUT" | grep -m4 -B2 -A8 "error:" || true
echo "$BUILD_OUT" | grep -qE "error:|FAILED|Error building extension" && { echo NINJA_OR_SETUP_FAILED; exit 3; }

echo "--- 5. locate + prove the .so ---"
SO=$(docker exec esimd-inc bash -c "find /src/esimd -name 'custom_esimd_kernels_lgrf*.so' -newer /src/esimd/.m2_stamp 2>/dev/null | head -1")
echo "built so (CONTAINER path): $SO"
[ -n "$SO" ] || { echo ABORT_NO_SO; exit 4; }
docker exec esimd-inc bash -c "ls -la '$SO'; file '$SO'"
# Symbol proof: under hidden visibility only the pybind WRAPPERS are
# dynamic symbols (the SYCL kernels/launchers are internal) — expect
# exactly 1 exported wrapper each for scaled + production + 1 PyInit.
echo "scaled wrapper symbol (expect 1):"
docker exec esimd-inc bash -c "nm -D '$SO' | grep -c 'esimd_gdn_conv_fused_seq_spec_scaled'"
echo "production wrapper symbols still present (expect 1 each):"
docker exec esimd-inc bash -c "nm -D '$SO' | grep -cE 'esimd_gdn_conv_fused_seq_spec[^_]'" || true
docker exec esimd-inc bash -c "nm -D '$SO' | grep -cE 'esimd_gdn_conv_fused_seq[^_]'" || true
echo "PyInit (expect 1):"
docker exec esimd-inc bash -c "nm -D '$SO' | grep -c PyInit"
docker exec esimd-inc bash -c "sha256sum '$SO'"

echo "--- 6. stage out to /root/build/v132_so ---"
# STAGING LAW: $SO is a CONTAINER path (/src/esimd/...) — on the host the
# same file lives under $TREE (bind mount). Copying $SO on the host fails
# SILENTLY under 'set -uo pipefail' (no -e) — v132 first run staged an
# empty dir and still printed DONE. Stage from the host path and GATE on
# the artifact existing.
SO_HOST=$(echo "$SO" | sed 's|^/src/esimd|'"$TREE"'|')
mkdir -p /root/build/v132_so
cp -a "$SO_HOST" /root/build/v132_so/custom_esimd_kernels_lgrf.so
[ -s /root/build/v132_so/custom_esimd_kernels_lgrf.so ] || { echo ABORT_STAGE_EMPTY; exit 5; }
ls -la /root/build/v132_so/
sha256sum /root/build/v132_so/custom_esimd_kernels_lgrf.so
echo "M2_V132_BUILD_DONE $(date +%T)"
} > "$L" 2>&1
tail -60 "$L"
echo M2_V132_BUILD_SCRIPT_DONE
