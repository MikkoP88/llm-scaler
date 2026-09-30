#!/bin/bash
# build_esimd_v131.sh — v126 P29B-FIX: ESIMD rebuild carrying the ring-row
# snapshot fix for the spec kernel (p29fix_install.py) on top of the current
# tree (p29a roundtrip removals + p29b diag op stay in).
#   1. host-side: install fixed headers + snapshot wrappers (marker
#      'v126 P29B-FIX'); one-time backup to /root/build/p29fix_backup/
#   2. esimd-inc builder (intel/omix:0.1.0-devel-ubuntu24.04), tree at
#      /src/esimd — NEVER touches lsv-test
#   3. setup_gdn_only.py build, MAX_JOBS=52 KERNELS_MAX_JOBS=52 (standing
#      directive)
#   4. symbol + freshness proof; .so staged to /root/build/v131_so/
#      Swap into lsv-test is a SEPARATE manual step.
# Full compiler output preserved at /root/build/lce1/p29fix_full_build.log
# (lesson KNOWN_ISSUES: tail -60 inside the exec swallows 'error:' lines —
# the gate below greps the FULL log).
set -uo pipefail
TREE=/root/llm-scaler/vllm/custom-esimd-kernels-vllm
L=/root/build/lce1/p29fix_build.log
FULL=/root/build/lce1/p29fix_full_build.log
mkdir -p /root/build/lce1 /root/build/v131_so
{
echo "=== P29B-FIX build start $(date +%F' '%T) ==="

echo "--- 1. host-side install ---"
python3 /root/build/p29fix_install.py
IRC=$?
echo "install rc=$IRC"
[ $IRC -eq 0 ] || { echo ABORT_INSTALL; exit 1; }
grep -c "P29B-FIX" "$TREE/csrc/xpu/esimd_kernel_lgrf.sycl"
grep -c "P29B-FIX" "$TREE/csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec.h"

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
touch "$TREE/.p29fix_stamp"
docker exec esimd-inc bash -c '
  set -o pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  export MAX_JOBS=52 KERNELS_MAX_JOBS=52
  export CC=icx CXX=icpx
  cd /src/esimd
  rm -rf build
  /opt/venv/bin/python setup_gdn_only.py build_ext --inplace 2>&1
  echo "BUILD_EXIT_RC=$?"
' > "$FULL" 2>&1
tail -40 "$FULL"
grep -q "BUILD_EXIT_RC=0" "$FULL" || { echo BUILD_NONZERO; exit 3; }
if grep -nE "error:|FAILED|Error building extension" "$FULL" | head -20; then
  echo NINJA_OR_SETUP_FAILED; exit 3
fi

echo "--- 5. locate + prove the .so ---"
SO=$(docker exec esimd-inc bash -c "find /src/esimd -name 'custom_esimd_kernels_lgrf*.so' -newer /src/esimd/.p29fix_stamp 2>/dev/null | head -1")
echo "built so: $SO"
[ -n "$SO" ] || { echo ABORT_NO_SO; exit 4; }
docker exec esimd-inc bash -c "ls -la '$SO'"
docker exec esimd-inc bash -c "nm -D '$SO' | grep -c 'esimd_gdn_conv_fused_seq_spec_dbg'" || true
docker exec esimd-inc bash -c "sha256sum '$SO'"

echo "--- 6. stage out to /root/build/v131_so ---"
cp -a "$SO" /root/build/v131_so/
SO2=$(find "$TREE" -name 'custom_esimd_kernels_lgrf*.so' -newer "$TREE/.p29fix_stamp" 2>/dev/null | head -1)
[ -n "$SO2" ] && cp -a "$SO2" /root/build/v131_so/inplace.so || true
ls -la /root/build/v131_so/
sha256sum /root/build/v131_so/* 2>/dev/null
echo "P29FIX_BUILD_DONE $(date +%T)"
} > "$L" 2>&1
tail -60 "$L"
echo P29FIX_BUILD_SCRIPT_DONE
