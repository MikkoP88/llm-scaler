#!/bin/bash
# cclrestore.sh N — LEG 12: remove pip-installed oneCCL 2021.17.2 from
# /opt/venv/lib and restore the image's oneAPI 2021.15 symlinks.
set -u
N=${1:-100}
B=/root/build
S=/opt/venv/lib/python3.12/site-packages
L=/opt/venv/lib
IMG=llm-scaler-exp:v1.2.32

echo "=== CCLRESTORE START n=$N $(date -u +%H:%M:%S)"

echo "--- oneAPI 2021.15 base present in lsv-test?"
docker exec lsv-test sh -c "ls -la /opt/intel/oneapi/ccl/2021.15/lib/ | head -8"

echo "--- current (pip) state:"
docker exec lsv-test sh -c "ls -la $L/libccl* 2>/dev/null | head -12"

# remove everything pip's oneccl wheel owns in /opt/venv/lib (+ dist-infos)
docker exec lsv-test sh -c "rm -f $L/libccl.so $L/libccl.so.1 $L/libccl.so.1.0 \
  $L/libccl.so.2 $L/libccl.so.2.0 $L/libccl_legacy.so $L/libccl_legacy_cpu.so \
  $L/libccl_openmp.so $L/libccl_openmp.so.0 $L/libccl_openmp.so.0.1; \
  rm -rf $L/ccl $S/oneccl-2021.17.2.dist-info $S/oneccl_devel-2021.17.2.dist-info; true"
echo "pip oneCCL removed"

# restore image's symlink set (absolute targets -> oneAPI 2021.15)
docker run --rm --entrypoint sh $IMG -c "cd $L && tar cf - libccl.so libccl.so.1" \
  | docker exec -i lsv-test sh -c "cd $L && tar xf -"
echo "--- restored state:"
docker exec lsv-test sh -c "ls -la $L/libccl* 2>/dev/null"
docker exec lsv-test sh -c "ldd /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_xpu.so 2>/dev/null | grep -i ccl" || true
echo "--- any other stray pip ccl pieces:"
docker exec lsv-test sh -c "find /opt/venv -maxdepth 2 -name \"*ccl*\" 2>/dev/null | grep -v python3.12 | head"

bash $B/legboot.sh cclfix $N
echo "=== CCLRESTORE DONE $(date -u +%H:%M:%S)"
