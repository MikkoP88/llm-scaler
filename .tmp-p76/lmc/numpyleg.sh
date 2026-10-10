#!/bin/bash
# numpyleg.sh N — LEG 11: restore ONLY the image's numpy 2.3.5 into lsv-test
# (pip downgrade 2.3.5->2.2.6 is prime suspect). Then boot + probe N.
set -u
N=${1:-100}
B=/root/build
S=/opt/venv/lib/python3.12/site-packages
IMG=llm-scaler-exp:v1.2.32

echo "=== NUMPYLEG START n=$N $(date -u +%H:%M:%S)"

# stop + remove lsv-fresh (frees port 8000 + VRAM)
for p in $(docker exec lsv-fresh pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-fresh kill -TERM $p 2>/dev/null || true
done
sleep 14
for p in $(docker exec lsv-fresh pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-fresh kill -9 $p 2>/dev/null || true
done
docker rm -f lsv-fresh >/dev/null 2>&1 || true
echo "lsv-fresh removed"

# what numpy pieces does the image ship?
echo "--- image numpy pieces:"
docker run --rm --entrypoint ls $IMG $S | grep -i "^numpy" || true

# remove 2.2.6 from lsv-test, restore 2.3.5 from image
docker exec lsv-test sh -c "rm -rf $S/numpy $S/numpy-2.2.6.dist-info $S/numpy.libs"
docker run --rm --entrypoint sh $IMG -c "cd $S && tar cf - numpy numpy-2.3.5.dist-info numpy.libs 2>/dev/null" \
  | docker exec -i lsv-test sh -c "cd $S && tar xf -"
echo "--- numpy in lsv-test now:"
docker exec lsv-test /opt/venv/bin/python -c "import numpy; print(numpy.__version__, numpy.__file__)" \
  || { echo NUMPY_IMPORT_FAIL; exit 1; }

bash $B/legboot.sh numpyfix $N
echo "=== NUMPYLEG DONE $(date -u +%H:%M:%S)"
