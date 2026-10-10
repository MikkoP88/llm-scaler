#!/bin/bash
# leg_p9sd.sh TAG MODE — P9b SAVE_DECODE_CACHE A/B leg (user directive
# 2026-10-10: LMCACHE_SAVE_DECODE_CACHE=True vs baseline on v1.2.33).
#   MODE=true|false -> flips LMCACHE_SAVE_DECODE_CACHE in the serve script
#   MODE=none       -> NO edit, NO restart; bench the engine as-is
# Standing shape otherwise fixed: mamba float16, ratio 2, connector ON.
# Bench suite identical to leg_p9.sh (warm-engine law: p1 warm, p2/p3
# records, run7n x2). Banks under /root/build/bench_{TAG}_*.
set -u
TAG=$1
MODE=$2
SP=/root/serve_lmc_v133.sh
B=/root/build

echo "=== LEG_P9SD START tag=$TAG mode=$MODE $(date -u +%H:%M:%S)"

if [ "$MODE" != "none" ]; then
  python3 - "$SP" "$MODE" <<'PYEOF'
import sys
sp, mode = sys.argv[1:3]
lines = open(sp).read().split("\n")
sline = None
for i, l in enumerate(lines):
    if "LMCACHE_SAVE_DECODE_CACHE" in l:
        sline = i
        break
if sline is None:
    raise SystemExit("SAVE_LINE_NOT_FOUND")
lines[sline] = "export LMCACHE_SAVE_DECODE_CACHE=%s" % ("True" if mode == "true" else "False")
open(sp, "w").write("\n".join(lines))
print("EDIT_OK line=%d value=%s" % (sline + 1, mode))
PYEOF
  [ $? -eq 0 ] || { echo EDIT_ABORT; exit 1; }

  docker cp $SP lsv-test:$SP

  PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    for p in $PIDS; do docker exec lsv-test kill -TERM "$p" 2>/dev/null || true; done
    sleep 14
  fi
  PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "TERM_LEFTOVERS=$PIDS -> kill -9"
    for p in $PIDS; do docker exec lsv-test kill -9 "$p" 2>/dev/null || true; done
    sleep 6
  fi

  docker exec -d lsv-test bash $SP
  ok=0
  for i in $(seq 1 110); do
    code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
    if [ "$code" = "200" ]; then ok=1; echo "HEALTH_OK tick=$i $(date -u +%H:%M:%S)"; break; fi
    sleep 10
  done
  if [ "$ok" != "1" ]; then
    echo "HEALTH_TIMEOUT"
    docker exec lsv-test tail -20 /root/serve_full.log
    exit 1
  fi
fi

echo "SHAPE_CHECK: $(grep LMCACHE_SAVE_DECODE_CACHE $SP | head -1) | $(docker exec lsv-test sh -c "grep -o 'mamba-ssm-cache-dtype [a-z0-9_]*' /proc/\$(pgrep -f 'vllm serve' | head -1)/cmdline | head -1" 2>/dev/null | tr '\0' ' ')"

cd $B || exit 1
echo "--- prefix pass1 (warm-up) start $(date -u +%H:%M:%S)"
python3 bench_prefix.py > $B/bench_${TAG}_prefix_p1.out 2>&1
echo "P1_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- prefix pass2 (record 1) start $(date -u +%H:%M:%S)"
python3 bench_prefix.py > $B/bench_${TAG}_prefix_p2.out 2>&1
echo "P2_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- prefix pass3 (record 2) start $(date -u +%H:%M:%S)"
python3 bench_prefix.py > $B/bench_${TAG}_prefix_p3.out 2>&1
echo "P3_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- run7n x2 start $(date -u +%H:%M:%S)"
python3 bench_run7n.py > $B/bench_${TAG}_run7n_1.out 2>&1
echo "R1_EXIT=$? $(date -u +%H:%M:%S)"
python3 bench_run7n.py > $B/bench_${TAG}_run7n_2.out 2>&1
echo "R2_EXIT=$? $(date -u +%H:%M:%S)"

echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_sd.log
docker cp lsv-test:/tmp/telem_sd.log $B/telem_bench_${TAG}_serve.log >/dev/null
md5sum $B/telem_bench_${TAG}_serve.log
echo "--- mechanism counts:"
for pat in "seed parked" "store(final)" "LMC joint lookup" "LMCNANDETECT" "Traceback" "Double unpin" "lookup_unpin"; do
  n=$(docker exec lsv-test sh -c "grep -ac \"$pat\" /root/serve_full.log" 2>/dev/null || echo 0)
  echo "COUNT[$pat]=$n"
done
echo "--- prefix p2/p3 reports:"
tail -12 $B/bench_${TAG}_prefix_p2.out
tail -12 $B/bench_${TAG}_prefix_p3.out
echo "--- run7n reports:"
cat $B/bench_${TAG}_run7n_1.out
cat $B/bench_${TAG}_run7n_2.out
echo "--- engine alive:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== LEG_P9SD DONE tag=$TAG $(date -u +%H:%M:%S)"
