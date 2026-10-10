#!/bin/bash
# leg_p8.sh TAG DTYPE RATIO MODE — P8 perf-parity leg runner.
#   TAG    artifact suffix (e.g. f16on)
#   DTYPE  fp8_e4m3 | float16        (mamba ssm dtype, serve line 46)
#   RATIO  mamba_pool_ratio (2 for fp16, 1 for fp8)
#   MODE   on  = connector line 57 present with RATIO
#          off = line 57 deleted (engine-native baseline)
# Sequence per WARM-ENGINE CEILING LAW: bench_prefix pass1 (warm-up),
# pass2 + pass3 (records, N=2), bench_run7n x2. Banks everything under
# /root/build/telem_bench_{TAG}_* + mechanism counts from the serve log.
set -u
TAG=$1; DTYPE=$2; RATIO=$3; MODE=$4
SP=/root/serve_lmc_v133.sh
B=/root/build

echo "=== LEG_P8 START tag=$TAG dtype=$DTYPE ratio=$RATIO mode=$MODE $(date -u +%H:%M:%S)"

python3 - "$SP" "$DTYPE" "$RATIO" "$MODE" <<'PYEOF'
import sys
sp, dtype, ratio, mode = sys.argv[1:5]
lines = open(sp).read().split("\n")
assert "mamba-ssm-cache-dtype" in lines[45], lines[45]
lines[45] = "  --mamba-ssm-cache-dtype %s \\" % dtype
KV = ('  --kv-transfer-config \'{"kv_connector":"LMCacheConnectorV1",'
      '"kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,'
      '"lmcache.extra_config":{"mamba_pool_ratio":%s}},"engine_id":"lsv65"}\' \\'
      % ratio)
if mode == "on":
    if "kv-transfer-config" in lines[56]:
        lines[56] = KV
    else:
        lines.insert(56, KV)
else:
    if "kv-transfer-config" in lines[56]:
        del lines[56]
open(sp, "w").write("\n".join(lines))
print("EDIT_OK line46=%s" % lines[45].strip())
print("EDIT line57=%s" % (lines[56].strip() if len(lines) > 56 else "(deleted)"))
PYEOF
if [ $? -ne 0 ]; then echo EDIT_ABORT; exit 1; fi

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
docker exec lsv-test rm -f /tmp/lmc_off /tmp/lmc_snap_off /tmp/lmc_store_off
rm -f $B/pyspy_stall_p8.txt

NDUMPS=0
(
  while [ -f $B/leg_p8.running ]; do
    sleep 15
    MT=$(docker exec lsv-test stat -c %Y /root/serve_full.log 2>/dev/null || echo 0)
    STALE=$(( $(date +%s) - MT ))
    if [ "$STALE" -ge 90 ] && [ "$STALE" -le 420 ]; then
      WP=$(docker exec lsv-test pgrep -f "VLLM::Worker_T[P]" 2>/dev/null || true)
      {
        echo "=== STALE=${STALE}s $(date -u +%H:%M:%S) workers=$WP"
        for p in $WP; do
          docker exec lsv-test /opt/venv/bin/py-spy dump --pid "$p" 2>&1
          echo "--- end dump pid=$p"
        done
      } >> $B/pyspy_stall_p8.txt 2>&1
      NDUMPS=$((NDUMPS+1))
      if [ "$NDUMPS" -ge 6 ]; then echo "WATCHDOG_MAXDUMPS" >> $B/pyspy_stall_p8.txt; break; fi
      sleep 45
    fi
  done
) &
WD=$!
touch $B/leg_p8.running

docker exec -d lsv-test bash $SP
ok=0
for i in $(seq 1 60); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; echo "HEALTH_OK tick=$i"; break; fi
  sleep 10
done
if [ "$ok" != "1" ]; then
  echo "HEALTH_TIMEOUT"
  docker exec lsv-test tail -20 /root/serve_full.log
  rm -f $B/leg_p8.running
  kill $WD 2>/dev/null
  exit 1
fi

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

sleep 5
rm -f $B/leg_p8.running
kill $WD 2>/dev/null

echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_p8.log
docker cp lsv-test:/tmp/telem_p8.log $B/telem_bench_${TAG}_serve.log >/dev/null
md5sum $B/telem_bench_${TAG}_serve.log
echo "--- mechanism counts:"
for pat in "seed parked" "store(final)" "LMC joint lookup" "LMCNANDETECT" "Traceback"; do
  n=$(docker exec lsv-test sh -c "grep -ac \"$pat\" /root/serve_full.log" 2>/dev/null || echo 0)
  echo "COUNT[$pat]=$n"
done
echo "--- pyspy stall dumps:"
wc -l $B/pyspy_stall_p8.txt 2>/dev/null || echo "NO_STALL_FILE (no freeze captured)"
echo "--- prefix p2/p3 reports:"
tail -12 $B/bench_${TAG}_prefix_p2.out
tail -12 $B/bench_${TAG}_prefix_p3.out
echo "--- run7n reports:"
cat $B/bench_${TAG}_run7n_1.out
cat $B/bench_${TAG}_run7n_2.out
echo "--- engine alive:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== LEG_P8 DONE tag=$TAG $(date -u +%H:%M:%S)"
