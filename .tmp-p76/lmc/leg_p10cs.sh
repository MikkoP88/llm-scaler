#!/bin/bash
# leg_p10cs.sh TAG MODE — P10 CHUNK-SIZE A/B leg (user directive 2026-10-10:
# do LMCACHE_CHUNK_SIZE=256 + LMCACHE_MIN_RETRIEVE_TOKENS=256 help the
# long-context reprompt TTFT / looping issues?).
#   MODE=256  -> both envs to 256 (edit + restart + bench)
#   MODE=1024 -> both back to 1024 (revert leg)
#   MODE=none -> NO edit, NO restart; bench the engine as-is
# Standing shape otherwise fixed: v1.2.34 content, mamba float16, ratio 2,
# connector ON, SAVE_DECODE_CACHE=True, CPU pool 100, LOG_LEVEL=INFO.
# Bench suite = P9b (prefix p1 warm / p2 p3 records, run7n x2) + NEW
# bench_longctx (same-session reprompt probe: small 600tok / long 32ktok
# paired turns — the MIN_RETRIEVE gate + chunk-boundary depth axes).
set -u
TAG=$1
MODE=$2
SP=/root/serve_lmc_v133.sh
B=/root/build

echo "=== LEG_P10CS START tag=$TAG mode=$MODE $(date -u +%H:%M:%S)"

if [ "$MODE" != "none" ]; then
  python3 - "$SP" "$MODE" <<'PYEOF'
import sys
sp, mode = sys.argv[1:3]
lines = open(sp).read().split("\n")
hits = {}
for i, l in enumerate(lines):
    for tgt in ("LMCACHE_CHUNK_SIZE", "LMCACHE_MIN_RETRIEVE_TOKENS"):
        if l.startswith("export " + tgt + "="):
            lines[i] = "export %s=%s" % (tgt, mode)
            hits[tgt] = i + 1
if len(hits) != 2:
    raise SystemExit("ENV_LINES_NOT_FOUND %s" % hits)
open(sp, "w").write("\n".join(lines))
print("EDIT_OK %s" % hits)
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
  echo "--- env-at-start verification (serving pid environ):"
  SPID=$(docker exec lsv-test pgrep -f "vllm serve" | head -1)
  docker exec lsv-test sh -c "tr '\0' '\n' < /proc/$SPID/environ" | grep -E "LMCACHE_|PYTHONHASHSEED" | sort
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
echo "--- longctx reprompt probe start $(date -u +%H:%M:%S)"
python3 bench_longctx.py > $B/bench_${TAG}_longctx.out 2>&1
echo "LC_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- run7n x2 start $(date -u +%H:%M:%S)"
python3 bench_run7n.py > $B/bench_${TAG}_run7n_1.out 2>&1
echo "R1_EXIT=$? $(date -u +%H:%M:%S)"
python3 bench_run7n.py > $B/bench_${TAG}_run7n_2.out 2>&1
echo "R2_EXIT=$? $(date -u +%H:%M:%S)"

echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_cs.log
docker cp lsv-test:/tmp/telem_cs.log $B/telem_bench_${TAG}_serve.log >/dev/null
md5sum $B/telem_bench_${TAG}_serve.log
echo "--- mechanism counts:"
for pat in "Retrieved" "Stored" "POISONED" "Pin timeout" "LMCNANDETECT" "Traceback" "Double unpin"; do
  n=$(docker exec lsv-test sh -c "grep -ac \"$pat\" /root/serve_full.log" 2>/dev/null || echo 0)
  echo "COUNT[$pat]=$n"
done
echo "--- store chunk-size sample (256-chunk must be ~0.014gb not 0.038gb):"
docker exec lsv-test sh -c "grep Stored /root/serve_full.log | tail -3"
echo "--- prefix p2/p3 reports:"
tail -12 $B/bench_${TAG}_prefix_p2.out
tail -12 $B/bench_${TAG}_prefix_p3.out
echo "--- longctx report:"
cat $B/bench_${TAG}_longctx.out
echo "--- run7n reports:"
cat $B/bench_${TAG}_run7n_1.out
cat $B/bench_${TAG}_run7n_2.out
echo "--- engine alive:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== LEG_P10CS DONE tag=$TAG $(date -u +%H:%M:%S)"
