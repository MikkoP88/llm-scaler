#!/bin/bash
# run_lane127.sh — v127-audit lane runner (v125 run_lane.sh + loop8 + 262k deep).
# boot(+fallback) -> warmup v51 -> battery -> ctxscan(2k-65k) ->
# deep(128k + 262k-class) -> loop8 (think-trap loop scan) ->
# conc(8@2k, 4 mixed, 2@65k) -> health tail.
# Usage: run_lane127.sh <tag> <kv> <specjson|nospec> [gmu] [maxlen] [extraenv] [image]
set -u
TAG="$1"; KV="$2"; SPEC="$3"; GMU="${4:-0.9}"; MAXLEN="${5:-262144}"; ENVV="${6:-}"; IMG="${7:-llm-scaler-exp:v1.2.5}"
OUT="/root/build/bench127/lane_${TAG}"
mkdir -p "$OUT"
RLOG="$OUT/report.txt"

cnt() { C=$(docker exec lsv-test grep -c "$1" /root/b_$TAG.log 2>/dev/null); echo "${C:-0}"; }

echo "== [lane $TAG] boot $(date +%H:%M:%S) kv=$KV spec=$SPEC gmu=$GMU maxlen=$MAXLEN env=$ENVV img=$IMG" | tee -a "$RLOG"
USED_FALLBACK=0
if ! bash /root/build/serve_bench.sh "$KV" "$SPEC" "$GMU" "$MAXLEN" "b_$TAG" "$ENVV" '' "$IMG" \
    > "$OUT/boot.out" 2>&1; then
  echo "BOOT_FAIL at gmu=$GMU maxlen=$MAXLEN — retrying 0.85/245760" | tee -a "$RLOG"
  tail -15 "$OUT/boot.out" | tee -a "$RLOG"
  USED_FALLBACK=1
  if ! bash /root/build/serve_bench.sh "$KV" "$SPEC" "0.85" "245760" "b_$TAG" "$ENVV" '' "$IMG" \
      > "$OUT/boot_fb.out" 2>&1; then
    echo "LANE_BOOT_FAIL $TAG" | tee -a "$RLOG"
    exit 1
  fi
fi
echo "BOOT_OK fallback=$USED_FALLBACK" | tee -a "$RLOG"
docker exec lsv-test grep -m3 -E "TurboQuant graph-fixed KV splits|GPU KV cache size|# GPU blocks|avail mem" /root/b_$TAG.log 2>/dev/null | tee -a "$RLOG" || true

echo "== [lane $TAG] warmup v51 (p1-p9) $(date +%H:%M:%S)" | tee -a "$RLOG"
ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
WOK=0
for i in $(seq 1 150); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
  curl -s -o /dev/null -m 4 http://localhost:8000/health || true
done
if [ "$WOK" = "1" ]; then echo "WARMUP_OK $(date +%H:%M:%S)" | tee -a "$RLOG"; else echo "WARMUP_TIMEOUT" | tee -a "$RLOG"; fi

echo "== [lane $TAG] battery $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_probe2.py > "$OUT/battery.log" 2>&1
grep -E '^PROBE|^ACCEPT' "$OUT/battery.log" | tee -a "$RLOG"

echo "== [lane $TAG] ctxscan 2k-65k $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_ctxscan.py "$TAG" > "$OUT/ctxscan.out" 2>&1
grep CTXSCAN "$OUT/ctxscan.out" | tee -a "$RLOG"

echo "== [lane $TAG] deep 128k + 262k-class $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_ctxscan2.py "$TAG" "115000,221000" 256 > "$OUT/deep.out" 2>&1
grep CTXSCAN2 "$OUT/deep.out" | tee -a "$RLOG"

echo "== [lane $TAG] loop8 think-trap scan $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_loop8.py "${TAG}_l8" > "$OUT/loop8.out" 2>&1
tail -6 "$OUT/loop8.out" | tee -a "$RLOG"

echo "== [lane $TAG] conc8@2k $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_conc.py "${TAG}_c8" 8 1800 256 > "$OUT/conc8.out" 2>&1
tail -2 "$OUT/conc8.out" | tee -a "$RLOG"

echo "== [lane $TAG] conc4 mixed 2k/16k $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_conc.py "${TAG}_cmix" 4 1800 256 14800 > "$OUT/concmix.out" 2>&1
tail -2 "$OUT/concmix.out" | tee -a "$RLOG"

echo "== [lane $TAG] conc2@65k $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_conc.py "${TAG}_c65k" 2 60000 192 > "$OUT/conc65k.out" 2>&1
tail -2 "$OUT/conc65k.out" | tee -a "$RLOG"

echo "== [lane $TAG] health tail $(date +%H:%M:%S)" | tee -a "$RLOG"
echo "STALLS=$(cnt DFLASH_STALL) STRIKES=$(cnt 'v52m strike') F8=$(cnt 'F8 guard') JIT=$(cnt jit_monitor) ERRS=$(cnt -E 'EngineDeadError|CUDA assert|fatal') DEVLOST=$(cnt 'DEVICE_LOST')" | tee -a "$RLOG"
docker exec lsv-test tail -3 /root/b_$TAG.log 2>/dev/null | tee -a "$RLOG" || true
echo "LANE_DONE $TAG $(date +%H:%M:%S)" | tee -a "$RLOG"
