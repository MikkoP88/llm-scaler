#!/bin/bash
# run_lane126.sh — one validation lane on llm-scaler-exp:v1.2.6t1 (2026-09-07).
# Same battery as run_lane.sh but: image arg, dt_warmup_v53 (p10/p11 deep
# buckets), and v53-specific log gates (L2 arm/fire, P-1 routes, L4 counters,
# L10 tier logs).
# Usage: run_lane126.sh <tag> <kv> <specjson|nospec> [gmu] [maxlen] [extraenv] [image]
set -u
TAG="$1"; KV="$2"; SPEC="$3"; GMU="${4:-0.9}"; MAXLEN="${5:-262144}"; ENVV="${6:-}"
IMG="${7:-llm-scaler-exp:v1.2.6t1}"
OUT="/root/build/bench126/lane_${TAG}"
mkdir -p "$OUT"
RLOG="$OUT/report.txt"

echo "== [lane $TAG] boot $(date +%H:%M:%S) kv=$KV spec=$SPEC gmu=$GMU maxlen=$MAXLEN env=$ENVV img=$IMG" | tee -a "$RLOG"
USED_FALLBACK=0
if ! bash /root/build/serve_bench.sh "$KV" "$SPEC" "$GMU" "$MAXLEN" "b_$TAG" "$ENVV" "" "$IMG" \
    > "$OUT/boot.out" 2>&1; then
  echo "BOOT_FAIL at gmu=$GMU maxlen=$MAXLEN — retrying 0.85/245760" | tee -a "$RLOG"
  tail -15 "$OUT/boot.out" | tee -a "$RLOG"
  USED_FALLBACK=1
  if ! bash /root/build/serve_bench.sh "$KV" "$SPEC" "0.85" "245760" "b_$TAG" "$ENVV" "" "$IMG" \
      > "$OUT/boot_fb.out" 2>&1; then
    echo "LANE_BOOT_FAIL $TAG" | tee -a "$RLOG"
    exit 1
  fi
fi
echo "BOOT_OK fallback=$USED_FALLBACK" | tee -a "$RLOG"
docker exec lsv-test grep -m3 -E "TurboQuant graph-fixed KV splits|KV cache size|# GPU blocks|avail mem" /root/b_$TAG.log 2>/dev/null | tee -a "$RLOG" || true

echo "== [lane $TAG] warmup v53 (p1-p11) $(date +%H:%M:%S)" | tee -a "$RLOG"
ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v53.py >/dev/null 2>&1 &
WOK=0
for i in $(seq 1 240); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
  curl -s -o /dev/null -m 4 http://localhost:8000/health || true
done
if [ "$WOK" = "1" ]; then echo "WARMUP_OK $(date +%H:%M:%S)" | tee -a "$RLOG"; else echo "WARMUP_TIMEOUT" | tee -a "$RLOG"; fi
grep -E 'p10-131k|p11-227k' /root/dt_warmup.log | tail -2 | tee -a "$RLOG" || true

echo "== [lane $TAG] battery $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_probe2.py > "$OUT/battery.log" 2>&1
grep -E '^PROBE|^ACCEPT' "$OUT/battery.log" | tee -a "$RLOG"

echo "== [lane $TAG] ctxscan $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_ctxscan.py "$TAG" > "$OUT/ctxscan.out" 2>&1
grep CTXSCAN "$OUT/ctxscan.out" | tee -a "$RLOG"

echo "== [lane $TAG] deep 128k/227k $(date +%H:%M:%S)" | tee -a "$RLOG"
python3 /root/build/dt_ctxscan2.py "$TAG" "115000,205000" 256 > "$OUT/deep.out" 2>&1
grep CTXSCAN2 "$OUT/deep.out" | tee -a "$RLOG"

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
STALLS=$(docker exec lsv-test grep -c DFLASH_STALL /root/b_$TAG.log 2>/dev/null || echo 0)
STRIKES=$(docker exec lsv-test grep -c "v52m strike" /root/b_$TAG.log 2>/dev/null || echo 0)
F8=$(docker exec lsv-test grep -c "F8 guard" /root/b_$TAG.log 2>/dev/null || echo 0)
JIT=$(docker exec lsv-test grep -c jit_monitor /root/b_$TAG.log 2>/dev/null || echo 0)
ERRS=$(docker exec lsv-test grep -c -E "EngineDeadError|Traceback|CUDA assert|fatal" /root/b_$TAG.log 2>/dev/null || echo 0)
L2ARM=$(docker exec lsv-test grep -c "ctx-adaptive k armed" /root/b_$TAG.log 2>/dev/null || echo 0)
L2FIRE=$(docker exec lsv-test grep -c "ctx-adaptive k fired" /root/b_$TAG.log 2>/dev/null || echo 0)
P1=$(docker exec lsv-test grep -c "V125_P1 routes" /root/b_$TAG.log 2>/dev/null || echo 0)
TIER=$(docker exec lsv-test grep -c "V53_TQTIER" /root/b_$TAG.log 2>/dev/null || echo 0)
echo "STALLS=$STALLS STRIKES=$STRIKES F8=$F8 JIT=$JIT ERRS=$ERRS L2ARM=$L2ARM L2FIRE=$L2FIRE P1=$P1 TIER=$TIER" | tee -a "$RLOG"
docker exec lsv-test grep -m4 "ctx-adaptive k" /root/b_$TAG.log 2>/dev/null | tee -a "$RLOG" || true
docker exec lsv-test tail -3 /root/b_$TAG.log 2>/dev/null | tee -a "$RLOG" || true
echo "LANE_DONE $TAG $(date +%H:%M:%S)" | tee -a "$RLOG"
