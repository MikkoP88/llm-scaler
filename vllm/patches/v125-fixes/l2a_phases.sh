#!/bin/bash
# l2a_phases.sh — resume Lane A phases on the LIVE lsv-test server
# (v1.2.6t1, mtp4+tq4nc, VLLM_SPEC_CTX_K=65536:1; boot done 13:09).
# Runs: warmup v53 (fixed p10/p11) -> battery -> ctxscan -> deep ->
# conc8/concmix/conc65k -> health tail. Appends to lane report.
set -u
TAG=l2a_mtp4_tq4
OUT=/root/build/bench126/lane_${TAG}
RLOG=$OUT/report.txt

echo "== [lane $TAG] warmup v53 RESUME (calibrated p10/p11) $(date +%H:%M:%S)" | tee -a "$RLOG"
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
grep -E 'p10-|p11-' /root/dt_warmup.log | tail -2 | tee -a "$RLOG" || true

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
F8=$(docker exec lsv-test grep -c "F8 guard fire" /root/b_$TAG.log 2>/dev/null || echo 0)
JIT=$(docker exec lsv-test grep -c jit_monitor /root/b_$TAG.log 2>/dev/null || echo 0)
ERRS=$(docker exec lsv-test grep -c -E "EngineDeadError|CUDA assert|fatal" /root/b_$TAG.log 2>/dev/null || echo 0)
L2ARM=$(docker exec lsv-test grep -c "ctx-adaptive k armed" /root/b_$TAG.log 2>/dev/null || echo 0)
L2FIRE=$(docker exec lsv-test grep -c "ctx-adaptive k fired" /root/b_$TAG.log 2>/dev/null || echo 0)
echo "STALLS=$STALLS STRIKES=$STRIKES F8=$F8 JIT=$JIT ERRS=$ERRS L2ARM=$L2ARM L2FIRE=$L2FIRE" | tee -a "$RLOG"
docker exec lsv-test grep -m4 "ctx-adaptive k" /root/b_$TAG.log 2>/dev/null | tee -a "$RLOG" || true
echo "LANE_DONE $TAG $(date +%H:%M:%S)" | tee -a "$RLOG"
