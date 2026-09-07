#!/bin/bash
# dt_vlane.sh — llm-scaler v52 Phase V lane runner (SPLITS validation matrix).
# One lane = boot + warmup + battery + cargame(single,concurrent) + loop8
# (optional) + ctxscan, all into /root/build/vlane_<tag>/, then a summary.
# Usage: dt_vlane.sh <tag> <specjson|nospec> <extraenv> <extraflags> [noloop8]
# Image fixed: llm-scaler-exp:v1.2.5t2 (candidate-final content, md5-gated
# identical to what v1.2.5 will bake: TQ MQ default-32 + G2 env-only gate).
set -u
TAG="$1"; SPEC="$2"; ENVV="$3"; FLAGS="$4"; NOLOOP="${5:-}"
IMG='llm-scaler-exp:v1.2.5t2'
OUT="/root/build/vlane_${TAG}"
mkdir -p "$OUT"
LOG="$TAG"

echo "== [vlane $TAG] boot $(date +%H:%M:%S)"
if ! bash /root/build/bootp.sh "$SPEC" "$ENVV" "$LOG" "$FLAGS" "$IMG" \
    > "$OUT/boot.out" 2>&1; then
  echo "VLANE_BOOT_FAIL $TAG"; tail -25 "$OUT/boot.out"; exit 1
fi
echo "== [vlane $TAG] warmup start $(date +%H:%M:%S)"
ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
for i in $(seq 1 120); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  [ "$ND" -gt "$ND0" ] && break
done
if [ "$ND" -le "$ND0" ]; then
  echo "VLANE_WARMUP_TIMEOUT $TAG"; exit 2
fi
echo "== [vlane $TAG] warmup done $(date +%H:%M:%S)"

echo "== [vlane $TAG] battery"
python3 /root/build/dt_probe2.py > "$OUT/battery.log" 2>&1

echo "== [vlane $TAG] cargame single+concurrent"
python3 /root/build/dt_cargame.py single > "$OUT/cargame_single.log" 2>&1
python3 /root/build/dt_cargame.py concurrent > "$OUT/cargame_conc.log" 2>&1

if [ "$NOLOOP" != "noloop8" ]; then
  echo "== [vlane $TAG] loop8 (long) $(date +%H:%M:%S)"
  python3 /root/build/dt_loop8.py "v_$TAG" > "$OUT/loop8.out" 2>&1
  echo "== [vlane $TAG] loop8 done $(date +%H:%M:%S)"
fi

echo "== [vlane $TAG] ctxscan"
python3 /root/build/dt_ctxscan.py "$TAG" > "$OUT/ctxscan.out" 2>&1

STALLS=$(docker exec lsv-test grep -c DFLASH_STALL "/root/${LOG}.log" 2>/dev/null || echo '?')
{
  echo "VLANE_SUMMARY $TAG stalls=$STALLS"
  grep -E '^PROBE' "$OUT/battery.log" | sed 's/^/  /'
  grep -E 'ACCEPT' "$OUT/battery.log" | sed 's/^/  /'
  cat "$OUT/cargame_single.log" "$OUT/cargame_conc.log" | grep -E '^\[' | sed 's/^/  /'
  [ -f "$OUT/loop8.out" ] && grep -E '^P[0-9]|DT_LOOP8' "$OUT/loop8.out" | sed 's/^/  /'
  grep CTXSCAN "$OUT/ctxscan.out" | sed 's/^/  /'
} | tee "$OUT/summary.txt"
echo "VLANE_DONE $TAG"
