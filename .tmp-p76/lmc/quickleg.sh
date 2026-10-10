#!/bin/bash
# quickleg.sh DTYPE MODE N [PFX] — lean boot + nonce quality probe.
#   DTYPE fp8_e4m3|float16  MODE on|off  N prompts  PFX off = disable prefix caching
# Edits serve script (line 46 dtype, line 57 connector), restarts engine,
# health-waits, runs nonce_probe.py N, banks output + mechanism counts.
set -u
DTYPE=$1; MODE=$2; N=${3:-20}; PFX=${4:-on}; DBG=${5:-on}
SP=/root/serve_lmc_v133.sh
B=/root/build
TAG="q_${DTYPE}_${MODE}_pfx${PFX}_dbg${DBG}"

echo "=== QUICKLEG START tag=$TAG n=$N $(date -u +%H:%M:%S)"

python3 - "$SP" "$DTYPE" "$MODE" "$PFX" "$DBG" <<'PYEOF'
import sys
sp, dtype, mode, pfx, dbg = sys.argv[1:6]
lines = open(sp).read().split("\n")
if dbg == "off":
    lines = [l for l in lines
             if not l.strip().startswith("export LMC_DBG_")]
for i, l in enumerate(lines):
    if "--mamba-ssm-cache-dtype" in l:
        lines[i] = "  --mamba-ssm-cache-dtype %s \\" % dtype
        break
else:
    raise SystemExit("MAMBA_LINE_NOT_FOUND")
if mode == "off":
    for i, l in enumerate(lines):
        if "kv-transfer-config" in l:
            del lines[i]
            break
for i, l in enumerate(lines):
    if "--enable-prefix-caching" in l or "--no-enable-prefix-caching" in l:
        want = "--no-enable-prefix-caching" if pfx == "off" else \
               "--enable-prefix-caching"
        lines[i] = l.replace("--no-enable-prefix-caching", want) \
                    .replace("--enable-prefix-caching", want)
        break
open(sp, "w").write("\n".join(lines))
print("EDIT_OK dtype=%s" % dtype)
print("EDIT connector=%s pfx=%s dbg=%s" % (
    any("kv-transfer-config" in l for l in lines), pfx, dbg))
PYEOF
[ $? -ne 0 ] && { echo EDIT_ABORT; exit 1; }

docker cp $SP lsv-test:$SP

PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  for p in $PIDS; do docker exec lsv-test kill -TERM "$p" 2>/dev/null || true; done
  sleep 14
fi
PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "TERM_LEFTOVERS=$PIDS"
  for p in $PIDS; do docker exec lsv-test kill -9 "$p" 2>/dev/null || true; done
  sleep 6
fi
docker exec lsv-test rm -f /tmp/lmc_off /tmp/lmc_snap_off /tmp/lmc_store_off

docker exec -d lsv-test bash $SP
ok=0
for i in $(seq 1 60); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; docker exec lsv-test tail -15 /root/serve_full.log; exit 1; }

echo "--- nonce probe (warm-up 2 + record N):"
python3 $B/nonce_probe.py 2 | tail -n 3
python3 $B/nonce_probe.py $N > $B/nonce_${TAG}.out 2>&1
tail -n 4 $B/nonce_${TAG}.out
grep -c DEGENERATE $B/nonce_${TAG}.out || true
echo "--- mechanisms:"
for pat in "Double unpin" "LMCNANDETECT" "Traceback"; do
  n=$(docker exec lsv-test sh -c "grep -ac \"$pat\" /root/serve_full.log" 2>/dev/null || echo 0)
  echo "COUNT[$pat]=$n"
done
echo "--- health:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== QUICKLEG DONE tag=$TAG $(date -u +%H:%M:%S)"
