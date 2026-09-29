#!/bin/bash
# ship_v1225_finish.sh — finish the v1.2.25 ship: CC battery + watchdog
# (steps 6-7 of ship_v1225.sh; gates re-run and step 5 prod-boot already
# PASSED in ship_v1225_resume.sh — log /root/build/lce1/v1225_ship_resume.log).
# Fix vs ship_v1225.sh: the litellm up-probe now uses /health/liveness —
# the detailed /health endpoint requires the master key (401 without) and
# round-trips the backend lane (times out at 5s mid-boot). The CC battery
# requests themselves carry the auth and prove the path end-to-end.
set -u
L=/root/build/lce1/v1225_ship_finish.log
{
echo "=== SHIP_V1225_FINISH start $(date +%F' '%T) ==="
FAIL=0

echo "--- preconditions ---"
grep -q 'prod fresh-boot sanity + admission + v123 posture + v125 opsall PASS' /root/build/lce1/v1225_ship_resume.log || { echo "ABORT: step-5 prod boot not green"; exit 1; }
curl -s -o /dev/null -m 5 http://localhost:8000/health && echo "lane UP" || { echo "ABORT: lane not up"; exit 1; }
docker ps --format '{{.Names}} {{.Image}}' | grep -q 'lsv-test.*v1.2.25' && echo "lane on v1.2.25 (committed image)" || { echo "ABORT: lane not on committed v1.2.25"; exit 1; }

echo "--- 6. CC battery through litellm :4000 ---"
B=http://127.0.0.1:4000
LIT=$(curl -s -o /dev/null -m 5 -w '%{http_code}' $B/health/liveness 2>/dev/null)
echo "litellm_liveness=$LIT"
[ "$LIT" = "200" ] || { echo "SHIP ABORT: litellm :4000 down (restart litellm-proxy with -e LITELLM_MASTER_KEY=sk-dummy before re-running)"; exit 1; }
for t in t1 t2 t3; do
  case $t in
    t1) PROMPT='Use the Bash tool to run: echo ship-v1225-ok';;
    t2) PROMPT='Read the file /etc/hostname and report its contents using the Read tool';;
    t3) PROMPT='What is 17*23? Answer with just the number.';;
  esac
  code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
    -H "Authorization: Bearer sk-dummy" \
    -d "{\"model\":\"qwen3.8-27b-fp8-opus\",\"max_tokens\":600,\"messages\":[{\"role\":\"user\",\"content\":$(
      python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT")}]}" \
    -o /tmp/cc_$t.json -w '%{http_code}')
  TUSE=$(python3 -c "import json; d=json.load(open('/tmp/cc_$t.json')); print('tool_use' if any(b.get('type')=='tool_use' for b in d.get('content',[])) else ('text' if d.get('content') else 'EMPTY'))" 2>/dev/null)
  echo "cc_$t http=$code blocks=$TUSE"
  [ "$code" = "200" ] || FAIL=1
done
TH=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
  -H "Authorization: Bearer sk-dummy" \
  -d '{"model":"qwen3.8-27b-fp8-opus","max_tokens":400,"thinking":{"type":"enabled","budget_tokens":2000},"messages":[{"role":"user","content":"Think step by step: what is the capital of Finland?"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); types=[b.get("type") for b in d.get("content",[])]; print("thinking" if "thinking" in types else "NO_THINKING")' 2>/dev/null)
echo "cc_thinking=$TH"
[ "$TH" = "thinking" ] || FAIL=1

echo "--- 7. watchdog repoint + re-arm (systemctl RESTART: stale-parse trap) ---"
cp /root/build/lane_watchdog.sh /root/build/lane_watchdog.sh.pre_v1225
sed -i 's|repro_bootV[0-9][0-9]*_prod.sh|repro_bootV1225_prod.sh|' /root/build/lane_watchdog.sh
grep -n 'repro_bootV1225_prod' /root/build/lane_watchdog.sh
rm -f /root/build/lane_watchdog.paused
systemctl restart lane-watchdog.service
# was disabled before the pre-validation host reboot — restore boot posture
systemctl enable lane-watchdog.service >/dev/null 2>&1 || true
sleep 5
systemctl is-active lane-watchdog.service || { echo "SHIP ABORT: watchdog service failed to restart"; exit 1; }
WLN=$(wc -l < /root/build/lane_watchdog.log)
sleep 220
tail -n +$((WLN+1)) /root/build/lane_watchdog.log
if tail -n +$((WLN+1)) /root/build/lane_watchdog.log | grep -q 'alive armed'; then
  echo "watchdog ARMED (fresh armed-line seen post restart)"
else
  echo "SHIP WARN: no fresh armed-line yet (10-cycle = 100s cadence; check log)"
fi

echo "--- summary ---"
docker images 'llm-scaler-exp*' --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E 'v1.2.2[45]'
if [ "$FAIL" != "0" ]; then
  echo "=== SHIP_V1225 COMPLETED WITH WARNINGS (CC battery) $(date +%T) ==="
else
  echo "=== SHIP_V1225 ALL GREEN $(date +%T) ==="
fi
} > "$L" 2>&1
tail -40 "$L"
grep -q 'SHIP_V1225 ALL GREEN' "$L" || exit 1
echo SHIP_V1225_DONE
