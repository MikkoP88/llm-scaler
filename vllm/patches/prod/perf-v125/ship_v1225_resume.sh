#!/bin/bash
# ship_v1225_resume.sh — resume the v1.2.25 ship AFTER the gates-quoting fix.
# Background: ship_v1225.sh aborted at step 4 (gates) with exactly ONE
# failure — v125_opsall_none2all — caused by a QUOTING BUG in the gate
# itself ('\\"' inside single quotes = literal backslashes in the -F
# pattern), not by the image. The lane-commit (llm-scaler-exp:v1.2.25 =
# sha256:3f3c91637692..., from the fully-warmed SHIPWARM lane,
# shipwarm_new_jit=0, cache 79) is VALID and KEPT — no second commit.
# This script re-runs the FIXED gates against that committed image, then
# executes ship steps 5-7 verbatim (prod fresh-boot on the COMMITTED image,
# sanity gates, CC battery via :4000, watchdog repoint + restart + enable).
set -u
L=/root/build/lce1/v1225_ship_resume.log
{
echo "=== SHIP_V1225_RESUME start $(date +%F' '%T) ==="
FAIL=0

echo "--- 4r. full gate battery on the committed image (fixed gates) ---"
RAWID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.25-raw)
NEWID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.25)
echo "raw_id=$RAWID shipped_id=$NEWID"
V123_ID=$NEWID V123RAW_ID=$RAWID bash /root/build/gates_v1225_lane.sh > /root/build/lce1/v1225_gates_lane_run2.out 2>&1
GRC=$?
tail -5 /root/build/lce1/v1225_gates_lane_run2.out
[ "$GRC" = "0" ] || { echo "SHIP ABORT: gates failed rc=$GRC"; exit 1; }
grep -c 'GATE-OK' /root/build/lce1/v1225_gates_lane.log
grep 'GATE-FAIL' /root/build/lce1/v1225_gates_lane.log || echo "no GATE-FAIL lines"

echo "--- 5. fresh-boot verification on the committed production image ---"
# v1224 bugfix: the old sed 's/v1\.2\.23-raw/v1.2.24/' was a no-op on a script
# that already said v1.2.24-raw, so the "prod" boot re-ran -raw. This sed
# actually swaps the tag.
sed 's/llm-scaler-exp:v1\.2\.25-raw/llm-scaler-exp:v1.2.25/g' /root/build/repro_bootV1225.sh > /root/build/repro_bootV1225_prod.sh
chmod +x /root/build/repro_bootV1225_prod.sh
grep -n 'llm-scaler-exp:v1.2.25' /root/build/repro_bootV1225_prod.sh | head -3
docker rm -f lsv-test >/dev/null 2>&1 || true
bash /root/build/repro_bootV1225_prod.sh SHIP > /root/build/lce1/boot_v1225_prod.out 2>&1
grep -q 'HEALTH_OK' /root/build/lce1/boot_v1225_prod.out || { echo "SHIP ABORT: prod boot failed"; tail -20 /root/build/lce1/boot_v1225_prod.out; exit 1; }
sh /root/build/sanity_v1225.sh > /dev/null 2>&1
grep -q SANITY_V1225_EXTRA_DONE /root/build/_v1225_sanity.txt || { echo "SHIP ABORT: prod sanity incomplete"; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1225_sanity.txt || { echo "SHIP ABORT: prod admission not PASS"; exit 1; }
ASY=$(awk '/V123 ASYNC ENGAGED/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1225_sanity.txt)
[ "$ASY" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: async not engaged on prod boot"; exit 1; }
BARR=$(awk '/V123 BARRIER DEFAULT/{f=1;next} f&&/False/{print;exit}' /root/build/_v1225_sanity.txt)
[ "$BARR" = "False False 0" ] || { echo "SHIP ABORT: barrier posture wrong on prod boot got=$BARR"; exit 1; }
grep -q 'VLLM_RPC_TIMEOUT=60000' /root/build/_v1225_sanity.txt || { echo "SHIP ABORT: rpc timeout env missing on prod boot"; exit 1; }
FXP=$(docker exec lsv-test sh -c 'grep -c "v125 root fix" /root/serve_full.log' | tr -d '[:space:]')
[ "$FXP" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: opsall fix line missing on prod boot"; exit 1; }
echo "prod fresh-boot sanity + admission + v123 posture + v125 opsall PASS"

echo "--- 6. CC battery through litellm :4000 ---"
B=http://127.0.0.1:4000
LIT=$(curl -s -o /dev/null -m 5 -w '%{http_code}' $B/health 2>/dev/null)
echo "litellm_health=$LIT"
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
tail -60 "$L"
grep -q 'SHIP_V1225 ALL GREEN' "$L" || exit 1
echo SHIP_V1225_DONE
