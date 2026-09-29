#!/bin/bash
# ship_v1224_cont.sh — v1.2.24 ship CONTINUATION after the 16:52 gates abort.
# Steps 1-3 of ship_v1224.sh completed cleanly (shipwarm boot, warm rounds
# jit-delta 0 / cache 72, docker commit -> llm-scaler-exp:v1.2.24 =
# sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e).
# The abort was gates-script-side (v124 block misplacement + SP clobber),
# fixed by fix_gates_v1224.py — the image was never in question (image-id,
# raw-id, wheel, serve-config, v123-posture, pedigree, stamps, cache gates
# all passed in the same run). This script re-runs steps 4-7 verbatim, with
# ONE correction: the _prod derivation sed now targets v1.2.24-raw (the
# inherited 'v1.2.23-raw' pattern no longer matches repro_bootV1224.sh and
# would have silently produced a -raw-image prod boot).
set -u
L=/root/build/lce1/v1224_ship.log
FAIL=0
NEWID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.24)
RAWID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.24-raw)
{
echo "=== SHIP_V1224_CONT start $(date +%F' '%T) shipped_id=$NEWID ==="

echo "--- 4. full gate battery on the committed image (gates fixed: v124 block relocated + SPR) ---"
V123_ID=$NEWID V123RAW_ID=$RAWID bash /root/build/gates_v1224_lane.sh > /root/build/lce1/v1224_gates_lane_run.out 2>&1
GRC=$?
tail -5 /root/build/lce1/v1224_gates_lane_run.out
[ "$GRC" = "0" ] || { echo "SHIP ABORT: gates failed rc=$GRC"; exit 1; }

echo "--- 5. fresh-boot verification on the committed production image ---"
sed 's/v1\.2\.24-raw/v1.2.24/' /root/build/repro_bootV1224.sh > /root/build/repro_bootV1224_prod.sh
chmod +x /root/build/repro_bootV1224_prod.sh
grep -n 'llm-scaler-exp:v1.2.24' /root/build/repro_bootV1224_prod.sh | head -2
docker rm -f lsv-test >/dev/null 2>&1 || true
bash /root/build/repro_bootV1224_prod.sh SHIP > /root/build/lce1/boot_v1224_prod.out 2>&1
grep -q 'HEALTH_OK' /root/build/lce1/boot_v1224_prod.out || { echo "SHIP ABORT: prod boot failed"; tail -20 /root/build/lce1/boot_v1224_prod.out; exit 1; }
sh /root/build/sanity_v1224.sh > /dev/null 2>&1
grep -q SANITY_V1224_EXTRA_DONE /root/build/_v1224_sanity.txt || { echo "SHIP ABORT: prod sanity incomplete"; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1224_sanity.txt || { echo "SHIP ABORT: prod admission not PASS"; exit 1; }
ASY=$(awk '/V123 ASYNC ENGAGED/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1224_sanity.txt)
[ "$ASY" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: async not engaged on prod boot"; exit 1; }
BARR=$(awk '/V123 BARRIER DEFAULT/{f=1;next} f&&/False/{print;exit}' /root/build/_v1224_sanity.txt)
[ "$BARR" = "False False 0" ] || { echo "SHIP ABORT: barrier posture wrong on prod boot got=$BARR"; exit 1; }
grep -q 'VLLM_RPC_TIMEOUT=60000' /root/build/_v1224_sanity.txt || { echo "SHIP ABORT: rpc timeout env missing on prod boot"; exit 1; }
echo "prod fresh-boot sanity + admission + v123 posture PASS"

echo "--- 6. CC battery through litellm :4000 ---"
B=http://127.0.0.1:4000
for t in t1 t2 t3; do
  case $t in
    t1) PROMPT='Use the Bash tool to run: echo ship-v1224-ok';;
    t2) PROMPT='Read the file /etc/hostname and report its contents using the Read tool';;
    t3) PROMPT='What is 17*23? Answer with just the number.';;
  esac
  code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
    -H "Authorization: Bearer sk-dummy" \
    -d "{\"model\":\"qwen3.8-27b-fp8-opus\",\"max_tokens\":600,\"messages\":[{\"role\":\"user\",\"content\":$(
      python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT")}]}\"" \
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

echo "--- 7. watchdog repoint + re-arm ---"
cp /root/build/lane_watchdog.sh /root/build/lane_watchdog.sh.pre_v1224
sed -i 's|repro_bootV[0-9][0-9]*_prod.sh|repro_bootV1224_prod.sh|' /root/build/lane_watchdog.sh
grep -n 'repro_bootV1224_prod' /root/build/lane_watchdog.sh
rm -f /root/build/lane_watchdog.paused
systemctl is-active lane-watchdog.service || systemctl start lane-watchdog.service
WLN=$(wc -l < /root/build/lane_watchdog.log)
sleep 220
tail -n +$((WLN+1)) /root/build/lane_watchdog.log
if tail -n +$((WLN+1)) /root/build/lane_watchdog.log | grep -q 'alive armed'; then
  echo "watchdog ARMED (fresh armed-line seen post re-arm)"
else
  echo "SHIP WARN: no fresh armed-line yet (10-cycle = 100s cadence; check log)"
fi

echo "--- summary ---"
docker images 'llm-scaler-exp*' --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E 'v1.2.2[234]'
if [ "$FAIL" != "0" ]; then
  echo "=== SHIP_V1224 COMPLETED WITH WARNINGS (CC battery) $(date +%T) ==="
else
  echo "=== SHIP_V1224 ALL GREEN $(date +%T) ==="
fi
} >> "$L" 2>&1
tail -60 "$L"
grep -q 'SHIP_V1224 ALL GREEN' "$L" || exit 1
echo SHIP_V1224_CONT_DONE
