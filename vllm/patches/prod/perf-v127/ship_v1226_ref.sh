#!/bin/bash
# mk_ship_v1226.py note: v126 ship = v1225 chain + dualbridge surface
# (rm patch_v126_dualbridge.py in lane-commit cleanup, DUAL BRIDGE marker
# on the fresh prod boot; battery completion literal stays V1225 caps —
# rename carryover in validate_v1226_run.sh; gates_v1226_lane.sh adds the
# v126 marker/.so-sha/P29-absence gates).
# ship_v1226.sh — ship production llm-scaler-exp:v1.2.26.
# Preconditions (asserted, not assumed):
#   - validate_v1226_run.sh finished V1225_VALIDATION_COMPLETE (full battery +
#     wedge suite + P23F quality gates green on v1.2.26-raw, incl. async +
#     barrier-off posture + opsall/C7 deltas)
#   - lane lsv-test UP on v1.2.26-raw
# Ship = v1222/v1223 precedent: CLEAN WARM RELAUNCH of the lane from -raw
# (fresh container, standard warm rounds only — no battery debris), docker
# commit -> llm-scaler-exp:v1.2.26, ENTIRE gate battery re-run on the
# committed image (gates_v1226_lane.sh with V123_ID/V123RAW_ID), fresh-boot
# verification on the committed image (v1224's prod-boot sed was a no-op —
# it re-booted -raw; FIXED here: sed actually swaps v1.2.26-raw -> v1.2.26),
# CC battery through litellm :4000 (Authorization: Bearer sk-dummy), watchdog
# repoint + systemctl RESTART (stale-parse trap).
set -u
L=/root/build/lce1/v1226_ship.log
{
echo "=== SHIP_V1226 start $(date +%F' '%T) ==="
FAIL=0

echo "--- preconditions ---"
# v126 note: validate_v1226_run.sh emits the V1225 caps literal (rename carryover)
grep -q V1225_VALIDATION_COMPLETE /root/build/lce1/v1226_validate_run.log 2>/dev/null || { echo "SHIP ABORT: validation not complete"; exit 1; }
curl -s -o /dev/null -m 4 http://localhost:8000/health && echo "lane UP" || { echo "SHIP ABORT: lane not up"; exit 1; }
docker ps --format '{{.Names}} {{.Image}}' | grep -q 'lsv-test.*v1.2.26-raw' || { echo "SHIP ABORT: lane not on v1.2.26-raw"; docker ps --format '{{.Names}} {{.Image}}' | head -3; exit 1; }
test -f /root/build/repro_bootV1226.sh || { echo "SHIP ABORT: repro_bootV1226.sh missing"; exit 1; }
RAWID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.26-raw)
echo "raw_id=$RAWID"

echo "--- 1. clean-warm relaunch from -raw (fresh container for the lane-commit) ---"
docker rm -f lsv-test >/dev/null 2>&1 || true
bash /root/build/repro_bootV1226.sh SHIPWARM > /root/build/lce1/boot_v1226_shipwarm.out 2>&1
grep -q 'HEALTH_OK' /root/build/lce1/boot_v1226_shipwarm.out || { echo "SHIP ABORT: shipwarm boot failed"; tail -20 /root/build/lce1/boot_v1226_shipwarm.out; exit 1; }
grep -q 'BOOT_SHIPWARM baked config verified' /root/build/lce1/boot_v1226_shipwarm.out || { echo "SHIP ABORT: boot config verify missing"; exit 1; }

echo "--- 2. standard warm rounds (mirror of bake warm; async posture) ---"
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 qwen3.8-27b-fp8 831 || { echo "SHIP ABORT: jitwarm failed"; exit 1; }
python3 /root/build/warm_ext_v1219.py http://127.0.0.1:8000 qwen3.8-27b-fp8 833 || { echo "SHIP ABORT: warm_ext failed"; exit 1; }
for shape in default topp topk all; do
  case $shape in
    default) BODY='{"model":"qwen3.8-27b-fp8","max_tokens":2000,"messages":[{"role":"user","content":"Write a 300-word story about a lighthouse."}]}';;
    topp)    BODY='{"model":"qwen3.8-27b-fp8","max_tokens":2000,"top_p":0.9,"messages":[{"role":"user","content":"Write a 300-word story about a harbor."}]}';;
    topk)    BODY='{"model":"qwen3.8-27b-fp8","max_tokens":2000,"top_k":40,"messages":[{"role":"user","content":"Write a 300-word story about a foghorn."}]}';;
    all)     BODY='{"model":"qwen3.8-27b-fp8","max_tokens":2000,"temperature":0.9,"top_p":0.95,"top_k":20,"messages":[{"role":"user","content":"Write a 300-word story about a buoy."}]}';;
  esac
  code=$(curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" -o /dev/null -w '%{http_code}')
  echo "sampler_$shape http=$code"
  [ "$code" = "200" ] || { echo "SHIP ABORT: sampler $shape failed"; exit 1; }
done
echo "--- warm-verify: zero JIT on replay (log-delta) ---"
OFF=$(docker exec lsv-test sh -c 'wc -l < /root/serve_full.log' | tr -d '[:space:]')
python3 /root/build/warm_ext_v1219.py http://127.0.0.1:8000 qwen3.8-27b-fp8 839 --verify || { echo "SHIP ABORT: warm_ext verify failed"; exit 1; }
JN=$(docker exec lsv-test sh -c "tail -n +${OFF} /root/serve_full.log | grep -cE 'expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel'" || true)
echo "shipwarm_new_jit=$JN (0 = fully warm)"
CNT=$(docker exec lsv-test sh -c 'ls /root/.triton/cache 2>/dev/null | wc -l')
echo "shipwarm_triton_cache_entries=$CNT"

echo "--- 3. lane-commit -> llm-scaler-exp:v1.2.26 ---"
docker exec lsv-test sh -c 'rm -f /root/probe_admission_v66.py /root/patch_barrier_v123_bake.py /root/patch_v125_c7_fp8state_raise.py /root/patch_v126_dualbridge.py 2>/dev/null; rm -rf /tmp/t.py 2>/dev/null' || true
docker commit lsv-test llm-scaler-exp:v1.2.26 || { echo "SHIP ABORT: commit failed"; exit 1; }
NEWID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.26)
echo "shipped_id=$NEWID"
docker images llm-scaler-exp:v1.2.26 --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'

echo "--- 4. full gate battery on the committed image ---"
V123_ID=$NEWID V123RAW_ID=$RAWID bash /root/build/gates_v1226_lane.sh > /root/build/lce1/v1226_gates_lane_run.out 2>&1
GRC=$?
tail -5 /root/build/lce1/v1226_gates_lane_run.out
[ "$GRC" = "0" ] || { echo "SHIP ABORT: gates failed rc=$GRC"; exit 1; }

echo "--- 5. fresh-boot verification on the committed production image ---"
# v1224 bugfix: the old sed 's/v1\.2\.23-raw/v1.2.24/' was a no-op on a script
# that already said v1.2.24-raw, so the "prod" boot re-ran -raw. This sed
# actually swaps the tag.
sed 's/llm-scaler-exp:v1\.2\.26-raw/llm-scaler-exp:v1.2.26/g' /root/build/repro_bootV1226.sh > /root/build/repro_bootV1226_prod.sh
chmod +x /root/build/repro_bootV1226_prod.sh
grep -n 'llm-scaler-exp:v1.2.26' /root/build/repro_bootV1226_prod.sh | head -3
docker rm -f lsv-test >/dev/null 2>&1 || true
bash /root/build/repro_bootV1226_prod.sh SHIP > /root/build/lce1/boot_v1226_prod.out 2>&1
grep -q 'HEALTH_OK' /root/build/lce1/boot_v1226_prod.out || { echo "SHIP ABORT: prod boot failed"; tail -20 /root/build/lce1/boot_v1226_prod.out; exit 1; }
sh /root/build/sanity_v1226.sh > /dev/null 2>&1
grep -q SANITY_V1225_EXTRA_DONE /root/build/_v1226_sanity.txt || { echo "SHIP ABORT: prod sanity incomplete"; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1226_sanity.txt || { echo "SHIP ABORT: prod admission not PASS"; exit 1; }
ASY=$(awk '/V123 ASYNC ENGAGED/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1226_sanity.txt)
[ "$ASY" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: async not engaged on prod boot"; exit 1; }
BARR=$(awk '/V123 BARRIER DEFAULT/{f=1;next} f&&/False/{print;exit}' /root/build/_v1226_sanity.txt)
[ "$BARR" = "False False 0" ] || { echo "SHIP ABORT: barrier posture wrong on prod boot got=$BARR"; exit 1; }
grep -q 'VLLM_RPC_TIMEOUT=60000' /root/build/_v1226_sanity.txt || { echo "SHIP ABORT: rpc timeout env missing on prod boot"; exit 1; }
FXP=$(docker exec lsv-test sh -c 'grep -c "v125 root fix" /root/serve_full.log' | tr -d '[:space:]')
[ "$FXP" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: opsall fix line missing on prod boot"; exit 1; }
DBP=$(docker exec lsv-test sh -c 'grep -c "llm-scaler v126 DUAL BRIDGE" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py' | tr -d '[:space:]')
[ "$DBP" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: dualbridge marker missing on prod boot got=$DBP"; exit 1; }
echo "prod fresh-boot sanity + admission + v123 posture + v125 opsall + v126 dualbridge PASS"

echo "--- 6. CC battery through litellm :4000 ---"
B=http://127.0.0.1:4000
# /health needs the master key (401 bare) and round-trips the backend lane
# (times out mid-boot) — /health/liveness is the correct up-probe.
LIT=$(curl -s -o /dev/null -m 5 -w '%{http_code}' $B/health/liveness 2>/dev/null)
echo "litellm_liveness=$LIT"
[ "$LIT" = "200" ] || { echo "SHIP ABORT: litellm :4000 down (restart litellm-proxy with -e LITELLM_MASTER_KEY=sk-dummy before re-running)"; exit 1; }
for t in t1 t2 t3; do
  case $t in
    t1) PROMPT='Use the Bash tool to run: echo ship-v1226-ok';;
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
cp /root/build/lane_watchdog.sh /root/build/lane_watchdog.sh.pre_v1226
sed -i 's|repro_bootV[0-9][0-9]*_prod.sh|repro_bootV1226_prod.sh|' /root/build/lane_watchdog.sh
grep -n 'repro_bootV1226_prod' /root/build/lane_watchdog.sh
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
docker images 'llm-scaler-exp*' --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E 'v1.2.2[56]'
if [ "$FAIL" != "0" ]; then
  echo "=== SHIP_V1226 COMPLETED WITH WARNINGS (CC battery) $(date +%T) ==="
else
  echo "=== SHIP_V1226 ALL GREEN $(date +%T) ==="
fi
} > "$L" 2>&1
tail -60 "$L"
grep -q 'SHIP_V1226 ALL GREEN' "$L" || exit 1
echo SHIP_V1226_DONE
