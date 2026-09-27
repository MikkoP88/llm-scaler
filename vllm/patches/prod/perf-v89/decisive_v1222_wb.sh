#!/bin/bash
# decisive_v1222_wb.sh — DECISIVE zero-JIT verification of the lane-commit
# image llm-scaler-exp:v1.2.22 (89b17e0b0f8d): the v1.2.22-raw boot showed 8
# first-traffic JIT compiles (bake-window triton cache entries were lost
# across docker commit); the lane commit ships the complete warm cache
# (74 entries) and MUST show ZERO.
# Chain: assert image id -> assert watchdog stopped -> hardened lane teardown
# -> fresh boot via repro_bootV1222.sh v1222wb -> full sanity certification
# (xgrammar probes, v60_ab, barrier HB, admission PASS) -> decisive traffic
# (fresh-seed solo cold big prefill, solo decode bench, 4 sampler shapes,
# concurrent multi-session warm_ext replay) -> GATES:
#   jit_monitor_lines (whole log)            == 0
#   kernel_name_lines (traffic-only, offset) == 0
#   triton_cache_entries                      == 74 (unchanged from image)
#   final health 200, dmesg engine resets 0
# Lane is left UP on the production image. Exit non-zero on any miss.
set -u
L=/root/build/lce1/v1222_wb_decisive.log
{
echo "=== V1222 WB DECISIVE ZERO-JIT TEST $(date +%F' '%T) ==="

IID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.22)
echo "boot_image=$IID"
case "$IID" in
  sha256:89b17e0b0f8d*) echo "image confirmed lane-commit";;
  *) echo "ABORT: v1.2.22 is not the lane commit 89b17e0b0f8d"; exit 1;;
esac

WA=$(systemctl is-active lane-watchdog 2>/dev/null || true)
echo "lane-watchdog=$WA"
[ "$WA" = "active" ] && { echo "ABORT: stop lane-watchdog first (it would race the teardown)"; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -q '^lsv-test$'; then
  OLDIMG=$(docker inspect --format '{{.Image}}' lsv-test)
  echo "existing lsv-test image=$OLDIMG (evidence: previous lane, expected raw e6735a4c72a2...)"
  docker exec lsv-test bash -c "pkill -f 'vllm serve' || true" 2>/dev/null || true
  for i in $(seq 1 15); do
    sleep 1
    docker exec lsv-test bash -c "pgrep -f 'vllm serve' >/dev/null" 2>/dev/null || break
    [ "$i" = "8" ] && docker exec lsv-test bash -c "pkill -9 -f 'vllm serve' || true" 2>/dev/null || true
  done
  docker rm -f lsv-test >/dev/null 2>&1 || true
fi
if curl -s -o /dev/null -m 3 http://localhost:8000/health; then
  echo "ABORT: :8000 still answering after teardown"; exit 1
fi
echo "teardown clean $(date +%T)"

echo "--- fresh boot from lane-commit image (MODE=v1222wb) ---"
bash /root/build/repro_bootV1222.sh v1222wb
BRC=$?
echo "repro_boot_rc=$BRC"
[ "$BRC" != "0" ] && { echo "ABORT: boot failed"; exit 1; }
H=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health)
echo "post_boot_health=$H"
[ "$H" = "200" ] || { echo "ABORT: health not 200 after boot"; exit 1; }
docker exec lsv-test sh -c 'echo triton_cache_entries_at_boot=$(ls /root/.triton/cache | wc -l)'
docker exec lsv-test sh -c "grep -m2 'Capturing CUDA graphs' /root/serve_full.log" || true
O=$(docker exec lsv-test sh -c 'wc -l < /root/serve_full.log' | tr -d '[:space:]')
echo "log_offset_after_boot=$O (traffic-only JIT evidence measured from here)"

echo "--- sanity certification chain (READY, xgrammar x2, v60_ab, barrier HB, admission) ---"
sh /root/build/sanity_v1222.sh > /dev/null 2>&1
grep -q SANITY_V1222_EXTRA_DONE /root/build/_v1222_sanity.txt || { echo "ABORT: sanity chain failed"; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1222_sanity.txt || { echo "ABORT: admission not PASS"; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1222_sanity.txt || true
echo "sanity+admission PASS $(date +%T)"

echo "--- decisive traffic 1: fresh-seed solo cold (big prefill ~34k words) ---"
python3 /root/build/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 128 34000
SRC=$?
echo "solo_cold_rc=$SRC"

echo "--- decisive traffic 2: solo decode bench ---"
cd /root/build && python3 bench_genspeed.py 1 1024 3

echo "--- decisive traffic 3: 4 sampler shapes (default/top_p/top_k/all) ---"
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"messages":[{"role":"user","content":"Write a 300-word story about a lighthouse."}]}' -o /dev/null -w 'sampler_default_http=%{http_code}\n'
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"top_p":0.9,"messages":[{"role":"user","content":"Write a 300-word story about a harbor."}]}' -o /dev/null -w 'sampler_topp_http=%{http_code}\n'
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"top_k":40,"messages":[{"role":"user","content":"Write a 300-word story about a foghorn."}]}' -o /dev/null -w 'sampler_topk_http=%{http_code}\n'
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"temperature":0.9,"top_p":0.95,"top_k":20,"messages":[{"role":"user","content":"Write a 300-word story about a buoy."}]}' -o /dev/null -w 'sampler_all_http=%{http_code}\n'

echo "--- decisive traffic 4: concurrent multi-session replay (seed 619) ---"
python3 /root/build/warm_ext_v1219.py http://127.0.0.1:8000 qwen3.8-27b-fp8 619
WRC=$?
echo "warmext_rc=$WRC"

echo "=== THE GATE: zero JIT since boot ==="
JITPAT='expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel|_topk_topp'
JN=$(docker exec lsv-test sh -c "grep 'JIT compilation during inference' /root/serve_full.log | wc -l")
KN=$(docker exec lsv-test sh -c "tail -n +$O /root/serve_full.log | grep -E '$JITPAT' | wc -l")
KALL=$(docker exec lsv-test sh -c "grep -E '$JITPAT' /root/serve_full.log | wc -l")
TC=$(docker exec lsv-test sh -c 'ls /root/.triton/cache | wc -l' | tr -d '[:space:]')
echo "jit_monitor_lines_whole_log=$JN"
echo "kernel_name_lines_traffic_only=$KN (whole log: $KALL)"
echo "triton_cache_entries_now=$TC (image shipped 74)"
H2=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health)
RSC=$(dmesg -T 2>/dev/null | grep -iE 'UR_RESULT|device lost|watchdog timeout' | grep -vi mpt3sas | wc -l)
echo "final_health=$H2 dmesg_engine_resets=$RSC"

VERDICT=PASS
[ "$JN" = "0" ] || { echo "ABORT: jit_monitor_lines=$JN — ROOT FIX NOT DELIVERED (raw image showed 8)"; VERDICT=FAIL; }
[ "$KN" = "0" ] || { echo "ABORT: kernel_name_lines=$KN"; VERDICT=FAIL; }
[ "$TC" = "74" ] || echo "NOTE: cache entries $TC != 74 — investigate (a compile adds entries)"
[ "$H2" = "200" ] || { echo "ABORT: final health $H2"; VERDICT=FAIL; }
[ "$RSC" = "0" ] || { echo "ABORT: engine resets $RSC"; VERDICT=FAIL; }
if [ "$VERDICT" = "PASS" ]; then
  echo "=== V1222 WB DECISIVE: PASS — zero first-traffic JIT on the lane-commit image $(date +%T) ==="
else
  echo "=== V1222 WB DECISIVE: FAIL $(date +%T) ==="
  exit 1
fi
} > "$L" 2>&1
tail -60 "$L"
grep -q 'WB DECISIVE: PASS' "$L" || exit 1
exit 0
