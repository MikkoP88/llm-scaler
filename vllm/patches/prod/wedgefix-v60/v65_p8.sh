#!/bin/bash
# v65 Phase-1 P1b: TRITON_ATTN measurement boot + solo cold (then revert env)
B=/root/build
OUT=$B/v65_p8_out.txt
{
echo "=== add TRITON_ATTN env ==="
docker exec lsv-test grep -q "VLLM_ATTENTION_BACKEND" /root/serve_user.sh || \
  docker exec lsv-test sed -i '1i export VLLM_ATTENTION_BACKEND=TRITON_ATTN' /root/serve_user.sh
docker exec lsv-test grep -n "VLLM_ATTENTION_BACKEND" /root/serve_user.sh | head -2
docker exec lsv-test pkill -f "vllm serve" || true
sleep 5
docker exec -d lsv-test bash /root/serve_user.sh
echo "relaunched, waiting health..."
CODE=000
for i in $(seq 1 60); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  [ "$CODE" = "200" ] && break
  sleep 10
done
echo "HEALTH=$CODE"
if [ "$CODE" != "200" ]; then
  echo "BOOT_FAILED — backend selection + tail:"
  docker exec lsv-test sh -c "grep -iE 'attention|triton|error|Traceback' /root/serve_full.log | tail -25"
  echo "REVERTING env and relaunching stock..."
  docker exec lsv-test sed -i '/VLLM_ATTENTION_BACKEND/d' /root/serve_user.sh
  docker exec lsv-test pkill -f "vllm serve" || true
  sleep 5
  docker exec -d lsv-test bash /root/serve_user.sh
  for i in $(seq 1 60); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
    [ "$CODE" = "200" ] && break
    sleep 10
  done
  echo "RESTOCK_HEALTH=$CODE"
  echo "P8_ABORTED_BOOT_FAIL"
  exit 1
fi
echo
echo "=== backend confirmation ==="
docker exec lsv-test sh -c "grep -iE 'Using .*backend|attention backend' /root/serve_full.log | tail -6"
echo
echo "=== solo cold seed 211 (TRITON_ATTN) ==="
date +%H:%M:%S
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 211 > $B/v65_p8_solo.txt 2>&1
date +%H:%M:%S
tail -4 $B/v65_p8_solo.txt
echo
echo "=== GDN cadence under triton (step walls via V65_GDN) ==="
docker exec lsv-test sh -c "grep 'V65_GDN' /root/serve_full.log | tail -20"
echo
echo "=== revert env + relaunch stock ==="
docker exec lsv-test sed -i '/VLLM_ATTENTION_BACKEND/d' /root/serve_user.sh
docker exec lsv-test pkill -f "vllm serve" || true
sleep 5
docker exec -d lsv-test bash /root/serve_user.sh
CODE=000
for i in $(seq 1 60); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  [ "$CODE" = "200" ] && break
  sleep 10
done
echo "RESTOCK_HEALTH=$CODE"
docker exec lsv-test sh -c "grep -c VLLM_ATTENTION_BACKEND /root/serve_user.sh" || true
echo "P8_DONE"
} > $OUT 2>&1
echo "SCRIPT_EXIT=$?"
