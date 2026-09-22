#!/bin/bash
# v65 Phase-1 P1a: apply layer-type timer probe + instrumented solo cold (seed 210)
B=/root/build
OUT=$B/v65_lt_out.txt
{
echo "=== copy + apply patcher ==="
docker cp $B/patch_v65_lt.py lsv-test:/root/patch_v65_lt.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v65_lt.py --apply
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v65_lt.py --check
echo
echo "=== env + relaunch ==="
docker exec lsv-test grep -c VLLM_V65_LAYER_LOG /root/serve_user.sh || \
  docker exec lsv-test sed -i '1i export VLLM_V65_LAYER_LOG=1' /root/serve_user.sh
docker exec lsv-test grep -n VLLM_V65_LAYER_LOG /root/serve_user.sh | head -2
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
  echo "ABORT_NO_HEALTH"
  docker exec lsv-test sh -c "tail -40 /root/serve_full.log"
  exit 1
fi
echo
echo "=== probe activation check (V65_LT lines in log) ==="
docker exec lsv-test sh -c "grep -E 'V65_LT|V65_ATTN|V65_GDN' /root/serve_full.log | tail -5"
echo
echo "=== solo cold seed 210 ==="
date +%H:%M:%S
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 210 > $B/v65_lt_solo.txt 2>&1
date +%H:%M:%S
tail -6 $B/v65_lt_solo.txt
echo
echo "=== layer lines ==="
docker exec lsv-test sh -c "grep -E 'V65_ATTN|V65_GDN|V65_LT' /root/serve_full.log" | tail -300 > $B/v65_lt_layers.txt
wc -l $B/v65_lt_layers.txt
echo "LT_RUN_DONE"
} > $OUT 2>&1
echo "SCRIPT_EXIT=$?"
