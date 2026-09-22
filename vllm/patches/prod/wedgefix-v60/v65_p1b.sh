#!/bin/sh
# v65_p1b.sh — corrected: docker cp patcher into container, apply, relaunch,
# solo cold seed202, collect V65_STEP curve. Output /root/build/_v65_p1b.txt.
B=/root/build
OUT=$B/_v65_p1b.txt
{
echo "=== V65 P1B $(date +%T) ==="
# wait for any in-flight v65_p1.sh to finish
for i in $(seq 1 60); do grep -q V65_P1_DONE $B/_v65_p1.txt 2>/dev/null && break; sleep 5; done
# 1. copy patcher INTO container, apply, check
docker cp $B/patch_v65_probe.py lsv-test:/root/patch_v65_probe.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v65_probe.py --apply
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v65_probe.py --check
# 2. env line (idempotent)
docker exec lsv-test sh -c "sed -i '/VLLM_V65_STEP_LOG/d' /root/serve_user.sh; sed -i '1i export VLLM_V65_STEP_LOG=1' /root/serve_user.sh; head -3 /root/serve_user.sh"
# 3. relaunch serve
docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
sleep 3
docker exec -d lsv-test bash /root/serve_user.sh
HW=0
for i in $(seq 1 60); do
  sleep 10
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then echo "serve HEALTH_OK ~$((i*10))s"; HW=1; break; fi
done
[ "$HW" = "1" ] || { echo ABORT_SERVE; exit 1; }
# 4. offset + solo cold seed202
docker exec lsv-test sh -c "wc -c < /root/serve_full.log" > $B/_v65_p1b_off.txt
echo "== solo cold seed202 =="
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 202 2>&1 | tail -4
# 5. collect step lines
OFF=$(cat $B/_v65_p1b_off.txt)
echo "== V65_STEP lines =="
docker exec lsv-test sh -c "tail -c +$OFF /root/serve_full.log | grep 'V65_STEP' | head -200"
echo "=== V65_P1B DONE $(date +%T) ==="
echo V65_P1B_DONE
} > $OUT 2>&1
