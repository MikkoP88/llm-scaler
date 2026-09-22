#!/bin/sh
# v65_runFG.sh — mnbt grid solo cold: F=4096 seed203, G=6144 seed204, then
# restore mnbt 8192. Collects V65_STEP curves for each. Output _v65_runFG.txt.
B=/root/build
OUT=$B/_v65_runFG.txt
relaunch() {
  docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
  for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
  sleep 3
  docker exec -d lsv-test bash /root/serve_user.sh
  for i in $(seq 1 60); do
    sleep 10
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
    if [ "$c" = "200" ]; then echo "serve HEALTH_OK ~$((i*10))s"; return 0; fi
  done
  echo ABORT_SERVE; return 1
}
{
echo "=== V65 RUNFG $(date +%T) ==="
# F: mnbt 4096
docker exec lsv-test sh -c "sed -i 's/--max-num-batched-tokens 8192/--max-num-batched-tokens 4096/' /root/serve_user.sh; grep 'max-num-batched' /root/serve_user.sh"
relaunch || exit 1
docker exec lsv-test sh -c "wc -c < /root/serve_full.log" > $B/_v65_f_off.txt
echo "== F: mnbt4096 solo seed203 =="
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 203 2>&1 | tail -3
OFF=$(cat $B/_v65_f_off.txt)
echo "-- F steps --"
docker exec lsv-test sh -c "tail -c +$OFF /root/serve_full.log | grep -A1 'V65_STEP' | grep Arguments | head -60"
# G: mnbt 6144
docker exec lsv-test sh -c "sed -i 's/--max-num-batched-tokens 4096/--max-num-batched-tokens 6144/' /root/serve_user.sh; grep 'max-num-batched' /root/serve_user.sh"
relaunch || exit 1
docker exec lsv-test sh -c "wc -c < /root/serve_full.log" > $B/_v65_g_off.txt
echo "== G: mnbt6144 solo seed204 =="
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 204 2>&1 | tail -3
OFF=$(cat $B/_v65_g_off.txt)
echo "-- G steps --"
docker exec lsv-test sh -c "tail -c +$OFF /root/serve_full.log | grep -A1 'V65_STEP' | grep Arguments | head -60"
# restore 8192
docker exec lsv-test sh -c "sed -i 's/--max-num-batched-tokens 6144/--max-num-batched-tokens 8192/' /root/serve_user.sh; grep 'max-num-batched' /root/serve_user.sh"
relaunch || exit 1
echo "=== V65_RUNFG DONE $(date +%T) ==="
echo V65_RUNFG_DONE
} > $OUT 2>&1
