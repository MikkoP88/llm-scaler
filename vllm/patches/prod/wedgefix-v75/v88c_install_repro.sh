#!/bin/bash
# v88c — install the fixed wheel into lsv-test, rerun the standalone repro
# (expect CLEAN exit 0: LO ok + bit-identical vs pre-fix ref + HI(4173) ok +
# BND(4096) ok), then boot the engine stock and warm it.
# Run AFTER a host reboot (device was DEVICE_LOST by the pre-fix repro).
set -u
TS=$(date +%H%M%S)
L=/root/build/lce1/v88c_$TS.out
echo "v88c install+postfix-repro start $TS uptime=$(uptime -p)" | tee -a $L

docker start lsv-test >/dev/null 2>&1 || true
sleep 3

# 1. install the fixed wheel into lsv-test (test lane only — never a bake source)
#    pip requires the CANONICAL wheel filename — copy with basename intact.
WHL=$(ls /root/build/wheels_v88/*.whl | head -1)
WF=$(basename "$WHL")
echo "wheel: $WHL" | tee -a $L
docker cp "$WHL" "lsv-test:/root/$WF"
docker exec lsv-test /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | sed -n '1,2p' | tee -a $L
docker exec lsv-test /opt/venv/bin/pip install --force-reinstall --no-deps "/root/$WF" 2>&1 | tail -3 | tee -a $L
WV_AFTER=$(docker exec lsv-test /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | grep '^Version')
echo "installed: $WV_AFTER" | tee -a $L
echo "$WV_AFTER" | grep -q 'd20260925' || { echo "WHEEL_NOT_INSTALLED — aborting before repro"; exit 1; }

# 2. rotate the pre-fix in-range reference for the bit-compare
docker exec lsv-test mkdir -p /root/v88out
docker cp /root/build/lce1/v88repro/ref100.prefix.pt lsv-test:/root/v88out/ref100.prev.pt

# 3. post-fix repro: expect exit 0 — the exact call that DEVICE_LOST'd pre-fix
docker exec lsv-test sh -c 'cd /root && /opt/venv/bin/python repro_v88_int32.py /root/gdn_cap_input.pt /root/v88out; echo REPRO_EXIT=$?' 2>&1 | tee -a $L

# 4. boot engine stock, health-gate, one warm request
docker exec lsv-test cp /root/serve_user.sh.stock_bak /root/serve_user.sh
docker exec lsv-test pkill -f 'vllm serv[e]' || true
sleep 5
docker exec -d lsv-test bash /root/serve_user.sh
OK=0
for i in $(seq 1 60); do
  sleep 10
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://localhost:8000/health)
  [ "$H" = "200" ] && { echo "[v88c] health 200 after ~$((i*10))s" | tee -a $L; OK=1; break; }
done
[ "$OK" = "1" ] || { echo HEALTH_TIMEOUT | tee -a $L; exit 1; }
curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"Count from 1 to 5.","max_tokens":8,"temperature":0}' \
  -o /dev/null -w 'warm http=%{http_code}\n' | tee -a $L
echo V88C_DONE | tee -a $L
