#!/bin/bash
# check_attractor.sh — rerun f8ref on the failing lane (reproduce vs wobble)
# + compare loaded GPU clocks pre-reboot (P2E1F) vs now.
echo "== f8ref rerun (RESTORE26R14B) =="
cd /root/build
timeout 360 python3 -u f8ref.py nv_RESTORE26R14B 2>&1 | grep -o 'hashes=.*'
echo "== P2E1F (pre-reboot, under load) graphics clocks =="
for f in /root/build/lce1/nv_P2E1F_clk_*.txt; do
  awk -F', ' 'NR==1{for(i=1;i<=NF;i++) if($i=="clocks.current.graphics (MHz)") c=i} NR==2{print FILENAME": "$c" MHz"}' "$f" 2>/dev/null
done
echo "== R14 (today, under load) graphics clocks =="
for f in /root/build/lce1/nv_RESTORE26R14_clk_*.txt; do
  awk -F', ' 'NR==1{for(i=1;i<=NF;i++) if($i=="clocks.current.graphics (MHz)") c=i} NR==2{print FILENAME": "$c" MHz"}' "$f" 2>/dev/null
done
echo "== live loaded clock (fire 10s decode then sample) =="
curl -s -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"count from 1 to 200 slowly"}],"max_tokens":400,"temperature":0.9}' -o /dev/null -m 20 &
sleep 4
xpu-smi dump --device 0 --metrics all --number 1 2>/dev/null | awk -F', ' 'NR==1{for(i=1;i<=NF;i++) if($i=="clocks.current.graphics (MHz)") c=i} NR==2{print "dev0 now: "$c" MHz"}'
xpu-smi dump --device 1 --metrics all --number 1 2>/dev/null | awk -F', ' 'NR==1{for(i=1;i<=NF;i++) if($i=="clocks.current.graphics (MHz)") c=i} NR==2{print "dev1 now: "$c" MHz"}'
wait
