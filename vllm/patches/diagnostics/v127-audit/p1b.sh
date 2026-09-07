#!/bin/bash
# p1b.sh — v127-audit P-1 axis-split boot B: mtp4_fp8 @ GMU 0.9 with
# num_gpu_blocks_override 626 (fork units: ~599k tokens = the 0.85-class
# pool; p1a showed override units ~956.73 tok: 740 natural == 707,980 stock,
# 1171 == 1,120,330). If FAST -> pool size / memory pressure is the P-1
# driver (sufficient); if SLOW -> margin/allocator axis stands.
# Plus: py-spy dump loop (host py-spy, PYSPY_ON_HOST) during ctxscan.
set -u
OUT=/root/build/bench127/p1b
mkdir -p "$OUT"
R="$OUT/report.txt"

echo "== P-1b boot 0.9 + override 626 $(date +%H:%M:%S)" | tee -a "$R"
docker rm -f lsv-test >/dev/null 2>&1
sleep 5
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_p1b '' '--num-gpu-blocks-override 626' \
    > "$OUT/boot.out" 2>&1; then
  echo "BOOT_OK" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R"; tail -15 "$OUT/boot.out" | tee -a "$R"; exit 1
fi
docker exec lsv-test grep -m3 -E "GPU KV cache size|Overriding num_gpu_blocks" /root/b_p1b.log 2>/dev/null | tee -a "$R"

HOK=0
for i in $(seq 1 60); do
  sleep 10
  if curl -s -o /dev/null -m 4 http://localhost:8000/health; then HOK=1; break; fi
done
[ "$HOK" = "1" ] && echo "HEALTH_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo "HEALTH_TIMEOUT" | tee -a "$R"; exit 1; }

ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
WOK=0
for i in $(seq 1 90); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
done
[ "$WOK" = "1" ] && echo "WARMUP_OK $(date +%H:%M:%S)" | tee -a "$R" || echo "WARMUP_TIMEOUT (continuing)" | tee -a "$R"

# py-spy dump loop against EngineCore (host py-spy; container pids visible)
nohup bash -c 'for j in $(seq 1 45); do sleep 25; ECP=$(docker exec lsv-test pgrep -f "EngineCore" | head -1); if [ -n "$ECP" ]; then HP=$(pgrep -f "EngineCore" | head -1); py-spy dump --pid "$HP" 2>&1 | head -80; echo ---DUMP-END---; fi; done' > "$OUT/pyspy_dumps.txt" 2>&1 &

echo "== P-1b ctxscan $(date +%H:%M:%S)" | tee -a "$R"
python3 /root/build/dt_ctxscan.py p1b > "$OUT/ctxscan_b.out" 2>&1
grep CTXSCAN "$OUT/ctxscan_b.out" | tee -a "$R"

docker cp lsv-test:/root/b_p1b.log /root/build/bench127/logs/b_p1b.log >/dev/null 2>&1 || true
echo "P1B_DONE $(date +%H:%M:%S)" | tee -a "$R"
