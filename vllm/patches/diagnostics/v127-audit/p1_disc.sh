#!/bin/bash
# p1_disc.sh — v127-audit P-1 discriminator (fp8 x spec x GMU0.9 penalty).
# P-1a: mtp4_fp8 @ GMU 0.9 + --num-gpu-blocks-override 1171 (block_size 512
#       => pool ~599,392 tok = the 0.85-class pool; margin returns to ~0.85
#       class because vLLM does not reserve unused GMU headroom).
#       FAST (42/34/22-class) => sufficient condition found; ops mitigation
#       = cap pool for fp8+spec. SLOW (16/14/9-class) => neither axis alone.
# P-1c: py-spy stack capture on the SLOW config (0.9 stock) during a 16k
#       bs=1 decode — host-side allocator/sycl stalls vs device time.
#       (Manual comparison vs a 0.85 boot if py-spy exists in the image.)
set -u
OUT=/root/build/bench127/p1
mkdir -p "$OUT"
R="$OUT/report.txt"

echo "== P-1a boot 0.9 + override 1171 $(date +%H:%M:%S)" | tee -a "$R"
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_p1a '' '--num-gpu-blocks-override 1171' \
    > "$OUT/boot_a.out" 2>&1; then
  echo "BOOT_A_OK" | tee -a "$R"
  docker exec lsv-test grep -m2 -E 'GPU KV cache size|num_gpu_blocks' /root/b_p1a.log 2>/dev/null | tee -a "$R"
else
  echo "BOOT_A_FAIL" | tee -a "$R"; tail -15 "$OUT/boot_a.out" | tee -a "$R"
  exit 1
fi

echo "== P-1a warmup $(date +%H:%M:%S)" | tee -a "$R"
ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
WOK=0
for i in $(seq 1 90); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
  curl -s -o /dev/null -m 4 http://localhost:8000/health || true
done
[ "$WOK" = "1" ] && echo "WARMUP_OK" | tee -a "$R" || echo "WARMUP_TIMEOUT" | tee -a "$R"

echo "== P-1a ctxscan $(date +%H:%M:%S)" | tee -a "$R"
python3 /root/build/dt_ctxscan.py p1a > "$OUT/ctxscan_a.out" 2>&1
grep CTXSCAN "$OUT/ctxscan_a.out" | tee -a "$R"

echo "== P-1c py-spy availability probe" | tee -a "$R"
if docker exec lsv-test sh -c 'command -v py-spy || /opt/venv/bin/py-spy --version' >/dev/null 2>&1; then
  echo "PYSPY_IN_IMAGE" | tee -a "$R"
else
  echo "PYSPY_ABSENT (host check next)" | tee -a "$R"
  command -v py-spy >/dev/null 2>&1 && echo "PYSPY_ON_HOST" | tee -a "$R" || echo "NO_PYSPY_ANYWHERE" | tee -a "$R"
fi
docker cp lsv-test:/root/b_p1a.log /root/build/bench127/logs/b_p1a.log >/dev/null 2>&1 || true
echo "P1_DISC_A_DONE $(date +%H:%M:%S)" | tee -a "$R"
