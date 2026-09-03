#!/bin/bash
# dt_matrix.sh <image> [dtype,...] — boot each kv-cache-dtype on the image
# (EXTRAFLAGS argparse last-wins override), run dt_loop + dt_bench per dtype.
# Default matrix: auto (fp16 GT), fp8_e4m3, turboquant_4bit_nc, turboquant_k8v4.
set -u
IMG="${1:-llm-scaler-prod:v1}"
shift || true
DTYPES="${*:-auto fp8_e4m3 turboquant_4bit_nc turboquant_k8v4}"
OUT=/root/build/dt-matrix.out
echo "[$(date +%H:%M:%S)] MATRIX START img=$IMG dtypes=$DTYPES" >> "$OUT"
for DT in $DTYPES; do
  echo "[$(date +%H:%M:%S)] === DTYPE $DT (img $IMG) ===" >> "$OUT"
  if ! bash /root/build/bootp.sh nospec "" "dt_${DT}" "--kv-cache-dtype ${DT}" "$IMG" >> "$OUT" 2>&1; then
    echo "[$(date +%H:%M:%S)] BOOT FAILED: $DT — skipping" >> "$OUT"
    continue
  fi
  python3 /root/build/dt_loop.py "${DT}" > "/root/build/dt-loop-${DT}.out" 2>&1
  echo "[$(date +%H:%M:%S)] loop done: $DT ($(tail -1 /root/build/dt-loop-${DT}.out))" >> "$OUT"
  python3 /root/build/dt_bench.py "${DT}" > "/root/build/dt-bench-${DT}.out" 2>&1
  echo "[$(date +%H:%M:%S)] bench done: $DT" >> "$OUT"
done
echo "[$(date +%H:%M:%S)] MATRIX DONE" >> "$OUT"
