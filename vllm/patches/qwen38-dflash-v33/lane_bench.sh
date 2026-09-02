#!/bin/bash
# lane_bench.sh <tag> [conc16] — standard v33 lane battery on the RUNNING
# server: 2k ctxbench, 65k cold + warm x2, greedy hashes. Prints one
# LANERESULT line. conc16 optional extra (0/1).
TAG="$1"; CONC="${2:-0}"
# warmup: first traffic after boot pays one-time graph-capture/JIT (~10s
# TTFT); discard it so lane numbers are comparable.
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":8,"temperature":0}' >/dev/null
python3 /root/build/ctxbench.py "$TAG" 2048 512 1
python3 /root/build/t65k.py
python3 /root/build/t65k.py
python3 /root/build/t65k.py
python3 /root/build/f8ref.py "$TAG"
if [ "$CONC" = "1" ]; then python3 /root/build/ctxbench.py "${TAG}c16" 8192 256 16; fi
echo "LANEBENCH_DONE $TAG"
