#!/bin/bash
# v60_ab_barrier.sh — A/B tok/s: barrier ON (baked default) vs OFF (env
# override), identical probe. Usage: bash v60_ab_barrier.sh on|off
# Writes /root/build/lce1/v60_ab_$1.log. Lane must be healthy.
set -u
PHASE="$1"
L=/root/build/lce1/v60_ab_${PHASE}.log
B=/root/build/repro_bootV1212.sh
BOFF=/root/build/repro_bootV1212_barrieroff.sh
{
echo "=== V60 A/B PHASE=$PHASE $(date +%F' '%T) ==="
H=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)
echo "health=$H"
BAR=$(docker exec lsv-test /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(int(g._SPEC_DRAFT_BARRIER), g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
echo "barrier_state=$BAR"
if [ "$H" != "200" ]; then echo "ABORT: lane not healthy"; exit 1; fi
echo "--- warmup probe (1 short req, discarded) ---"
curl -s -o /dev/null -m 120 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer sk-dummy" \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":32,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"hi"}]}'
echo "--- probe (3 repeats x 3 shapes) ---"
python3 /root/build/v60_ab_probe.py 3
echo "--- spec metrics tail (context) ---"
docker exec lsv-test sh -c "grep 'SpecDecoding metrics' /root/serve_full.log | tail -2"
echo "=== PHASE=$PHASE DONE $(date +%T) ==="
} > "$L" 2>&1
tail -20 "$L"
echo AB_PHASE_${PHASE}_DONE
