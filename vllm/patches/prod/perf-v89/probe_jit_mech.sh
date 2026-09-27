#!/bin/bash
# probe_jit_mech.sh — live JIT-mechanism probes on the warm lsv-test lane.
# E1 replay: warm_ext fresh seed MUST add 0 new monitor lines (once-per-process).
# E2 novel shape: top_k=37 sampler curl SHOULD add a _topk_topp warn line
#   (new specialization) — then the triton-cache dir count tells whether
#   stores work in this container (+1 dir) or not (+0).
set -u
dq() { docker exec lsv-test sh -c "$1"; }
W="grep -a 'JIT compilation during inference' /root/serve_full.log | wc -l"
T="grep -ac 'inference: _topk_topp_kernel' /root/serve_full.log"
D='ls /root/.triton/cache | wc -l'
echo "BEFORE warn_total=$(dq "$W") topk=$(dq "$T") dirs=$(dq "$D")"

echo "--- E1: warm_ext replay seed 723 ---"
python3 /root/build/warm_ext_v1219.py http://127.0.0.1:8000 qwen3.8-27b-fp8 723 >/dev/null 2>&1
echo "replay_rc=$?"
echo "AFTER_REPLAY warn_total=$(dq "$W") dirs=$(dq "$D")"

echo "--- E2: novel sampler shape top_k=37 ---"
curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":48,"top_k":37,"messages":[{"role":"user","content":"Say OK."}]}' \
  -o /dev/null -w 'novel_http=%{http_code} t=%{time_total}s\n'
echo "AFTER_NOVEL warn_total=$(dq "$W") topk=$(dq "$T") dirs=$(dq "$D")"
echo "--- newest cache dirs ---"
docker exec lsv-test sh -c 'ls -lt /root/.triton/cache | head -4'
