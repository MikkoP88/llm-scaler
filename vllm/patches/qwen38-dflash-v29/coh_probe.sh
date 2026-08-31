#!/bin/bash
# coh_probe.sh [label] — temp-0 determinism probe for graphs+MTP numerics
# (v29 Tier-2 session). Three measurements against localhost:8000:
#   P1: 8x identical short completions (first ~6 tokens) -> distinct variants
#   P2: 6x identical 40-token completions on a high-margin prompt -> distinct
#   P1L: 3x logprobs=5 on P1 -> top-1 prob swing (percent-level = forward
#        pass instability; 1e-3-level = FP noise)
L="${1:-coh}"
echo "=== coh_probe $L $(date +%H:%M:%S)"
echo "--- P1 first-token x8:"
for i in 1 2 3 4 5 6 7 8; do
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"The capital of France is","max_tokens":6,"temperature":0}' \
    | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'
done
echo "--- P1 distinct count:"
for i in 1 2 3 4 5 6 7 8; do
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"The capital of France is","max_tokens":6,"temperature":0}' \
    | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'
done | sort -u | wc -l
echo "--- P2 40-tok x6:"
for i in 1 2 3 4 5 6; do
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a python function that sorts a list","max_tokens":40,"temperature":0}' \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:80].replace(chr(10)," "))'
done
echo "--- P2 distinct count:"
for i in 1 2 3 4 5 6; do
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a python function that sorts a list","max_tokens":40,"temperature":0}' \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:80].replace(chr(10)," "))'
done | sort -u | wc -l
echo "--- P1L top-1 logprob x3:"
for i in 1 2 3; do
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"The capital of France is","max_tokens":1,"temperature":0,"logprobs":5}' \
    | python3 -c '
import sys,json
r=json.load(sys.stdin); ch=r["choices"][0]
lp=ch.get("logprobs",{}); top=(lp.get("top_logprobs",[None]) or [{}])[0] or {}
best=sorted(((float(v),k) for k,v in top.items()), reverse=True)[:3]
print(" text=%r top3=%s" % (ch["text"], " ".join("%s:%.3f"%(k,v) for v,k in best)))
'
done
echo "=== coh_probe $L done $(date +%H:%M:%S)"
