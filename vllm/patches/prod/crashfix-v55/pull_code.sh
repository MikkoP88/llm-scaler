#!/bin/bash
# pull_code.sh — extract the event producer + deferred mamba postprocess code.
set -u
O=/root/build/lce1/code
mkdir -p "$O"
G=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
M=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py

# every reference to the event + pending pp across the worker dir
docker exec lsv-test sh -c "grep -rn 'num_accepted_tokens_event\|_v32_pending_pp' /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/ | grep -v Binary" > "$O/refs.txt"

# sample_tokens: find via grep line, print the function
SL=$(docker exec lsv-test sh -c "grep -n 'def sample_tokens' $G" | cut -d: -f1)
echo "sample_tokens at line $SL" >> "$O/refs.txt"
docker exec lsv-test sh -c "sed -n \"${SL},$((SL+150))p\" $G" > "$O/sample_tokens.txt"

# async copy thread region 270-350
docker exec lsv-test sh -c "sed -n '260,350p' $G" > "$O/async_copy_region.txt"

# mamba deferred postprocess
DL=$(docker exec lsv-test sh -c "grep -n 'def run_deferred_postprocess' $M" | cut -d: -f1)
echo "run_deferred_postprocess at line $DL" >> "$O/refs.txt"
docker exec lsv-test sh -c "sed -n \"$((DL-20)),$((DL+130))p\" $M" > "$O/deferred_postprocess.txt"

# v52f sites in mamba_utils (PRE-COPY/POST-COPY/DEF-PP)
docker exec lsv-test sh -c "grep -n 'v52f\|PRE-COPY\|POST-COPY\|DEF-PP' $M" > "$O/v52f_sites.txt"

wc -l "$O"/*.txt
echo "=== refs.txt ==="; cat "$O/refs.txt"
echo "=== v52f_sites.txt ==="; cat "$O/v52f_sites.txt"
