#!/bin/bash
# v1.2.19 boot verify: SANITY + xgrammar + spec + C4x1 + barrier + async
# + v63/v64 baked checks + v1219 stamps + THE v1219 acceptance gate:
# zero first-traffic JIT warnings for the known warm-gap kernels on a
# FRESH boot (serve_full.log is truncated by the boot script, so the
# whole file is this boot). Runs the JIT check LAST, after all traffic.
R=/root/build/_v1219_sanity.txt
: > $R
B=http://127.0.0.1:8000
echo "== SANITY READY-v1219 ==" >> $R
curl -s -m 60 $B/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"messages":[{"role":"user","content":"Reply with exactly this token and nothing else: READY-v1219"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["choices"][0]; print("SANITY finish=%s content_head=%r" % (c["finish_reason"], c["message"]["content"][:60]))' >> $R 2>&1
echo "== XGRAMMAR x2 ==" >> $R
cd /root/build && python3 t2_xgrammar_probe.py >> $R 2>&1
echo "== AB C4x1 ==" >> $R
python3 v60_ab_probe.py >> $R 2>&1
echo "== SPEC METRICS ==" >> $R
curl -s -m 10 $B/metrics | grep -E 'spec_decode_(draft|accept)' | grep -v '^#' | tail -4 >> $R
echo "== BARRIER HB ==" >> $R
docker exec lsv-test sh -c 'ls -la /dev/shm/ | grep -i "llm_scaler_spec" | head -4' >> $R 2>&1
echo "== ASYNC ==" >> $R
docker exec lsv-test grep -c 'Asynchronous scheduling is enabled' /root/serve_full.log >> $R 2>&1
echo "== V63 BAKED CHECK ==" >> $R
docker exec lsv-test grep -c 'V63_TTFTFIX_ACTIVE' /root/serve_full.log >> $R 2>&1
echo "== V64 BAKED CHECK ==" >> $R
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v64_bake.py --check >> $R 2>&1
echo "== V1219 STAMPS ==" >> $R
docker exec lsv-test sh -c 'ls -la /root/.llm_scaler_exp_v1219_baked /root/.v1219_jit_warmed /root/.v64_jit_warmed 2>&1' >> $R 2>&1
echo "== JIT ZERO CHECK (acceptance: must be 0) ==" >> $R
docker exec lsv-test sh -c "grep -cE 'expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel' /root/serve_full.log || true" >> $R 2>&1
echo "-- any other jit lines (evidence only) --" >> $R
docker exec lsv-test sh -c "grep -iE 'jit' /root/serve_full.log | grep -vE 'jit_warm|JitWrapper' | tail -5 || true" >> $R 2>&1
echo "== IMAGE ==" >> $R
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | head -3 >> $R
echo "SANITY_V1219_DONE" >> $R
