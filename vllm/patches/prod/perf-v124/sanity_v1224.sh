#!/bin/bash
# v1.2.20 boot verify: SANITY + xgrammar + spec + C4x1 + barrier + async
# + v63/v64/v66 baked checks + v1224 stamps + admission probe (the v66
# acceptance gate on the shipped image) + JIT checks (zero-gate on the
# proven 8-kernel set, evidence-only for the rest per the v1219 lesson).
R=/root/build/_v1224_sanity.txt
: > $R
B=http://127.0.0.1:8000
echo "== SANITY READY-v1224 ==" >> $R
curl -s -m 120 $B/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"messages":[{"role":"user","content":"Reply with exactly this token and nothing else: READY-v1224"}]}' \
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
echo "== V66 BAKED CHECK ==" >> $R
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v66_bake.py --check >> $R 2>&1
echo "== V66 ADMISSION PROBE (acceptance: verdict=PASS) ==" >> $R
docker cp /root/build/probe_admission_v66.py lsv-test:/root/probe_admission_v66.py
docker exec lsv-test /opt/venv/bin/python3 /root/probe_admission_v66.py http://127.0.0.1:8000 qwen3.8-27b-fp8 661 4000 >> $R 2>&1
echo "== V1224 STAMPS ==" >> $R
docker exec lsv-test sh -c 'ls -la /root/.llm_scaler_exp_v1224_baked /root/.v1224_jit_warmed /root/.v64_jit_warmed 2>&1' >> $R 2>&1
echo "== JIT ZERO CHECK (acceptance: must be 0) ==" >> $R
docker exec lsv-test sh -c "grep -cE 'expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel' /root/serve_full.log || true" >> $R 2>&1
echo "-- any other jit lines (evidence only) --" >> $R
docker exec lsv-test sh -c "grep -iE 'jit' /root/serve_full.log | grep -vE 'jit_warm|JitWrapper' | tail -5 || true" >> $R 2>&1
echo "== IMAGE ==" >> $R
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | head -3 >> $R
echo "SANITY_V1224_DONE" >> $R
echo "== V88 WHEEL CHECK ==" >> $R
docker exec lsv-test /opt/venv/bin/pip freeze 2>/dev/null | grep -i vllm-xpu-kernels >> $R 2>&1
docker exec lsv-test sh -c "ls /root/.llm_scaler_exp_v1220_baked /root/.llm_scaler_exp_v1224_baked 2>&1" >> $R 2>&1
echo "SANITY_V1224_EXTRA_DONE" >> $R
echo "== V88 WHEEL CHECK ==" >> $R
docker exec lsv-test /opt/venv/bin/pip freeze 2>/dev/null | grep -i vllm-xpu-kernels >> $R 2>&1
docker exec lsv-test sh -c "ls /root/.llm_scaler_exp_v1221_baked /root/.llm_scaler_exp_v1224_baked 2>&1" >> $R 2>&1
echo "== V1224 SERVE CONFIG (parser + capture sizes) ==" >> $R
docker exec lsv-test sh -c "grep -o \"--tool-call-parser [a-z0-9_]*\" /root/serve_user.sh" >> $R 2>&1
docker exec lsv-test sh -c "grep -c cudagraph_capture_sizes /root/serve_user.sh" >> $R 2>&1
echo "SANITY_V1224_EXTRA_DONE" >> $R
echo "== V88 WHEEL CHECK ==" >> $R
docker exec lsv-test /opt/venv/bin/pip freeze 2>/dev/null | grep -i vllm-xpu-kernels >> $R 2>&1
echo "== V123 PEDIGREE ==" >> $R
docker exec lsv-test sh -c "ls /root/.llm_scaler_exp_v1221_baked /root/.llm_scaler_exp_v1222_baked /root/.llm_scaler_exp_v1224_baked /root/.v1224_jit_warmed 2>&1" >> $R 2>&1
echo "== V123 SERVE CONFIG (parser + sizes + ASYNC REQUIRED) ==" >> $R
docker exec lsv-test sh -c "grep -o \"--tool-call-parser [a-z0-9_]*\" /root/serve_user.sh" >> $R 2>&1
docker exec lsv-test sh -c "grep -c cudagraph_capture_sizes /root/serve_user.sh" >> $R 2>&1
docker exec lsv-test sh -c "grep -c -- \"--async-scheduling --port 8000\" /root/serve_user.sh" >> $R 2>&1
echo "== V123 BARRIER DEFAULT (acceptance: False False 0) ==" >> $R
docker cp /root/build/patch_barrier_v123_bake.py lsv-test:/root/patch_barrier_v123_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_barrier_v123_bake.py --check >> $R 2>&1
docker exec lsv-test /opt/venv/bin/python3 -c "from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)" >> $R 2>&1
echo "== V123 ASYNC ENGAGED (acceptance: >=1 both) ==" >> $R
docker exec lsv-test sh -c "grep -c \"Asynchronous scheduling is enabled\" /root/serve_full.log" >> $R 2>&1
docker exec lsv-test sh -c "grep -c \"async_scheduling.: True\" /root/serve_full.log" >> $R 2>&1
echo "== V123 SERVE ENV (acceptance: BARRIER=0 + RPC_TIMEOUT=60000) ==" >> $R
docker exec lsv-test sh -c "P=\$(pgrep -f \"vllm serve\" | head -1); tr \"\\0\" \"\\n\" < /proc/\$P/environ | grep -E \"^VLLM_XPU_SPEC_DRAFT_BARRIER=|^VLLM_RPC_TIMEOUT=\"" >> $R 2>&1
echo "== V123 NOTE: BARRIER HB /dev/shm section above EXPECTED EMPTY (mode 0) ==" >> $R
echo "SANITY_V1224_EXTRA_DONE" >> $R
echo "== V124 CODE DELTAS (acceptance: P16=2; P19.5a cache=1 mamba=1 ops=1) ==" >> $R
docker exec lsv-test sh -c "grep -c \"v124 P16\" /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py" >> $R 2>&1
docker exec lsv-test sh -c "grep -c \"v124 P19.5a\" /opt/venv/lib/python3.12/site-packages/vllm/config/cache.py" >> $R 2>&1
docker exec lsv-test sh -c "grep -c \"v124 P19.5a\" /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/mamba_utils.py" >> $R 2>&1
docker exec lsv-test sh -c "grep -c \"v124 P19.5a\" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py" >> $R 2>&1
echo "== V124 MAMBA SERVE DEFAULT (acceptance: 1 / 0 — plumbing inert) ==" >> $R
docker exec lsv-test sh -c "grep -c -- \"mamba-ssm-cache-dtype float16\" /root/serve_user.sh" >> $R 2>&1
docker exec lsv-test sh -c "grep -c -- \"mamba-ssm-cache-dtype fp8\" /root/serve_user.sh" >> $R 2>&1
echo "SANITY_V1224_EXTRA2_DONE" >> $R
