#!/bin/bash
# stage5_bake_v1222.sh — gate, bake, and boot llm-scaler-exp:v1.2.22.
# v1.2.22 = v1.2.21 + TWO serve-config changes (v89 T0/T2a winners; NO engine
# code changes, NO kernel wheel changes — the v88 wheel is inherited as-is):
#   1. tool parser qwen3_xml -> qwen3_coder (ran clean all day Sep 27 through
#      the CC fleet under litellm; certified engine-direct 3/3 via
#      cc_parser_battery_v89.py: flat args real strings, nested AskUserQuestion
#      questions native list, plain reply clean; CC battery green post-change).
#   2. cudagraph_capture_sizes extended from the engine default
#      [1,2,4,8,16,24,32,40,48,...,512] to 85 sizes = dense 1..16 (exact-fit
#      draft batch) U {5k: k=1..64} (exact-fit MTP-verify token shapes 5..320)
#      U preserved defaults. Measured: solo decode +3-4% (73.0-74.1 vs
#      71.2-71.4 tok/s; batch-1 verify 5->exact, was 5->8 pad +60%),
#      contention neutral (batch-8 verify 40 already exact in default list),
#      boot penalty ~0 (64/64 graphs in 31s, HEALTH ~140s), KV cache unchanged
#      476,451 tok, 0 evictions.
# v89 T1 REJECTED by measurement (do NOT bake): V63_CONTENDED_BUDGET 512 was
#   floored to 1024 by the v63 guard in all env-based legs; with a test-only
#   floor-256 lane leg the TRUE 512 measured agg_median 70.3 vs ~74.3 pool at
#   1024 = regression; 1024 default + floor STAY. V64_INTERLEAVE=3 measured
#   no gain (77.6 vs 77.1, inside round noise); K=2 stays.
# v89 T2b barrier-off leg: BARRIER=0 measured IDENTICAL to BARRIER=2 (solo
#   73.3/73.0 vs 73.6/74.1) -> keep BARRIER=2 (certified posture).
# Spec MTP x4 + XGrammar-2 0.2.7 support ASSERTED by gates (hard requirement).
# Bake source: FRESH container from llm-scaler-exp:v1.2.21 (NOT the running
# lsv-test lane). Lane MUST BE DOWN first (bake serves on :8000 for warm).
set -u
L=/root/build/lce1/v1222_bake.log
{
echo "=== V1222 BAKE v1.2.22 $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm
FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

if curl -s -o /dev/null -m 3 http://localhost:8000/health; then
  echo "BAKE ABORT: lane :8000 still up (tear down first)"; exit 1
fi

echo "--- fresh bake container from v1.2.21 ---"
docker rm -f lsv-bake >/dev/null 2>&1 || true
docker run -td --privileged --name lsv-bake \
    --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e CCL_TOPO_P2P_ACCESS=1 \
    -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
    -e CCL_ENABLE_SYCL_KERNELS=1 \
    -e CCL_ZE_IPC_EXCHANGE=pidfd \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
    -e ZE_AFFINITY_MASK=0,1 \
    -e VLLM_ALLOW_LONG_MODEL_LEN=1 \
    -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
    -e VLLM_USE_AOT_COMPILE=0 \
    -e CCL_SYSCALL_ALLREDUCE_TMP_BUF=0 \
    -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
    -e VLLM_USE_V2_MODEL_RUNNER=0 \
    -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_QUANTIZE_Q40_LIB=/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_XPU_ALLOC_CONF=expandable_segments:True \
    -e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 \
    -e VLLM_V55_EVENT_TIMEOUT_S=600 \
    -e VLLM_V55_DECODE_TIMEOUT_S=30 \
    -e VLLM_XPU_SPEC_DRAFT_BARRIER=2 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 \
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.21 || { echo "BAKE ABORT: container run failed"; exit 1; }

echo "--- v88 kernels wheel INHERITED (no reinstall): version gate ---"
WV=$(docker exec lsv-bake /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | grep '^Version' | head -1)
echo "wheel: $WV"
echo "$WV" | grep -q 'd20260925' || { echo "BAKE ABORT: v88 wheel not present in base image"; exit 1; }

echo "--- no debug instrumentation in baked image ---"
ck no_gdn_capture 0 "$(docker exec lsv-bake grep -c 'VLLM_GDN_CAPTURE' $SP/_xpu_ops.py)"

echo "--- v1222 serve-config changes: parser qwen3_coder + capture sizes ---"
docker exec lsv-bake sed -i "s/--tool-call-parser qwen3_xml/--tool-call-parser qwen3_coder/" /root/serve_user.sh
ck serve_parser_coder   1 "$(docker exec lsv-bake grep -c -- '--tool-call-parser qwen3_coder' /root/serve_user.sh)"
ck serve_parser_xml_gone 0 "$(docker exec lsv-bake grep -c -- '--tool-call-parser qwen3_xml' /root/serve_user.sh)"
docker cp /root/build/patch_cc_sizes.py lsv-bake:/root/patch_cc_sizes.py
docker exec lsv-bake /opt/venv/bin/python3 /root/patch_cc_sizes.py
CCS=$?
ck cc_sizes_patch_exit 0 "$CCS"
ck serve_capture_sizes 1 "$(docker exec lsv-bake grep -cF '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,' /root/serve_user.sh)"
ck serve_cudagraph_mode 1 "$(docker exec lsv-bake grep -cF '"cudagraph_mode":"FULL_DECODE_ONLY"' /root/serve_user.sh)"

echo "--- extended JIT warm inside bake ---"
docker exec lsv-bake sh -c 'chmod +x /root/serve_user.sh && : > /root/serve_full.log'
docker exec -d lsv-bake bash /root/serve_user.sh
HW=0
for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "bake HEALTH_OK after ~$((i*10))s"; HW=1; break; fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-bake$' || { echo "BAKE CONTAINER DIED during warm serve"; break; }
done
if [ "$HW" != "1" ]; then echo "BAKE ABORT: warm serve health timeout"; exit 1; fi
ck bake_graph_count_64 1 "$(docker exec lsv-bake sh -c "grep -c 'Capturing CUDA graphs' /root/serve_full.log | awk '{print (\$1>=1)?1:0}'")"

JITPAT='expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel'
SAMPAT='_topk_topp|kernel_unified_attention|reduce_segments'

echo "--- warm round 0: legacy v63 warm (solo big prefill + decode burst) ---"
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 qwen3.8-27b-fp8 412
JW=$?
echo "jitwarm_rc=$JW"
if [ "$JW" != "0" ]; then echo "BAKE ABORT: legacy jit warm failed"; exit 1; fi

echo "--- warm round 1: concurrent multi-session multi-turn (warm_ext) ---"
python3 /root/build/warm_ext_v1219.py http://127.0.0.1:8000 qwen3.8-27b-fp8 519
W1=$?
echo "warmext_rc=$W1"
if [ "$W1" != "0" ]; then echo "BAKE ABORT: warm_ext round 1 failed"; exit 1; fi

echo "--- warm round 1b: xgrammar structured-output paths ---"
( cd /root/build && python3 t2_xgrammar_probe.py 2 ) || echo "xgrammar warm nonfatal_rc=$?"

echo "--- warm round 1c: sampler-shape round (default / top_p / top_k, 2000 tok) ---"
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"messages":[{"role":"user","content":"Write a 300-word story about a lighthouse."}]}' -o /dev/null -w 'sampler_default_http=%{http_code}\n'
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"top_p":0.9,"messages":[{"role":"user","content":"Write a 300-word story about a harbor."}]}' -o /dev/null -w 'sampler_topp_http=%{http_code}\n'
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"top_k":40,"messages":[{"role":"user","content":"Write a 300-word story about a foghorn."}]}' -o /dev/null -w 'sampler_topk_http=%{http_code}\n'
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"temperature":0.9,"top_p":0.95,"top_k":20,"messages":[{"role":"user","content":"Write a 300-word story about a buoy."}]}' -o /dev/null -w 'sampler_all_http=%{http_code}\n'
echo "sampler-round jit evidence:"
docker exec lsv-bake sh -c "grep -E '$SAMPAT' /root/serve_full.log | tail -8" || echo "(no sampler jit lines — already covered)"

echo "--- round-1 JIT coverage evidence (kernels compiled during warm) ---"
docker exec lsv-bake sh -c "grep -E '$JITPAT' /root/serve_full.log | grep -oE '$JITPAT' | sort | uniq -c" || echo "(none matched — check patterns)"

echo "--- warm round 2 (verify replay, fresh seeds -> cache miss) ---"
OFF=$(docker exec lsv-bake sh -c 'wc -l < /root/serve_full.log' | tr -d '[:space:]')
echo "log_offset_before_round2=$OFF"
python3 /root/build/warm_ext_v1219.py http://127.0.0.1:8000 qwen3.8-27b-fp8 619 --verify
W2=$?
echo "warmext_verify_rc=$W2"
if [ "$W2" != "0" ]; then echo "BAKE ABORT: warm_ext round 2 failed"; exit 1; fi
JN=$(docker exec lsv-bake sh -c "tail -n +${OFF} /root/serve_full.log | grep -cE '$JITPAT'" || true)
echo "round2_new_jit_warnings=$JN"
ck warm_round2_zero_jit 0 "$JN"
SN=$(docker exec lsv-bake sh -c "tail -n +${OFF} /root/serve_full.log | grep -cE '$SAMPAT'" || true)
echo "round2_sampler_jit_evidence_only=$SN"

docker exec lsv-bake touch /root/.v1222_jit_warmed
docker exec lsv-bake bash -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do
  sleep 2
  docker exec lsv-bake bash -c "pgrep -f 'vllm serve' >/dev/null" || break
done
echo "bake serve stopped"

echo "--- content gates (full v1221 lineage chain, must hold on v1.2.21 base) ---"
GMR=$SP/v1/worker/gpu_model_runner.py
SCHED=$SP/v1/core/sched/scheduler.py
ck mark_g62_def   1 "$(docker exec lsv-bake grep -c 'llm-scaler v62 (WEDGEFIX-G): barrier default mode 2' $GMR)"
ck mark_g62_mach  1 "$(docker exec lsv-bake grep -c 'llm-scaler v62 (WEDGEFIX-G): HOST-side spec draft barrier' $GMR)"
ck mark_g62_site  1 "$(docker exec lsv-bake grep -c 'llm-scaler v62 (WEDGEFIX-G): mode dispatch' $GMR)"
ck no_v61_marks   0 "$(docker exec lsv-bake grep -rc 'llm-scaler v61' $SP/v1/worker/gpu_model_runner.py $SP/model_executor/models/qwen3_5_mtp.py $SP/v1/spec_decode/llm_base_proposer.py $SP/v1/worker/logits_processor.py 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')"
ck no_v61_repl    0 "$(docker exec lsv-bake grep -c 'REPLICATED_DRAFT' $GMR)"
ck no_v61_diag    0 "$(docker exec lsv-bake grep -c 'VLLM_V61_DIAG' $GMR)"
if docker exec lsv-bake test -f $SCHED.pre_v64; then
  echo "GATE-OK   v64_backup_present=1"
else
  echo "GATE-FAIL v64_backup_present"; FAIL=1
fi
if docker exec lsv-bake test -f $SCHED.pre_v63; then
  echo "GATE-OK   v63_backup_present=1"
else
  echo "GATE-FAIL v63_backup_present"; FAIL=1
fi
if docker exec lsv-bake test -f $SCHED.pre_v66; then
  echo "GATE-OK   v66_backup_present=1"
else
  echo "GATE-FAIL v66_backup_present"; FAIL=1
fi
ck mark_xpu_v60   1 "$(docker exec lsv-bake grep -c 'llm-scaler v60' $SP/platforms/xpu.py)"
ck mark_gmr_v60b  1 "$(docker exec lsv-bake grep -c 'llm-scaler v60 (wedge fix): barrier DEFAULT ON' $GMR)"
ck mark_gmr_v60g  1 "$(docker exec lsv-bake grep -c 'llm-scaler v60g (WEDGEFIX-E)' $GMR)"
ck mark_gmr_v60e  0 "$(docker exec lsv-bake grep -c 'llm-scaler v60e' $GMR)"
ck mark_f15b      1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler f15b' $GMR | awk '{print (\$1>=1)?1:0}'")"
ck mark_v553      1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v55.3' $GMR | awk '{print (\$1>=1)?1:0}'")"
ck mark_arstage   1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler arstage' $SP/distributed/device_communicators/xpu_communicator.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_stalfix   1 "$(docker exec lsv-bake sh -c "grep -c STALFIX $SP/entrypoints/openai/responses/serving.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_sched_v60c 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v60c' $SCHED | awk '{print (\$1>=1)?1:0}'")"
ck mark_parser    1 "$(docker exec lsv-bake sh -c "grep -c _stalfix_name_emitted $SP/tool_parsers/qwen3xml_tool_parser.py | awk '{print (\$1>=1)?1:0}'")"

ck mark_sched_v63 2 "$(docker exec lsv-bake grep -c 'llm-scaler v63 TTFTFIX' $SCHED)"
ck v63_default    1 "$(docker exec lsv-bake grep -c 'VLLM_V63_CONTENDED_BUDGET", "1024"' $SCHED)"
ck v63_no_2048    0 "$(docker exec lsv-bake grep -c 'VLLM_V63_CONTENDED_BUDGET", "2048"' $SCHED)"
ck v63_active_str 1 "$(docker exec lsv-bake grep -c 'V63_TTFTFIX_ACTIVE' $SCHED)"
ck v63_floor_1024 1 "$(docker exec lsv-bake grep -c 'if 0 < _V63_CONTENDED_BUDGET < 1024' $SCHED)"
ck v63_no_512floor 0 "$(docker exec lsv-bake grep -c '< 256' $SCHED)"

ck mark_sched_v64 2 "$(docker exec lsv-bake grep -c 'llm-scaler v64 TTFTFIX-2' $SCHED)"
ck v64_def_k      1 "$(docker exec lsv-bake grep -c 'VLLM_V64_DECODE_INTERLEAVE", "2"' $SCHED)"
ck v64_def_b      1 "$(docker exec lsv-bake grep -c 'VLLM_V64_DECODE_BUDGET", "512"' $SCHED)"
ck v64_active_str 1 "$(docker exec lsv-bake grep -c 'V64_INTERLEAVE_ACTIVE' $SCHED)"
ck v64_no_k3      0 "$(docker exec lsv-bake grep -c 'VLLM_V64_DECODE_INTERLEAVE", "3"' $SCHED)"

ck mark_sched_v66 4 "$(docker exec lsv-bake grep -c 'llm-scaler v66 FAIRFIX' $SCHED)"
ck v66_starve_default 1 "$(docker exec lsv-bake grep -c 'VLLM_V66_PREFILL_STARVE_S", "2.0"' $SCHED)"
ck v66_active_str 1 "$(docker exec lsv-bake grep -c 'V66_FAIRFIX_ACTIVE' $SCHED)"
ck v66_bypass_sites 2 "$(docker exec lsv-bake grep -c 'and not _v66_bypass_now  # llm-scaler v66 FAIRFIX' $SCHED)"
V66CHECK=$(docker exec lsv-bake /opt/venv/bin/python3 /root/patch_sched_v66_bake.py --check | tail -1 | grep -c APPLIED)
ck v66_check_applied 1 "$V66CHECK"

echo "--- v65 instrumentation absence (test-only probes NEVER baked) ---"
ck no_v65_sched   0 "$(docker exec lsv-bake grep -cE 'VLLM_V65|V65_STEP_LOG|V65_LAYER_LOG' $SCHED)"
ck no_v65_flash   0 "$(docker exec lsv-bake grep -cE 'VLLM_V65|V65_ATTN|V65_GDN|V65_LAYER' $SP/v1/attention/backends/flash_attn.py)"
ck no_v65_xpu     0 "$(docker exec lsv-bake grep -cE 'VLLM_V65|V65_FORCE_TRITON' $SP/platforms/xpu.py)"

echo "--- v88 test-probe absence (test-only patches NEVER baked) ---"
ck no_v84_probe   0 "$(docker exec lsv-bake grep -c 'V84_POOLPROBE' $SP/v1/worker/gpu_model_runner.py)"
ck no_v85_skip    0 "$(docker exec lsv-bake grep -c 'V85_SKIP' $SP/v1/core/block_pool.py)"
ck no_v86_cap     0 "$(docker exec lsv-bake grep -c 'V86CAP' $SP/_xpu_ops.py)"
ck no_v87_stride  0 "$(docker exec lsv-bake grep -c 'V87STR' $SP/_xpu_ops.py)"

echo "--- v89 test-probe absence (test-only floor sed NEVER baked) ---"
ck no_v89_pyspy_probe 0 "$(docker exec lsv-bake grep -c 't3_pyspy' $SCHED)"

BARR=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_default "False True 0" "$BARR"

XG=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

ck serve_spec_mtp4 1 "$(docker exec lsv-bake grep -cF '"method":"mtp","num_speculative_tokens":4' /root/serve_user.sh)"

docker exec lsv-bake test -f /root/.llm_scaler_exp_v1218_baked && echo "GATE-OK   v1218_pedigree_marker" || { echo "GATE-FAIL v1218_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1220_baked && echo "GATE-OK   v1220_pedigree_marker" || { echo "GATE-FAIL v1220_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1221_baked && echo "GATE-OK   v1221_pedigree_marker" || { echo "GATE-FAIL v1221_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.v64_jit_warmed && echo "GATE-OK   v64_jit_warm_stamp" || { echo "GATE-FAIL v64_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1220_jit_warmed && echo "GATE-OK   v1220_jit_warm_stamp" || { echo "GATE-FAIL v1220_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1221_jit_warmed && echo "GATE-OK   v1221_jit_warm_stamp" || { echo "GATE-FAIL v1221_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1222_jit_warmed && echo "GATE-OK   v1222_jit_warm_stamp" || { echo "GATE-FAIL v1222_jit_warm_stamp"; FAIL=1; }
if docker exec lsv-bake grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--block-size 64' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--speculative-config' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--max-num-batched-tokens 8192' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- '--async-scheduling' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- '--default-chat-template-kwargs' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- 'chat-template ' /root/serve_user.sh; then
  echo "GATE-OK   baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON mnbt8192 async-OFF no-template)"
else
  echo "GATE-FAIL baked_serve_config"; FAIL=1
fi

if [ "$FAIL" != "0" ]; then
  echo "BAKE ABORTED — gates failed"; docker rm -f lsv-bake >/dev/null 2>&1 || true
  echo "=== BAKE DONE (ABORT) $(date +%T) ==="; exit 1
fi

echo "--- gates PASS: committing v1.2.22 ---"
docker exec lsv-bake touch /root/.llm_scaler_exp_v1222_baked
docker commit lsv-bake llm-scaler-exp:v1.2.22
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.22 --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
docker rm -f lsv-bake >/dev/null 2>&1 || true

echo "--- build repro_bootV1222.sh from V1221 (image + marker only; all patches already baked) ---"
B=/root/build/repro_bootV1222.sh
cp /root/build/repro_bootV1221.sh "$B"
sed -i 's/llm-scaler-exp:v1\.2\.21/llm-scaler-exp:v1.2.22/g' "$B"
sed -i 's|test -f /root/.llm_scaler_exp_v1221_baked|test -f /root/.llm_scaler_exp_v1222_baked|' "$B"
bash -n "$B" || { echo "BOOT SCRIPT SYNTAX FAIL"; exit 1; }
echo "--- diff V1221 -> V1222 boot (expect ONLY image + marker lines) ---"
diff /root/build/repro_bootV1221.sh "$B" || true
grep -nE 'v1\.2\.22|v1222_baked' "$B" | head -8

echo "=== BAKE DONE (image committed, boot script ready) $(date +%T) ==="
} > "$L" 2>&1
tail -100 "$L"
echo STAGE5_BAKE_V1222_DONE
