#!/bin/bash
# gates_v1222_lane.sh — full content-gate battery re-run against the
# LANE-COMMIT image llm-scaler-exp:v1.2.22 (89b17e0b0f8d, docker-committed
# from the validated warm lsv-test lane so the complete warm triton cache
# ships inside the image — the raw bake commit is preserved untouched as
# llm-scaler-exp:v1.2.22-raw e6735a4c72a2).
# READ-ONLY: identical gate set to stage5_bake_v1222.sh run 2 (wheel version,
# no-debug-instrumentation, serve-config, full v1221 lineage chain, test-probe
# absence, barrier/xgrammar/spec assertions, pedigree markers, serve config
# block) + NEW triton-cache gate (>=74 entries — the root-fix payload) +
# raw-image cache-count contrast (evidence the lane layer adds the cache).
# No serve, no warm round, no commit. Throwaway container lsv-gate removed
# at the end. Exit non-zero on any GATE-FAIL.
set -u
L=/root/build/lce1/v1222_gates_lane.log
{
echo "=== V1222 LANE-COMMIT GATES $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm
FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

IID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.22)
echo "image_id=$IID"
case "$IID" in
  sha256:89b17e0b0f8d*) echo "GATE-OK   image_is_lane_commit 89b17e0b0f8d";;
  *) echo "GATE-FAIL image_is_lane_commit got=$IID"; FAIL=1;;
esac
RID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.22-raw)
case "$RID" in
  sha256:e6735a4c72a2*) echo "GATE-OK   raw_image_is_bake_commit e6735a4c72a2";;
  *) echo "GATE-FAIL raw_image_is_bake_commit got=$RID"; FAIL=1;;
esac

echo "--- throwaway gate container from the lane-commit image ---"
docker rm -f lsv-gate >/dev/null 2>&1 || true
docker run -td --privileged --name lsv-gate \
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
    llm-scaler-exp:v1.2.22 || { echo "ABORT: gate container run failed"; exit 1; }

echo "--- v88 kernels wheel INHERITED: version gate ---"
WV=$(docker exec lsv-gate /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | grep '^Version' | head -1)
echo "wheel: $WV"
echo "$WV" | grep -q 'd20260925' || { echo "GATE-FAIL v88_wheel_absent"; FAIL=1; }

echo "--- no debug instrumentation in image ---"
ck no_gdn_capture 0 "$(docker exec lsv-gate grep -c 'VLLM_GDN_CAPTURE' $SP/_xpu_ops.py)"

echo "--- v1222 serve-config (parser qwen3_coder + 85 capture sizes) ---"
ck serve_parser_coder   1 "$(docker exec lsv-gate grep -c -- '--tool-call-parser qwen3_coder' /root/serve_user.sh)"
ck serve_parser_xml_gone 0 "$(docker exec lsv-gate grep -c -- '--tool-call-parser qwen3_xml' /root/serve_user.sh)"
ck serve_capture_sizes 1 "$(docker exec lsv-gate grep -cF '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,' /root/serve_user.sh)"
ck serve_cudagraph_mode 1 "$(docker exec lsv-gate grep -cF '"cudagraph_mode":"FULL_DECODE_ONLY"' /root/serve_user.sh)"

echo "--- content gates (full v1221 lineage chain, must hold on lane commit) ---"
GMR=$SP/v1/worker/gpu_model_runner.py
SCHED=$SP/v1/core/sched/scheduler.py
ck mark_g62_def   1 "$(docker exec lsv-gate grep -c 'llm-scaler v62 (WEDGEFIX-G): barrier default mode 2' $GMR)"
ck mark_g62_mach  1 "$(docker exec lsv-gate grep -c 'llm-scaler v62 (WEDGEFIX-G): HOST-side spec draft barrier' $GMR)"
ck mark_g62_site  1 "$(docker exec lsv-gate grep -c 'llm-scaler v62 (WEDGEFIX-G): mode dispatch' $GMR)"
ck no_v61_marks   0 "$(docker exec lsv-gate grep -rc 'llm-scaler v61' $SP/v1/worker/gpu_model_runner.py $SP/model_executor/models/qwen3_5_mtp.py $SP/v1/spec_decode/llm_base_proposer.py $SP/v1/worker/logits_processor.py 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')"
ck no_v61_repl    0 "$(docker exec lsv-gate grep -c 'REPLICATED_DRAFT' $GMR)"
ck no_v61_diag    0 "$(docker exec lsv-gate grep -c 'VLLM_V61_DIAG' $GMR)"
if docker exec lsv-gate test -f $SCHED.pre_v64; then
  echo "GATE-OK   v64_backup_present=1"
else
  echo "GATE-FAIL v64_backup_present"; FAIL=1
fi
if docker exec lsv-gate test -f $SCHED.pre_v63; then
  echo "GATE-OK   v63_backup_present=1"
else
  echo "GATE-FAIL v63_backup_present"; FAIL=1
fi
if docker exec lsv-gate test -f $SCHED.pre_v66; then
  echo "GATE-OK   v66_backup_present=1"
else
  echo "GATE-FAIL v66_backup_present"; FAIL=1
fi
ck mark_xpu_v60   1 "$(docker exec lsv-gate grep -c 'llm-scaler v60' $SP/platforms/xpu.py)"
ck mark_gmr_v60b  1 "$(docker exec lsv-gate grep -c 'llm-scaler v60 (wedge fix): barrier DEFAULT ON' $GMR)"
ck mark_gmr_v60g  1 "$(docker exec lsv-gate grep -c 'llm-scaler v60g (WEDGEFIX-E)' $GMR)"
ck mark_gmr_v60e  0 "$(docker exec lsv-gate grep -c 'llm-scaler v60e' $GMR)"
ck mark_f15b      1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler f15b' $GMR | awk '{print (\$1>=1)?1:0}'")"
ck mark_v553      1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler v55.3' $GMR | awk '{print (\$1>=1)?1:0}'")"
ck mark_arstage   1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler arstage' $SP/distributed/device_communicators/xpu_communicator.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_stalfix   1 "$(docker exec lsv-gate sh -c "grep -c STALFIX $SP/entrypoints/openai/responses/serving.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_sched_v60c 1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler v60c' $SCHED | awk '{print (\$1>=1)?1:0}'")"
ck mark_parser    1 "$(docker exec lsv-gate sh -c "grep -c _stalfix_name_emitted $SP/tool_parsers/qwen3xml_tool_parser.py | awk '{print (\$1>=1)?1:0}'")"

ck mark_sched_v63 2 "$(docker exec lsv-gate grep -c 'llm-scaler v63 TTFTFIX' $SCHED)"
ck v63_default    1 "$(docker exec lsv-gate grep -c 'VLLM_V63_CONTENDED_BUDGET", "1024"' $SCHED)"
ck v63_no_2048    0 "$(docker exec lsv-gate grep -c 'VLLM_V63_CONTENDED_BUDGET", "2048"' $SCHED)"
ck v63_active_str 1 "$(docker exec lsv-gate grep -c 'V63_TTFTFIX_ACTIVE' $SCHED)"
ck v63_floor_1024 1 "$(docker exec lsv-gate grep -c 'if 0 < _V63_CONTENDED_BUDGET < 1024' $SCHED)"
ck v63_no_512floor 0 "$(docker exec lsv-gate grep -c '< 256' $SCHED)"

ck mark_sched_v64 2 "$(docker exec lsv-gate grep -c 'llm-scaler v64 TTFTFIX-2' $SCHED)"
ck v64_def_k      1 "$(docker exec lsv-gate grep -c 'VLLM_V64_DECODE_INTERLEAVE", "2"' $SCHED)"
ck v64_def_b      1 "$(docker exec lsv-gate grep -c 'VLLM_V64_DECODE_BUDGET", "512"' $SCHED)"
ck v64_active_str 1 "$(docker exec lsv-gate grep -c 'V64_INTERLEAVE_ACTIVE' $SCHED)"
ck v64_no_k3      0 "$(docker exec lsv-gate grep -c 'VLLM_V64_DECODE_INTERLEAVE", "3"' $SCHED)"

ck mark_sched_v66 4 "$(docker exec lsv-gate grep -c 'llm-scaler v66 FAIRFIX' $SCHED)"
ck v66_starve_default 1 "$(docker exec lsv-gate grep -c 'VLLM_V66_PREFILL_STARVE_S", "2.0"' $SCHED)"
ck v66_active_str 1 "$(docker exec lsv-gate grep -c 'V66_FAIRFIX_ACTIVE' $SCHED)"
ck v66_bypass_sites 2 "$(docker exec lsv-gate grep -c 'and not _v66_bypass_now  # llm-scaler v66 FAIRFIX' $SCHED)"
V66CHECK=$(docker exec lsv-gate /opt/venv/bin/python3 /root/patch_sched_v66_bake.py --check | tail -1 | grep -c APPLIED)
ck v66_check_applied 1 "$V66CHECK"

echo "--- v65 instrumentation absence (test-only probes NEVER baked) ---"
ck no_v65_sched   0 "$(docker exec lsv-gate grep -cE 'VLLM_V65|V65_STEP_LOG|V65_LAYER_LOG' $SCHED)"
ck no_v65_flash   0 "$(docker exec lsv-gate grep -cE 'VLLM_V65|V65_ATTN|V65_GDN|V65_LAYER' $SP/v1/attention/backends/flash_attn.py)"
ck no_v65_xpu     0 "$(docker exec lsv-gate grep -cE 'VLLM_V65|V65_FORCE_TRITON' $SP/platforms/xpu.py)"

echo "--- v88 test-probe absence (test-only patches NEVER baked) ---"
ck no_v84_probe   0 "$(docker exec lsv-gate grep -c 'V84_POOLPROBE' $SP/v1/worker/gpu_model_runner.py)"
ck no_v85_skip    0 "$(docker exec lsv-gate grep -c 'V85_SKIP' $SP/v1/core/block_pool.py)"
ck no_v86_cap     0 "$(docker exec lsv-gate grep -c 'V86CAP' $SP/_xpu_ops.py)"
ck no_v87_stride  0 "$(docker exec lsv-gate grep -c 'V87STR' $SP/_xpu_ops.py)"

echo "--- v89 test-probe absence (py-spy probe NEVER baked; floor sed covered by v63_floor_1024 + v63_no_512floor above) ---"
ck no_v89_pyspy_probe 0 "$(docker exec lsv-gate grep -c 't3_pyspy' $SCHED)"

BARR=$(docker exec lsv-gate /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_default "False True 0" "$BARR"

XG=$(docker exec lsv-gate /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

ck serve_spec_mtp4 1 "$(docker exec lsv-gate grep -cF '"method":"mtp","num_speculative_tokens":4' /root/serve_user.sh)"

docker exec lsv-gate test -f /root/.llm_scaler_exp_v1218_baked && echo "GATE-OK   v1218_pedigree_marker" || { echo "GATE-FAIL v1218_pedigree_marker"; FAIL=1; }
docker exec lsv-gate test -f /root/.llm_scaler_exp_v1220_baked && echo "GATE-OK   v1220_pedigree_marker" || { echo "GATE-FAIL v1220_pedigree_marker"; FAIL=1; }
docker exec lsv-gate test -f /root/.llm_scaler_exp_v1221_baked && echo "GATE-OK   v1221_pedigree_marker" || { echo "GATE-FAIL v1221_pedigree_marker"; FAIL=1; }
docker exec lsv-gate test -f /root/.llm_scaler_exp_v1222_baked && echo "GATE-OK   v1222_pedigree_marker" || { echo "GATE-FAIL v1222_pedigree_marker"; FAIL=1; }
docker exec lsv-gate test -f /root/.v64_jit_warmed && echo "GATE-OK   v64_jit_warm_stamp" || { echo "GATE-FAIL v64_jit_warm_stamp"; FAIL=1; }
docker exec lsv-gate test -f /root/.v1220_jit_warmed && echo "GATE-OK   v1220_jit_warm_stamp" || { echo "GATE-FAIL v1220_jit_warm_stamp"; FAIL=1; }
docker exec lsv-gate test -f /root/.v1221_jit_warmed && echo "GATE-OK   v1221_jit_warm_stamp" || { echo "GATE-FAIL v1221_jit_warm_stamp"; FAIL=1; }
docker exec lsv-gate test -f /root/.v1222_jit_warmed && echo "GATE-OK   v1222_jit_warm_stamp" || { echo "GATE-FAIL v1222_jit_warm_stamp"; FAIL=1; }
if docker exec lsv-gate grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--block-size 64' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--speculative-config' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--max-num-batched-tokens 8192' /root/serve_user.sh \
   && ! docker exec lsv-gate grep -q -- '--async-scheduling' /root/serve_user.sh \
   && ! docker exec lsv-gate grep -q -- '--default-chat-template-kwargs' /root/serve_user.sh \
   && ! docker exec lsv-gate grep -q -- 'chat-template ' /root/serve_user.sh; then
  echo "GATE-OK   baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON mnbt8192 async-OFF no-template)"
else
  echo "GATE-FAIL baked_serve_config"; FAIL=1
fi

echo "--- NEW: warm triton cache shipped in the image (root-fix payload) ---"
docker exec lsv-gate sh -c 'for d in /root/.triton/cache; do [ -d "$d" ] && echo "cache_dir=$d entries=$(ls "$d" | wc -l)"; done'
docker exec lsv-gate sh -c 'ls -lt /root/.triton/cache | head -5'
ck triton_cache_74plus 1 "$(docker exec lsv-gate sh -c "ls /root/.triton/cache 2>/dev/null | wc -l | awk '{print (\$1>=74)?1:0}'")"
RAWT=$(docker run --rm --entrypoint /bin/sh llm-scaler-exp:v1.2.22-raw -c 'ls /root/.triton/cache 2>/dev/null | wc -l' | tr -d '[:space:]')
echo "contrast raw_image_triton_cache_entries=$RAWT (expected 72 — bake entries lost across commit; lane commit carries 74)"

docker rm -f lsv-gate >/dev/null 2>&1 || true
if [ "$FAIL" != "0" ]; then
  echo "=== LANE-COMMIT GATES: FAIL $(date +%T) ==="
else
  echo "=== LANE-COMMIT GATES: ALL PASS $(date +%T) ==="
fi
} > "$L" 2>&1
tail -50 "$L"
grep -q 'GATES: ALL PASS' "$L" || exit 1
exit 0
