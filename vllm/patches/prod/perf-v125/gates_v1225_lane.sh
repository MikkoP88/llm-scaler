#!/bin/bash
# gates_v1225_lane.sh — full content-gate battery against the SHIPPED v1.2.25
# image (docker commit of the clean warm validated lane, v1222/v1223
# precedent). Pass V123_ID=sha256:<prefix> of the commit and V123RAW_ID of the
# v1.2.25-raw bake to assert both ids. Identity is primarily content-gated:
# the v123 layer (barrier default "0" + marker + --async-scheduling baked)
# plus the v125 layer (opsall root fix + C7 fp8-state refusal) uniquely mark
# it. READ-ONLY: wheel version + no-P22B-fp8-strings, no-debug-instrumentation,
# v124 deltas (P16 + P19.5a plumbing + fp8 dtype resolution), v125 deltas
# (opsall marker + C7 marker + functional refusal + C5/C6 absence),
# serve-config (async REQUIRED), full v1218..v1225 lineage chain incl. v1223
# (missing from the v1224 gate script — restored here), test-probe absence,
# barrier-off/xgrammar/spec assertions, triton cache floor vs raw.
# Throwaway container lsv-gate removed at the end. Exit non-zero on GATE-FAIL.
set -u
L=/root/build/lce1/v1225_gates_lane.log
{
echo "=== V1225 SHIP GATES $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm
SPR=/opt/venv/lib/python3.12/site-packages
SPK=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

IID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.25)
echo "image_id=$IID"
if [ -n "${V123_ID:-}" ]; then
  case "$IID" in
    "$V123_ID"*) echo "GATE-OK   image_is_expected_commit $V123_ID";;
    *) echo "GATE-FAIL image_is_expected_commit expected=$V123_ID got=$IID"; FAIL=1;;
  esac
else
  echo "NOTE      V123_ID not set — id assertion skipped (content gates below decide)"
fi
RID=$(docker inspect --format '{{.Id}}' llm-scaler-exp:v1.2.25-raw)
echo "raw_image_id=$RID"
if [ -n "${V123RAW_ID:-}" ]; then
  case "$RID" in
    "$V123RAW_ID"*) echo "GATE-OK   raw_image_is_bake_commit $V123RAW_ID";;
    *) echo "GATE-FAIL raw_image_is_bake_commit expected=$V123RAW_ID got=$RID"; FAIL=1;;
  esac
else
  echo "NOTE      V123RAW_ID not set — raw id assertion skipped"
fi

echo "--- throwaway gate container from the shipped image ---"

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
    -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 \
    -e VLLM_RPC_TIMEOUT=60000 \
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.25 || { echo "ABORT: gate container run failed"; exit 1; }

echo "--- v125 deltas: opsall root fix + C7 fp8-state refusal ---"
XO=$SP/_xpu_ops.py
GMR0=$SP/v1/worker/gpu_model_runner.py
ck v125_opsall_marker 1 "$(docker exec lsv-gate sh -c "grep -c 'perf-v125 root fix' $SP/platforms/xpu.py | awk '{print (\$1>=1)?1:0}'")"
ck v125_opsall_none2all 1 "$(docker exec lsv-gate grep -cF '"all" if c == "none" else c' $SP/platforms/xpu.py)"
if docker exec lsv-gate test -f $SP/platforms/xpu.py.pre_v125_opsfix; then
  echo "GATE-OK   v125_opsfix_backup_present=1"
else
  echo "GATE-FAIL v125_opsfix_backup_present"; FAIL=1
fi
ck v125_c7_marker 1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler v125 C7' $XO | awk '{print (\$1>=1)?1:0}'")"
# The raw base has NO P22B switches; after C7 the env name appears EXACTLY
# twice (environ.get + the RuntimeError message), both inside the guard.
ck v125_fp8_native_refs_guard_only 2 "$(docker exec lsv-gate grep -c 'VLLM_XPU_GDN_FP8_NATIVE' $XO)"
ck v125_no_fp8_mode_var 0 "$(docker exec lsv-gate grep -c '_GDN_FP8_NATIVE_MODE' $XO)"
ck v125_no_c5_marker 0 "$(docker exec lsv-gate grep -c 'P22C5' $XO)"
ck v125_no_c6_marker 0 "$(docker exec lsv-gate grep -c 'P22C6' $XO)"
ck v125_no_c5_in_gmr 0 "$(docker exec lsv-gate grep -c 'P22C5' $GMR0)"
ck v125_no_serve_variants 0 "$(docker exec lsv-gate sh -c "ls /root/ | grep -cE 'serve_(p22|p23|mfp8|spec0|mnbt16k)'" || true)"
echo "--- v125 C7 FUNCTIONAL (in-container: refuse =1/=2, clean import unset) ---"
C7G2=$(docker exec -e VLLM_XPU_GDN_FP8_NATIVE=2 lsv-gate /opt/venv/bin/python3 -c 'import vllm._xpu_ops' 2>&1; echo "RC=$?")
ck c7g_refuse_2_raise 1 "$(echo "$C7G2" | grep -c 'v125 C7')"
ck c7g_refuse_2_rc RC=1 "$(echo "$C7G2" | tail -1)"
C7G1=$(docker exec -e VLLM_XPU_GDN_FP8_NATIVE=1 lsv-gate /opt/venv/bin/python3 -c 'import vllm._xpu_ops' 2>&1; echo "RC=$?")
ck c7g_refuse_1_raise 1 "$(echo "$C7G1" | grep -c 'v125 C7')"
ck c7g_refuse_1_rc RC=1 "$(echo "$C7G1" | tail -1)"
C7G0=$(docker exec lsv-gate /opt/venv/bin/python3 -c 'import vllm._xpu_ops; print("IMPORT_OK")' 2>&1; echo "RC=$?")
ck c7g_unset_import_clean 1 "$(echo "$C7G0" | grep -c IMPORT_OK)"
ck c7g_unset_rc RC=0 "$(echo "$C7G0" | tail -1)"

echo "--- v125 kernels wheel: certified build only, NO P22B fp8-state kernel ---"
WV=$(docker exec lsv-gate /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | grep '^Version' | head -1)
echo "wheel: $WV"
echo "$WV" | grep -q 'd20260925' || { echo "GATE-FAIL v88_wheel_absent"; FAIL=1; }
ck wheel_no_fp8_state_strings 0 "$(docker exec lsv-gate bash -c "strings $SPK/_xpu_C.abi3.so | grep -c 'fp8_e4m3fn'" || true)"

echo "--- v124 deltas: P16 gate fix + P19.5a fp8 plumbing (inert on fp16 serve) ---"
ck v124_p16_markers 2 "$(docker exec lsv-gate sh -c "grep -c 'v124 P16' $SPR/vllm/platforms/xpu.py" | tr -d '[:space:]')"
ck v124_p195_cache_literal 1 "$(docker exec lsv-gate sh -c "grep -c 'v124 P19.5a' $SPR/vllm/config/cache.py" | tr -d '[:space:]')"
ck v124_p195_mamba_resolver 1 "$(docker exec lsv-gate sh -c "grep -c 'v124 P19.5a' $SPR/vllm/model_executor/layers/mamba/mamba_utils.py" | tr -d '[:space:]')"
ck v124_p195_sycl_bridge 1 "$(docker exec lsv-gate sh -c "grep -c 'v124 P19.5a' $SPR/vllm/_xpu_ops.py" | tr -d '[:space:]')"
ck v124_serve_mamba_fp16_default 1 "$(docker exec lsv-gate grep -c -- '--mamba-ssm-cache-dtype float16' /root/serve_user.sh | tr -d '[:space:]')"
ck v124_serve_mamba_fp8_absent 0 "$(docker exec lsv-gate grep -c -- '--mamba-ssm-cache-dtype fp8' /root/serve_user.sh | tr -d '[:space:]')"
V124RT=$(docker exec lsv-gate /opt/venv/bin/python3 -c "import torch; from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C; a=C.gated_delta_net_state_dtype(torch.float16, 'auto', 'fp8_e4m3'); b=C.gated_delta_net_state_dtype(torch.float16, 'auto', 'fp8_e5m2'); print(a[1], b[1])" 2>&1)
echo "v124_runtime_fp8_resolution: $V124RT"
echo "$V124RT" | grep -q float8_e4m3fn && echo "GATE-OK   v124_fp8_e4m3_resolves" || { echo "GATE-FAIL v124_fp8_e4m3_resolves"; FAIL=1; }
echo "$V124RT" | grep -q float8_e5m2 && echo "GATE-OK   v124_fp8_e5m2_resolves" || { echo "GATE-FAIL v124_fp8_e5m2_resolves"; FAIL=1; }

echo "--- no debug instrumentation in image ---"
ck no_gdn_capture 0 "$(docker exec lsv-gate grep -c 'VLLM_GDN_CAPTURE' $SP/_xpu_ops.py)"

echo "--- v1222 serve-config (parser qwen3_coder + 85 capture sizes) ---"
ck serve_parser_coder   1 "$(docker exec lsv-gate grep -c -- '--tool-call-parser qwen3_coder' /root/serve_user.sh)"
ck serve_parser_xml_gone 0 "$(docker exec lsv-gate grep -c -- '--tool-call-parser qwen3_xml' /root/serve_user.sh)"
ck serve_capture_sizes 1 "$(docker exec lsv-gate grep -cF '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,' /root/serve_user.sh)"
ck serve_cudagraph_mode 1 "$(docker exec lsv-gate grep -cF '"cudagraph_mode":"FULL_DECODE_ONLY"' /root/serve_user.sh)"

echo "--- v123 layer: async-scheduling BAKED + barrier default OFF ---"
ck serve_async_sched 1 "$(docker exec lsv-gate grep -c -- '--async-scheduling --port 8000' /root/serve_user.sh)"
ck no_v123_async_variant 0 "$(docker exec lsv-gate grep -c 'serve_user_async' /root/serve_user.sh)"
docker cp /root/build/patch_barrier_v123_bake.py lsv-gate:/root/patch_barrier_v123_bake.py
V123CHK=$(docker exec lsv-gate /opt/venv/bin/python3 /root/patch_barrier_v123_bake.py --check | tail -1 | grep -c APPLIED)
ck v123_barrier_check 1 "$V123CHK"
BARR=$(docker exec lsv-gate /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_default "False False 0" "$BARR"

echo "--- content gates (full v1221 lineage chain, must hold on ship image) ---"
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

echo "--- v89/v123 test-probe absence (py-spy probe NEVER baked) ---"
ck no_v89_pyspy_probe 0 "$(docker exec lsv-gate grep -c 't3_pyspy' $SCHED)"

XG=$(docker exec lsv-gate /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

ck serve_spec_mtp4 1 "$(docker exec lsv-gate grep -cF '"method":"mtp","num_speculative_tokens":4' /root/serve_user.sh)"

echo "--- pedigree markers (full chain v1218..v1225; v1223 RESTORED — was missing from gates_v1224_lane.sh) ---"
for M in v1218 v1220 v1221 v1222 v1223 v1224 v1225; do
  if docker exec lsv-gate test -f /root/.llm_scaler_exp_${M}_baked; then
    echo "GATE-OK   ${M}_pedigree_marker"
  else
    echo "GATE-FAIL ${M}_pedigree_marker"; FAIL=1
  fi
done
for M in v64 v1220 v1221 v1222 v1223 v1224 v1225; do
  if docker exec lsv-gate test -f /root/.${M}_jit_warmed; then
    echo "GATE-OK   ${M}_jit_warm_stamp"
  else
    echo "GATE-FAIL ${M}_jit_warm_stamp"; FAIL=1
  fi
done
if docker exec lsv-gate grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--block-size 64' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--speculative-config' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--max-num-batched-tokens 8192' /root/serve_user.sh \
   && docker exec lsv-gate grep -q -- '--async-scheduling' /root/serve_user.sh \
   && ! docker exec lsv-gate grep -q -- '--default-chat-template-kwargs' /root/serve_user.sh \
   && ! docker exec lsv-gate grep -q -- 'chat-template ' /root/serve_user.sh; then
  echo "GATE-OK   baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON mnbt8192 async-ON no-template)"
else
  echo "GATE-FAIL baked_serve_config"; FAIL=1
fi

echo "--- warm triton cache shipped in the image (warmed under the v123 async posture) ---"
docker exec lsv-gate sh -c 'for d in /root/.triton/cache; do [ -d "$d" ] && echo "cache_dir=$d entries=$(ls "$d" | wc -l)"; done'
docker exec lsv-gate sh -c 'ls -lt /root/.triton/cache | head -5'
RAWT=$(docker run --rm --entrypoint /bin/sh llm-scaler-exp:v1.2.25-raw -c 'ls /root/.triton/cache 2>/dev/null | wc -l' | tr -d '[:space:]')
SHIPN=$(docker exec lsv-gate sh -c 'ls /root/.triton/cache 2>/dev/null | wc -l' | tr -d '[:space:]')
echo "shipped_triton_cache_entries=$SHIPN raw_image_triton_cache_entries=$RAWT (floor: shipped >= raw)"
ck triton_cache_ge_raw 1 "$(docker exec lsv-gate sh -c "ls /root/.triton/cache 2>/dev/null | wc -l | awk -v r=$RAWT '{print (\$1>=r)?1:0}'")"

docker rm -f lsv-gate >/dev/null 2>&1 || true
if [ "$FAIL" != "0" ]; then
  echo "=== V1225 SHIP GATES: FAIL $(date +%T) ==="
else
  echo "=== V1225 SHIP GATES: ALL PASS $(date +%T) ==="
fi
} > "$L" 2>&1
tail -80 "$L"
grep -q 'GATES: ALL PASS' "$L" || exit 1
exit 0
