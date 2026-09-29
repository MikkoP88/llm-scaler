#!/bin/bash
# stage5_bake_v1224.sh — gate and bake llm-scaler-exp:v1.2.24-raw (production
# v1.2.24 = clean warm lane-commit AFTER full validation + wedge battery, with
# the complete gate battery re-run on the committed image — v1222/v1223
# precedent). v1.2.24 = v1.2.23-raw + TWO v124 changes (NO kernel wheel
# changes — v88 wheel inherited; no scheduler changes — v63/v64/v66 stay;
# no serve-config changes — async+coder+85-sizes+barrier-default-0 inherited
# from v1.2.23-raw and only GATED here, never re-applied):
#   1. P16 non-spec pipeline fix (round 3, user directive "fix non-spec
#      pipeline to also working"): xpu.py v31.1 compile gate extended from
#      (spec AND TP>1) to (TP>1) — spec-off boots previously died in dynamo
#      on arstage hasattr(StreamVariable, xpu_stream) at _arstage.py:98.
#      Validated live (P16-b: spec0 boots ~125 s, gate fires, solo 30.3 vs
#      74.2 spec4 = spec x2.45). Production spec4 posture unchanged.
#   2. P19.5a fp8 mamba/GDN SSM-cache plumbing (user directive: implement
#      --mamba-ssm-cache-dtype fp8_e4m3 + fp8_e5m2): MambaDType Literal,
#      real-fp8 dtype resolution in _mamba_state_dtype, SYCL-op fp8 bridge
#      (uint8-view scatter; index_copy_xpu unimplemented for fp8). INERT on
#      float16 (baked default) — _fp8_ssm short-circuits; flag availability
#      only. Stage-A verdict: e4m3 quality-viable/e5m2 degenerate; speed
#      unlock = future kernel-native round (P19.5b/c).
# Spec MTP x4 + XGrammar-2 0.2.7 support ASSERTED by gates (hard requirement).
# Bake source: FRESH container from llm-scaler-exp:v1.2.23-raw (NOT the
# running lane). Lane MUST BE DOWN first (bake serves on :8000 for warm).
set -u
L=/root/build/lce1/v1224_bake.log
{
echo "=== V1224 BAKE v1.2.24 $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm
FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

if curl -s -o /dev/null -m 3 http://localhost:8000/health; then
  echo "BAKE ABORT: lane :8000 still up (tear down first)"; exit 1
fi
for f in p16_fix_nonspec_gate.py p195a_python_plumbing.py patch_sched_v66_bake.py \
         probe_jitwarm_v63.py warm_ext_v1219.py t2_xgrammar_probe.py repro_bootV1223.sh; do
  test -f /root/build/$f || { echo "BAKE ABORT: missing /root/build/$f"; exit 1; }
done

echo "--- fresh bake container from v1.2.23-raw ---"
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
    -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 \
    -e VLLM_RPC_TIMEOUT=60000 \
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.23-raw || { echo "BAKE ABORT: container run failed"; exit 1; }

echo "--- v88 kernels wheel INHERITED (no reinstall): version gate ---"
WV=$(docker exec lsv-bake /opt/venv/bin/pip show vllm-xpu-kernels 2>/dev/null | grep '^Version' | head -1)
echo "wheel: $WV"
echo "$WV" | grep -q 'd20260925' || { echo "BAKE ABORT: v88 wheel not present in base image"; exit 1; }

echo "--- no debug instrumentation in baked image ---"
ck no_gdn_capture 0 "$(docker exec lsv-bake grep -c 'VLLM_GDN_CAPTURE' $SP/_xpu_ops.py)"

echo "--- v124 patch 1: P16 non-spec compile gate (TP>1, spec or not) ---"
docker cp /root/build/p16_fix_nonspec_gate.py lsv-bake:/root/p16_fix_nonspec_gate.py
docker exec lsv-bake /opt/venv/bin/python3 /root/p16_fix_nonspec_gate.py
P16CHK=$(docker exec lsv-bake /opt/venv/bin/python3 /root/p16_fix_nonspec_gate.py | tail -1 | grep -c -e PATCHED_OK -e ALREADY)
ck p16_patch_exit 1 "$P16CHK"
ck p16_markers 2 "$(docker exec lsv-bake sh -c "grep -c 'v124 P16' $SP/platforms/xpu.py | awk '{print (\$1>=2)?2:0}'")"
ck p16_syntax 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py").read()); print(1)' | tail -1)"

echo "--- v124 patch 2: P19.5a fp8 mamba SSM-cache plumbing (INERT on float16) ---"
docker cp /root/build/p195a_python_plumbing.py lsv-bake:/root/p195a_python_plumbing.py
docker exec lsv-bake /opt/venv/bin/python3 /root/p195a_python_plumbing.py
P195CHK=$(docker exec lsv-bake /opt/venv/bin/python3 /root/p195a_python_plumbing.py | tail -1 | grep -c P195A_PATCHED_OK)
ck p195_patch_exit 1 "$P195CHK"
ck p195_literal 1 "$(docker exec lsv-bake grep -c 'fp8_e4m3", "fp8_e5m2' $SP/config/cache.py)"
ck p195_dtype_e4m3 1 "$(docker exec lsv-bake grep -c 'temporal_state_dtype = torch.float8_e4m3fn' $SP/model_executor/layers/mamba/mamba_utils.py)"
ck p195_dtype_e5m2 1 "$(docker exec lsv-bake grep -c 'temporal_state_dtype = torch.float8_e5m2' $SP/model_executor/layers/mamba/mamba_utils.py)"
ck p195_bridge 1 "$(docker exec lsv-bake grep -c 'v124 P19.5a: fp8 SSM cache bridge' $SP/_xpu_ops.py)"
ck p195_scatter_uint8 1 "$(docker exec lsv-bake grep -c 'ssm_pool.view(torch.uint8).index_copy_' $SP/_xpu_ops.py)"
# runtime resolution: fp8 flag -> REAL fp8 temporal dtype, conv stays fp16;
# float16 (baked default) -> identical (fp16, fp16) = inertness proof.
P195RT=$(docker exec -i lsv-bake /opt/venv/bin/python3 - <<'PYEOF'
import torch, typing
from vllm.config.cache import MambaDType
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C
args = set(typing.get_args(MambaDType))
assert {'fp8_e4m3','fp8_e5m2'} <= args, f"Literal missing fp8: {args}"
a = C.gated_delta_net_state_dtype(torch.float16, 'auto', 'fp8_e4m3')
b = C.gated_delta_net_state_dtype(torch.float16, 'auto', 'fp8_e5m2')
c = C.gated_delta_net_state_dtype(torch.float16, 'auto', 'float16')
assert a == (torch.float16, torch.float8_e4m3fn), a
assert b == (torch.float16, torch.float8_e5m2), b
assert c == (torch.float16, torch.float16), c
print(1)
PYEOF
)
ck p195_runtime_resolution 1 "$(echo "$P195RT" | tail -1)"

echo "--- serve-config posture INHERITED from v1.2.23-raw (gated, never re-sed'd) ---"
ck serve_async_sched 1 "$(docker exec lsv-bake grep -c -- '--async-scheduling --port 8000' /root/serve_user.sh)"
ck serve_parser_coder   1 "$(docker exec lsv-bake grep -c -- '--tool-call-parser qwen3_coder' /root/serve_user.sh)"
ck serve_parser_xml_gone 0 "$(docker exec lsv-bake grep -c -- '--tool-call-parser qwen3_xml' /root/serve_user.sh)"
ck serve_capture_sizes 1 "$(docker exec lsv-bake grep -cF '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,' /root/serve_user.sh)"
ck serve_cudagraph_mode 1 "$(docker exec lsv-bake grep -cF '"cudagraph_mode":"FULL_DECODE_ONLY"' /root/serve_user.sh)"
ck serve_mamba_fp16_default 1 "$(docker exec lsv-bake grep -c -- '--mamba-ssm-cache-dtype float16' /root/serve_user.sh)"
ck serve_mamba_fp8_absent 0 "$(docker exec lsv-bake grep -c -- 'mamba-ssm-cache-dtype fp8' /root/serve_user.sh)"
BARR=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_default "False False 0" "$BARR"

echo "--- bake warm serve (async posture) ---"
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
ck bake_async_config 1 "$(docker exec lsv-bake sh -c "grep -c \"'async_scheduling': True\" /root/serve_full.log | awk '{print (\$1>=1)?1:0}'")"
ck bake_graph_count_64 1 "$(docker exec lsv-bake sh -c "grep -c 'Capturing CUDA graphs' /root/serve_full.log | awk '{print (\$1>=1)?1:0}'")"
ck bake_p16_gate_fired 1 "$(docker exec lsv-bake sh -c "grep -c 'inductor compilation disabled' /root/serve_full.log | awk '{print (\$1>=1)?1:0}'")"

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

echo "--- warm round 1c: sampler-shape round (default / top_p / top_k / all, 2000 tok) ---"
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

docker exec lsv-bake touch /root/.v1224_jit_warmed
docker exec lsv-bake bash -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do
  sleep 2
  docker exec lsv-bake bash -c "pgrep -f 'vllm serve' >/dev/null" || break
done
echo "bake serve stopped"

echo "--- content gates (full lineage chain, must hold on v1.2.23-raw base) ---"
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

echo "--- v89/v123/v124 test-artifact absence (test-only, NEVER baked) ---"
ck no_v89_pyspy_probe 0 "$(docker exec lsv-bake grep -c 't3_pyspy' $SCHED)"
ck no_v123_async_variant 0 "$(docker exec lsv-bake grep -c 'serve_user_async' /root/serve_user.sh)"
ck no_v124_serve_variants 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE 'serve_(mfp8|spec0|mnbt16k)' || true")"

XG=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

ck serve_spec_mtp4 1 "$(docker exec lsv-bake grep -cF '"method":"mtp","num_speculative_tokens":4' /root/serve_user.sh)"

docker exec lsv-bake test -f /root/.llm_scaler_exp_v1218_baked && echo "GATE-OK   v1218_pedigree_marker" || { echo "GATE-FAIL v1218_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1220_baked && echo "GATE-OK   v1220_pedigree_marker" || { echo "GATE-FAIL v1220_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1221_baked && echo "GATE-OK   v1221_pedigree_marker" || { echo "GATE-FAIL v1221_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1222_baked && echo "GATE-OK   v1222_pedigree_marker" || { echo "GATE-FAIL v1222_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1223_baked && echo "GATE-OK   v1223_pedigree_marker" || { echo "GATE-FAIL v1223_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.v64_jit_warmed && echo "GATE-OK   v64_jit_warm_stamp" || { echo "GATE-FAIL v64_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1220_jit_warmed && echo "GATE-OK   v1220_jit_warm_stamp" || { echo "GATE-FAIL v1220_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1221_jit_warmed && echo "GATE-OK   v1221_jit_warm_stamp" || { echo "GATE-FAIL v1221_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1222_jit_warmed && echo "GATE-OK   v1222_jit_warm_stamp" || { echo "GATE-FAIL v1222_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1223_jit_warmed && echo "GATE-OK   v1223_jit_warm_stamp" || { echo "GATE-FAIL v1223_jit_warm_stamp"; FAIL=1; }
if docker exec lsv-bake grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--block-size 64' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--speculative-config' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--max-num-batched-tokens 8192' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--async-scheduling' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- '--default-chat-template-kwargs' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- 'chat-template ' /root/serve_user.sh; then
  echo "GATE-OK   baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON mnbt8192 async-ON mamba-fp16 no-template)"
else
  echo "GATE-FAIL baked_serve_config"; FAIL=1
fi

if [ "$FAIL" != "0" ]; then
  echo "BAKE ABORTED — gates failed"; docker rm -f lsv-bake >/dev/null 2>&1 || true
  echo "=== BAKE DONE (ABORT) $(date +%T) ==="; exit 1
fi

echo "--- gates PASS: committing v1.2.24-raw (production v1.2.24 = clean warm lane-commit after full validation, v1223 precedent) ---"
# remove the bake patch scripts (reproducible from the repo; keep the image
# free of round artifacts — v1223 left patch_barrier_v123_bake.py behind,
# not repeated here)
docker exec lsv-bake rm -f /root/p16_fix_nonspec_gate.py /root/p195a_python_plumbing.py
ck no_bake_scripts_left 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE 'p16_fix|p195' || true")"
docker exec lsv-bake touch /root/.llm_scaler_exp_v1224_baked
docker commit lsv-bake llm-scaler-exp:v1.2.24-raw
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.24-raw --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
docker rm -f lsv-bake >/dev/null 2>&1 || true

echo "--- build repro_bootV1224.sh from V1223 (image + marker; async gate already inverted in lineage) ---"
B=/root/build/repro_bootV1224.sh
cp /root/build/repro_bootV1223.sh "$B"
sed -i 's/llm-scaler-exp:v1\.2\.23-raw/llm-scaler-exp:v1.2.24-raw/g' "$B"
sed -i 's|test -f /root/.llm_scaler_exp_v1223_baked|test -f /root/.llm_scaler_exp_v1224_baked|' "$B"
bash -n "$B" || { echo "BOOT SCRIPT SYNTAX FAIL"; exit 1; }
echo "--- diff V1223 -> V1224 boot (expect image + marker lines only) ---"
diff /root/build/repro_bootV1223.sh "$B" || true
grep -nE 'v1\.2\.24|v1224_baked' "$B" | head -8

echo "=== BAKE DONE (image committed, boot script ready) $(date +%T) ==="
} > "$L" 2>&1
tail -100 "$L"
echo STAGE5_BAKE_V1224_DONE
