#!/bin/bash
# stage5_bake_v1219.sh — gate, bake, and boot llm-scaler-exp:v1.2.19.
# v1.2.19 = v1.2.18 image + EXTENDED in-bake JIT warm, NO engine code
# change (v65 Phase-0/Phase-1 plateau posture sealed). The v63 warm
# covered only solo-big-prefill + single-session decode; every fresh
# v1.2.18 boot still JIT-compiles on first contended/big-turn traffic:
# expand_kernel (2.1-2.2 s), eagle_prepare_inputs/next_token_padded,
# eagle_step_slot_mapping_metadata, rejection_greedy_sample,
# _zero_kv_blocks, _compute_slot_mapping, batch_memcpy (v52f COW).
# v1219 extends the warm: legacy v63 warm + concurrent multi-session
# multi-turn round (warm_ext_v1219.py) + xgrammar + a round-2 replay
# gated on ZERO new jit warnings. Spec (MTP x4, BARRIER=2) + XGrammar-2
# preserved untouched.
#
# Bake source: FRESH container from llm-scaler-exp:v1.2.18 (NOT the
# running lsv-test lane). Lane MUST BE DOWN first (bake serves on :8000
# for the JIT warm). Content-only gates pre-commit after serve stopped.
set -u
L=/root/build/lce1/v1219_bake.log
{
echo "=== V1219 BAKE v1.2.19 $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm
FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

if curl -s -o /dev/null -m 3 http://localhost:8000/health; then
  echo "BAKE ABORT: lane :8000 still up (tear down first)"; exit 1
fi

echo "--- fresh bake container from v1.2.18 (no code patcher: content identical to v1.2.18) ---"
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
    llm-scaler-exp:v1.2.18 || { echo "BAKE ABORT: container run failed"; exit 1; }

echo "--- extended JIT warm inside bake (serve + warm rounds + zero-JIT gate) ---"
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

JITPAT='expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel'

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

docker exec lsv-bake touch /root/.v1219_jit_warmed
docker exec lsv-bake bash -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do
  sleep 2
  docker exec lsv-bake bash -c "pgrep -f 'vllm serve' >/dev/null" || break
done
echo "bake serve stopped"

echo "--- content gates ---"
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

ck mark_sched_v64 2 "$(docker exec lsv-bake grep -c 'llm-scaler v64 TTFTFIX-2' $SCHED)"
ck v64_def_k      1 "$(docker exec lsv-bake grep -c 'VLLM_V64_DECODE_INTERLEAVE", "2"' $SCHED)"
ck v64_def_b      1 "$(docker exec lsv-bake grep -c 'VLLM_V64_DECODE_BUDGET", "512"' $SCHED)"
ck v64_active_str 1 "$(docker exec lsv-bake grep -c 'V64_INTERLEAVE_ACTIVE' $SCHED)"

echo "--- v65 instrumentation absence (test-only probes NEVER baked) ---"
ck no_v65_sched   0 "$(docker exec lsv-bake grep -cE 'VLLM_V65|V65_STEP_LOG|V65_LAYER_LOG' $SCHED)"
ck no_v65_flash   0 "$(docker exec lsv-bake grep -cE 'VLLM_V65|V65_ATTN|V65_GDN|V65_LAYER' $SP/v1/attention/backends/flash_attn.py)"
ck no_v65_xpu     0 "$(docker exec lsv-bake grep -cE 'VLLM_V65|V65_FORCE_TRITON' $SP/platforms/xpu.py)"

BARR=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_default "False True 0" "$BARR"

XG=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

docker exec lsv-bake test -f /root/.llm_scaler_exp_v1218_baked && echo "GATE-OK   v1218_pedigree_marker" || { echo "GATE-FAIL v1218_pedigree_marker"; FAIL=1; }
docker exec lsv-bake test -f /root/.v64_jit_warmed && echo "GATE-OK   v64_jit_warm_stamp" || { echo "GATE-FAIL v64_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1219_jit_warmed && echo "GATE-OK   v1219_jit_warm_stamp" || { echo "GATE-FAIL v1219_jit_warm_stamp"; FAIL=1; }
if docker exec lsv-bake grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--block-size 64' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--speculative-config' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--max-num-batched-tokens 8192' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- '--async-scheduling' /root/serve_user.sh; then
  echo "GATE-OK   baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON mnbt8192 async-OFF)"
else
  echo "GATE-FAIL baked_serve_config"; FAIL=1
fi

if [ "$FAIL" != "0" ]; then
  echo "BAKE ABORTED — gates failed"; docker rm -f lsv-bake >/dev/null 2>&1 || true
  echo "=== BAKE DONE (ABORT) $(date +%T) ==="; exit 1
fi

echo "--- gates PASS: committing v1.2.19 ---"
docker exec lsv-bake touch /root/.llm_scaler_exp_v1219_baked
docker commit lsv-bake llm-scaler-exp:v1.2.19
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.19 --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
docker rm -f lsv-bake >/dev/null 2>&1 || true

echo "--- build repro_bootV1219.sh from V1218 snapshot (image + marker only; all patches already baked) ---"
[ -f /root/build/repro_bootV1218_snapshot.sh ] || cp /root/build/repro_bootV1218.sh /root/build/repro_bootV1218_snapshot.sh
B=/root/build/repro_bootV1219.sh
cp /root/build/repro_bootV1218_snapshot.sh "$B"
sed -i 's/llm-scaler-exp:v1\.2\.18/llm-scaler-exp:v1.2.19/g' "$B"
sed -i 's|test -f /root/.llm_scaler_exp_v1218_baked|test -f /root/.llm_scaler_exp_v1219_baked|' "$B"
grep -nE 'v1\.2\.19|v1219_baked' "$B" | head -8

echo "=== BAKE DONE (image committed, boot script ready) $(date +%T) ==="
} > "$L" 2>&1
tail -80 "$L"
echo STAGE5_BAKE_V1219_DONE
