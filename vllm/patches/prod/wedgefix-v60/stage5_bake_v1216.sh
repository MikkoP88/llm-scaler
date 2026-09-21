#!/bin/bash
# stage5_bake_v1216.sh — gate, bake, and boot llm-scaler-exp:v1.2.16.
# Pre-req (ALL PASS on boot 9, 2026-09-22): C4 crash-probe x3 clean,
# S1/L2/C4 perf >= drain reference (C4 avg +22%), XGrammar-2 + thinking
# probes all 200, 3x14-phase wedge drill SUSTAIN_COMPLETE_NO_WEDGE,
# stage4 soak+battery FULL PASS (err=0, resets 0, health 200).
#
# v1.2.16 = v1.2.15 image + WEDGEFIX-G (v62b bake patcher): barrier
# default flips from mode 1 (v60 full-device drain, -9..-31% decode)
# to mode 2 (HOST-side /dev/shm flock rendezvous, ~free). v61
# replicated-drafter needles are NOT in v1.2.15 and are NOT baked
# (dead end: deterministic C4 DEVICE_LOST + -40% perf).
#
# Bake source: FRESH container from llm-scaler-exp:v1.2.15 (NOT the
# running lsv-test lane, which carries v61 test needles + diag
# patcher). Content-only gates pre-commit (no serving, no :8000
# conflict with the standing lane). Spec + XGrammar-2 preserved.
set -u
L=/root/build/lce1/v62_bake.log
{
echo "=== V62 BAKE v1.2.16 $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm
FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

echo "--- fresh bake container from v1.2.15 ---"
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
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.15 || { echo "BAKE ABORT: container run failed"; exit 1; }

echo "--- apply v62b (WEDGEFIX-G bake variant) ---"
docker cp /root/build/patch_wedge_v62b_bake.py lsv-bake:/root/patch_wedge_v62b_bake.py
docker exec lsv-bake /opt/venv/bin/python3 /root/patch_wedge_v62b_bake.py --apply

echo "--- content gates ---"
GMR=$SP/v1/worker/gpu_model_runner.py
ck mark_g62_def   1 "$(docker exec lsv-bake grep -c 'llm-scaler v62 (WEDGEFIX-G): barrier default mode 2' $GMR)"
ck mark_g62_mach  1 "$(docker exec lsv-bake grep -c 'llm-scaler v62 (WEDGEFIX-G): HOST-side spec draft barrier' $GMR)"
ck mark_g62_site  1 "$(docker exec lsv-bake grep -c 'llm-scaler v62 (WEDGEFIX-G): mode dispatch' $GMR)"
ck no_v61_marks   0 "$(docker exec lsv-bake grep -rc 'llm-scaler v61' $SP/v1/worker/gpu_model_runner.py $SP/model_executor/models/qwen3_5_mtp.py $SP/v1/spec_decode/llm_base_proposer.py $SP/v1/worker/logits_processor.py 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')"
ck no_v61_repl    0 "$(docker exec lsv-bake grep -c 'REPLICATED_DRAFT' $GMR)"
ck no_v61_diag    0 "$(docker exec lsv-bake grep -c 'VLLM_V61_DIAG' $GMR)"

ck mark_xpu_v60   1 "$(docker exec lsv-bake grep -c 'llm-scaler v60' $SP/platforms/xpu.py)"
ck mark_gmr_v60b  1 "$(docker exec lsv-bake grep -c 'llm-scaler v60 (wedge fix): barrier DEFAULT ON' $GMR)"
ck mark_gmr_v60g  1 "$(docker exec lsv-bake grep -c 'llm-scaler v60g (WEDGEFIX-E)' $GMR)"
ck mark_gmr_v60e  0 "$(docker exec lsv-bake grep -c 'llm-scaler v60e' $GMR)"
ck mark_f15b      1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler f15b' $GMR | awk '{print (\$1>=1)?1:0}'")"
ck mark_v553      1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v55.3' $GMR | awk '{print (\$1>=1)?1:0}'")"
ck mark_arstage   1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler arstage' $SP/distributed/device_communicators/xpu_communicator.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_stalfix   1 "$(docker exec lsv-bake sh -c "grep -c STALFIX $SP/entrypoints/openai/responses/serving.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_sched_v60c 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v60c' $SP/v1/core/sched/scheduler.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_parser    1 "$(docker exec lsv-bake sh -c "grep -c _stalfix_name_emitted $SP/tool_parsers/qwen3xml_tool_parser.py | awk '{print (\$1>=1)?1:0}'")"

BARR=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_default "False True 0" "$BARR"

XG=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

docker exec lsv-bake test -f /root/.llm_scaler_exp_v1215_baked && echo "GATE-OK   v1215_pedigree_marker" || { echo "GATE-FAIL v1215_pedigree_marker"; FAIL=1; }
if docker exec lsv-bake grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--block-size 64' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
   && docker exec lsv-bake grep -q -- '--speculative-config' /root/serve_user.sh \
   && ! docker exec lsv-bake grep -q -- '--async-scheduling' /root/serve_user.sh; then
  echo "GATE-OK   baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON async-OFF)"
else
  echo "GATE-FAIL baked_serve_config"; FAIL=1
fi

if [ "$FAIL" != "0" ]; then
  echo "BAKE ABORTED — gates failed"; docker rm -f lsv-bake >/dev/null 2>&1 || true
  echo "=== BAKE DONE (ABORT) $(date +%T) ==="; exit 1
fi

echo "--- gates PASS: committing v1.2.16 ---"
docker exec lsv-bake touch /root/.llm_scaler_exp_v1216_baked
docker commit lsv-bake llm-scaler-exp:v1.2.16
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.16 --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
docker rm -f lsv-bake >/dev/null 2>&1 || true

echo "--- build repro_bootV1216.sh from V1215 snapshot ---"
B=/root/build/repro_bootV1216.sh
cp /root/build/repro_bootV1215_snapshot.sh "$B"
sed -i 's/llm-scaler-exp:v1\.2\.15/llm-scaler-exp:v1.2.16/g' "$B"
sed -i 's/VLLM_XPU_SPEC_DRAFT_BARRIER=1/VLLM_XPU_SPEC_DRAFT_BARRIER=2/' "$B"
sed -i 's|docker exec lsv-test pip install -q py-spy|docker cp /root/build/patch_wedge_v62b_bake.py lsv-test:/root/patch_wedge_v62b_bake.py\ndocker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v62b_bake.py --apply \| tee /root/build/lce1/v62b_apply_bootV1216.log\ndocker exec lsv-test grep -c "llm-scaler v62 (WEDGEFIX-G): mode dispatch" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py\n&|' "$B"
sed -i 's|test -f /root/.llm_scaler_exp_v1212_baked|test -f /root/.llm_scaler_exp_v1216_baked|' "$B"
grep -nE 'v1\.2\.16|SPEC_DRAFT_BARRIER=|v62b|v1216_baked' "$B" | head -10

echo "=== BAKE DONE (image committed, boot script ready) $(date +%T) ==="
} > "$L" 2>&1
tail -45 "$L"
echo STAGE5_BAKE_DONE
