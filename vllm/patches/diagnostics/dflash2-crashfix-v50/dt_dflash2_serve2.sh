#!/bin/bash
# dt_dflash2_serve2.sh — crash-repro variant of dt_dflash2_serve.sh.
# Adds:
#   KVDTYPE          — --kv-cache-dtype value (default turboquant_4bit_nc,
#                      i.e. the validated lane). e.g. fp8_e4m3 (user crash 1),
#                      auto, turboquant_k8v4.
#   DFLASH2_DBG_ASSERT=1 — runs /root/dt_instr_fp8.py (KDESCALE_DBG print
#                      before the flash-attn descale assert) before serve.
# Everything else identical to dt_dflash2_serve.sh (same flags, same knobs,
# same k=4 guard).
set -u
ASYNC_FLAG="--async-scheduling"
if [ "${DFLASH2_SYNC:-0}" = "1" ]; then ASYNC_FLAG="--no-async-scheduling"; fi
if [ "${DFLASH2_EMIT_K:-}" = "4" ] && [ "${DFLASH2_CGMODE:-FULL_DECODE_ONLY}" != "NONE" ]; then
  echo "FATAL: explicit DFLASH2_EMIT_K=4 corrupts numerics under graphs (v42 conviction)."
  exit 2
fi
# DFLASH2_SPEC: dflash7 (default, the lane) | dflash4 | mtp1 | mtp4 | mtp7 | none (no spec config)
SPEC_MODE=${DFLASH2_SPEC:-dflash7}
SPEC_ARGS=""
case "$SPEC_MODE" in
  dflash7) SPEC_ARGS="--speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":7}'" ;;
  dflash4) SPEC_ARGS="--speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":4}'" ;;
  mtp1)    SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'" ;;
  mtp4)    SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'" ;;
  mtp7)    SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":7}'" ;;
  none)    SPEC_ARGS="" ;;
  *) echo "FATAL: unknown DFLASH2_SPEC=$SPEC_MODE"; exit 2 ;;
esac

# v50 F7: MTP k>=4 = convicted CORRUPT temp-0 lane — refuse regardless of
# graph mode. Evidence (v50 cells C/H/I 2026-09-06, tq4nc KV): mtp4 +
# graphs = byte-reproducible garbage (' is a 2005 film directed by David
# O. Russell'); mtp4 EAGER = garbage too AND nondeterministic across 4
# probes (v29c 'eager = reference' claim OVERTURNED on this stack);
# mtp1 + graphs = CORRECT (' Paris.' x3, battery clean, 13.0 tok/s @65k).
# dflash lanes unaffected (dflash7 = certified prod lane).
case "$SPEC_MODE" in
  mtp4|mtp7)
    echo "FATAL (v50 F7): DFLASH2_SPEC=$SPEC_MODE refused - MTP k>=4 temp-0 outputs are CORRUPT under graphs (byte-reproducible garbage) AND in eager mode (nondeterministic garbage); v50 cells C/I 2026-09-06. mtp1 is validated clean; dflash7 is the certified lane." >&2
    exit 1
    ;;
esac

# v50 F6: fp8-class KV + any speculative drafter = convicted broken lane
# (multi-row verify forward on fp8 KV: 0.2-0.9 tok/s + engine-death wedge
# risk; v50 matrix B5/B5b/B5d/B5e 2026-09-05). Refuse at boot; the fp8
# nospec lane is certified healthy (34.4/30.7/22.4 tok/s @2k/16k/65k).
case "${KVDTYPE:-turboquant_4bit_nc}" in
  fp8*)
    if [ "$SPEC_MODE" != "none" ]; then
      echo "FATAL (v50 F6): KVDTYPE=${KVDTYPE} + DFLASH2_SPEC=${SPEC_MODE} refused - fp8-class KV + speculative decoding is a convicted broken lane (0.2-0.9 tok/s, engine-death wedge risk; v50 B5/B5b/B5d/B5e). Use a TurboQuant KV dtype for spec, or DFLASH2_SPEC=none (fp8 nospec lane certified 34.4/30.7/22.4 tok/s @2k/16k/65k)." >&2
      exit 1
    fi
    ;;
esac
# DFLASH2_EMIT_K is only valid with a dflash spec config (range 1..num_spec);
# nospec/mtp lanes must NOT carry it (validator: 'out of range 1..0').
# Default emit_k must not exceed num_spec (dflash4 -> 4, else 7).
case "$SPEC_MODE" in
  dflash4) EMITK_DEFAULT=4 ;;
  *)       EMITK_DEFAULT=7 ;;
esac
EMITK_ENV="-e DFLASH2_EMIT_K=${DFLASH2_EMIT_K:-$EMITK_DEFAULT}"
case "$SPEC_MODE" in
  dflash*) ;;
  *) EMITK_ENV="" ;;
esac
docker rm -f lsv-test >/dev/null 2>&1
docker run -d --name lsv-test \
  --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=${DFLASH2_XPUGRAPH:-1} \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e CCL_TOPO_P2P_ACCESS=1 \
  -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
  -e CCL_ENABLE_SYCL_KERNELS=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CCL_ZE_IPC_EXCHANGE=drmfd \
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
  -e DFLASH_DEBUG=1 \
  -e DFLASH_SYNC_END=${DFLASH_SYNC_END:-0} \
  -e DFLASH_ATTN_DBG=${DFLASH_ATTN_DBG:-0} \
  -e DFLASH_GDN_DBG=${DFLASH_GDN_DBG:-0} \
  -e DFLASH_TQ_DBG=${DFLASH_TQ_DBG:-0} \
  ${EMITK_ENV} \
  -e DFLASH2_CGMODE=${DFLASH2_CGMODE:-FULL_DECODE_ONLY} \
  -e VLLM_SPEC_TIMING=${DFLASH2_SPECTIMING:-0} \
  -e VLLM_SPEC_TIMING_FLUSH=${DFLASH2_SPECFLUSH:-100} \
  -e VLLM_TQ_TIME=${DFLASH2_TQTIME:-0} \
  -e VLLM_XPU_DFLASH_TP1=${DFLASH2_TP1:-0} \
  -e VLLM_DFLASH_DRAFT_KV_DTYPE=${VLLM_DFLASH_DRAFT_KV_DTYPE:-} \
  -v /root/build/dt_instr_fp8.py:/root/dt_instr_fp8.py:ro \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/qwen3.8-27b-dflash2:/models/dflash2:ro \
  ${TRITON_CACHE_VOL:+-v ${TRITON_CACHE_VOL}:/root/.triton/cache} \
  ${EXTRA_VOL:+-v ${EXTRA_VOL}} \
  ${EXTRA_VOL2:+-v ${EXTRA_VOL2}} \
  ${EXTRA_VOL3:+-v ${EXTRA_VOL3}} \
  ${EXTRA_VOL4:+-v ${EXTRA_VOL4}} \
  ${EXTRA_VOL5:+-v ${EXTRA_VOL5}} \
  ${EXTRA_VOL6:+-v ${EXTRA_VOL6}} \
  ${SYCL_CACHE_VOL:+-v ${SYCL_CACHE_VOL}:/root/.cache/sycl_cache} \
  -e SYCL_CACHE_PERSISTENT=${DFLASH2_SYCLCACHE:-0} \
  -e SYCL_CACHE_DIR=/root/.cache/sycl_cache \
  -w /llm-scaler/vllm \
  --entrypoint /bin/bash \
  ${DFLASH2_IMG:-llm-scaler-prod:v1} -c "if [ \"${DFLASH2_DBG_ASSERT:-0}\" = \"1\" ]; then python3 /root/dt_instr_fp8.py; fi; if [ -f /root/dt_warmup.py ] && [ \"${DFLASH2_WARMUP:-1}\" = \"1\" ]; then ( python3 /root/dt_warmup.py >> /root/dt_warmup_sidecar.log 2>&1 & ) ; fi; if [ -f /root/.dflash2_patched ]; then exec /opt/venv/bin/python3 /opt/venv/bin/vllm serve \
    --model /models/target \
    --served-model-name qwen3.8-27b-fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization ${DFLASH2_MEMUTIL:-0.85} \
    --max-model-len ${MAXLEN:-262144} \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --block-size ${BLOCKSIZE:-512} \
    --dtype float16 \
    --kv-cache-dtype ${KVDTYPE:-turboquant_4bit_nc} \
    --mamba-ssm-cache-dtype float16 \
  $ASYNC_FLAG \
    --enable-prefix-caching \
    --trust-remote-code \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    ${SPEC_ARGS} \
    --compilation-config \"{\\\"cudagraph_mode\\\":\\\"\$DFLASH2_CGMODE\\\"}\" \
    --port 8000 > /root/dt_dflash2.log 2>&1; else echo AWAIT_DFLASH2_PATCH; sleep 900; fi"
echo "booted dt_dflash2 (repro) log=/root/dt_dflash2.log emit_k=${DFLASH2_EMIT_K:-7} kv=${KVDTYPE:-turboquant_4bit_nc} dbg_assert=${DFLASH2_DBG_ASSERT:-0}"
