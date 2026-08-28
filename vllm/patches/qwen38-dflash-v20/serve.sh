#!/bin/bash
# Reference serving command for qwen3.8-27b-fp8 on llm-scaler-vllm-adv:v19
# (2x Intel Arc Pro B70, TP=2). Evolves patches/qwen38-dflash-v18/serve.sh:
#   - v19: turboquant KV + dflash drafter now serve TOGETHER (#05b guard
#     flipped: VLLM_ALLOW_TQ_SPEC defaults 1; =0 rolls back to target-only)
#   - v19: TQ verify steps use the multi-query kernel (VLLM_TQ_MQ_VERIFY,
#     default 1; =0 restores the v18 synthetic-decode path)
#   - gmu 0.8 restored (oneCCL arena pre-alloc fix, KNOWN_ISSUES #05e)
#   - KV_DTYPE selectable: unset (default bf16) | fp8 | turboquant_4bit_nc |
#     turboquant_k8v4
#   - SPEC=0 drops the drafter (target-only arms)
#   - SUPERVISED=1 wraps serve in /opt/serve_supervised.sh (#03)
#   - GENCFG default "default" avoids the generation_config.json temp-1.0
#     inheritance trap (PERF_TUNING: default-sampling trap)
#
# Usage:
#   TARGET_DIR=/path/to/qwen3.8-27b-fp8 DRAFTER_DIR=/path/to/drafter-fp8-v5 ./serve.sh
#   SPEC=0 TARGET_DIR=... ./serve.sh                          # no drafter
#   KV_DTYPE=turboquant_4bit_nc TARGET_DIR=... DRAFTER_DIR=... ./serve.sh  # TQ + spec (v19)
#   SUPERVISED=1 TARGET_DIR=... DRAFTER_DIR=... ./serve.sh    # auto-restart
set -euo pipefail
TARGET_DIR="${TARGET_DIR:?set TARGET_DIR}"
SPEC="${SPEC:-1}"
if [ "$SPEC" = "1" ]; then DRAFTER_DIR="${DRAFTER_DIR:?set DRAFTER_DIR (or SPEC=0)}"; fi
IMAGE="${IMAGE:-llm-scaler-vllm-adv:v20}"
NAME="${NAME:-lsv-test}"
PORT="${PORT:-8000}"
SPEC_K="${SPEC_K:-4}"
KV_DTYPE="${KV_DTYPE:-}"          # unset = model default
GMU="${GMU:-0.8}"                 # 0.8 restored by the #05e arena fix
MAXLEN="${MAXLEN:-64000}"
MAXSEQS="${MAXSEQS:-64}"
MNBT="${MNBT:-4096}"
BLOCKSIZE="${BLOCKSIZE:-64}"
GENCFG="${GENCFG:-vllm}"   # fork accepts auto|vllm|<path>; vllm = safe defaults
DTYPES="${DTYPES:-bfloat16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"      # extra vllm serve flags (space-separated)

ARGS=(serve --host 0.0.0.0 --port "$PORT"
  --model /models/target
  --served-model-name qwen3.8-27b-fp8
  --tensor-parallel-size 2
  --gpu-memory-utilization "$GMU"
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MNBT"
  --max-num-seqs "$MAXSEQS"
  --block-size "$BLOCKSIZE"
  --dtype "$DTYPES"
  --mamba-ssm-cache-dtype float16
  --async-scheduling
  --generation-config "$GENCFG"
)
[ -n "$KV_DTYPE" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
if [ "$SPEC" = "1" ]; then
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/drafter\",\"num_speculative_tokens\":${SPEC_K}}")
fi
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

DRAFTER_MOUNT=()
if [ "$SPEC" = "1" ]; then
  DRAFTER_MOUNT=(-v "${DRAFTER_DIR}:/models/drafter:ro")
fi

ENTRY=/opt/venv/bin/vllm
if [ "${SUPERVISED:-0}" = "1" ]; then ENTRY=/opt/serve_supervised.sh; fi

# Forward vllm env knobs only when explicitly set (empty-string injection
# would defeat in-code defaults like os.getenv(..., "1") == "1").
ENV_ARGS=()
for _v in VLLM_XPU_TQ_SAFE_WARMUP VLLM_XPU_SPEC_SAFE_WARMUP \
          VLLM_TQ_MQ_VERIFY VLLM_ALLOW_TQ_SPEC \
          VLLM_TQ_VERIFY_GRAPH_FIX VLLM_ESIMD_F8_SCALE_FIX \
          VLLM_SPEC_TIMING VLLM_SPEC_TIMING_FLUSH \
          VLLM_DFLASH_FULL_GRAPH; do
  if [ -n "${!_v:-}" ]; then ENV_ARGS+=(-e "${_v}=${!_v}"); fi
done

exec docker run --rm --name "$NAME" \
  --device /dev/dri/card1 --device /dev/dri/card2 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly \
  --network host --shm-size 32g --ipc=host \
  -e VLLM_ALLOW_LONG_MODEL_LEN=1 \
  "${ENV_ARGS[@]}" \
  -v "${TARGET_DIR}":/models/target:ro \
  "${DRAFTER_MOUNT[@]}" \
  --entrypoint "$ENTRY" \
  "$IMAGE" \
  "${ARGS[@]}"
