#!/bin/bash
# Reference serving command for qwen3.8-27b-fp8 on llm-scaler-vllm-adv:v18
# (2x Intel Arc Pro B70, TP=2). Evolves patches/qwen38-dflash/serve.sh:
#   - gmu 0.8 restored (oneCCL arena pre-alloc fix, KNOWN_ISSUES #05e)
#   - KV_DTYPE selectable: unset (default bf16) | fp8 | turboquant_4bit_nc |
#     turboquant_k8v4  (with turboquant + drafter, spec auto-disables: #05b)
#   - SPEC=0 drops the drafter (target-only arms)
#   - SUPERVISED=1 wraps serve in /opt/serve_supervised.sh (#03)
#   - GENCFG default "default" avoids the generation_config.json temp-1.0
#     inheritance trap (PERF_TUNING: default-sampling trap)
#
# Usage:
#   TARGET_DIR=/path/to/qwen3.8-27b-fp8 DRAFTER_DIR=/path/to/drafter-fp8-v5 ./serve.sh
#   SPEC=0 TARGET_DIR=... ./serve.sh                          # no drafter
#   KV_DTYPE=fp8 TARGET_DIR=... DRAFTER_DIR=... ./serve.sh    # fp8 KV
#   SUPERVISED=1 TARGET_DIR=... DRAFTER_DIR=... ./serve.sh    # auto-restart
set -euo pipefail
TARGET_DIR="${TARGET_DIR:?set TARGET_DIR}"
SPEC="${SPEC:-1}"
if [ "$SPEC" = "1" ]; then DRAFTER_DIR="${DRAFTER_DIR:?set DRAFTER_DIR (or SPEC=0)}"; fi
IMAGE="${IMAGE:-llm-scaler-vllm-adv:v18}"
NAME="${NAME:-lsv-test}"
PORT="${PORT:-8000}"
SPEC_K="${SPEC_K:-4}"
KV_DTYPE="${KV_DTYPE:-}"          # unset = model default
GMU="${GMU:-0.8}"                 # 0.8 restored by the #05e arena fix
MAXLEN="${MAXLEN:-64000}"
MAXSEQS="${MAXSEQS:-64}"
MNBT="${MNBT:-4096}"
GENCFG="${GENCFG:-vllm}"   # fork accepts auto|vllm|<path>; vllm = safe defaults
DTYPES="${DTYPES:-bfloat16}"

ARGS=(serve --host 0.0.0.0 --port "$PORT"
  --model /models/target
  --served-model-name qwen3.8-27b-fp8
  --tensor-parallel-size 2
  --gpu-memory-utilization "$GMU"
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MNBT"
  --max-num-seqs "$MAXSEQS"
  --block-size 64
  --dtype "$DTYPES"
  --mamba-ssm-cache-dtype float16
  --async-scheduling
  --generation-config "$GENCFG"
)
[ -n "$KV_DTYPE" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
if [ "$SPEC" = "1" ]; then
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/drafter\",\"num_speculative_tokens\":${SPEC_K}}")
fi

DRAFTER_MOUNT=()
if [ "$SPEC" = "1" ]; then
  DRAFTER_MOUNT=(-v "${DRAFTER_DIR}:/models/drafter:ro")
fi

ENTRY=/opt/venv/bin/vllm
if [ "${SUPERVISED:-0}" = "1" ]; then ENTRY=/opt/serve_supervised.sh; fi

exec docker run --rm --name "$NAME" \
  --device /dev/dri/card1 --device /dev/dri/card2 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly \
  --network host --shm-size 32g --ipc=host \
  -v "${TARGET_DIR}":/models/target:ro \
  "${DRAFTER_MOUNT[@]}" \
  --entrypoint "$ENTRY" \
  "$IMAGE" \
  "${ARGS[@]}"
