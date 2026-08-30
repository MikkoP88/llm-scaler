#!/bin/bash
# Reference serving command for qwen3.8-27b(-fp8) on llm-scaler-vllm-adv:v26
# (2x Intel Arc Pro B70, TP=2). Evolves patches/qwen38-dflash-v22/serve.sh:
#   - v22: MTP with graphs works (VLLM_XPU_MTP_EAGER_HEAD, default 1; =0
#     restores the stock crash). The MTP head stays eager, the target keeps
#     its FULL_DECODE_ONLY XPU decode graphs.
#   - v25: MTP draft-step collective shrink (VLLM_XPU_MTP_LOCAL_ARGMAX,
#     default 1; =0 restores v22): _greedy_sample uses the tiny [bs, topk]
#     top-token gather for ALL batch sizes, including bs=1 - the full-vocab
#     (248,320) all-gather per draft step is the #11 long-context wedge
#     driver). Default SPEC_METHOD is now mtp (the v25 fix target).
#   - v26: GDN spec-kernel ragged-batch fix (the #11 wedge ROOT CAUSE) in
#     the rebuilt vllm-xpu-kernels wheel; see README.md / gdn_spec_fix.patch
#     here. v25's two overlays are kept (harmless queue hygiene).
#   - SPEC_METHOD=mtp (default; target-internal head, no drafter mount) |
#     dflash (needs DRAFTER_DIR)
#   - COMP_MODE: --compilation-config cudagraph_mode (default
#     FULL_DECODE_ONLY; "eager" maps to --enforce-eager)
#   - KV_DTYPE selectable: unset (default bf16) | fp8 | turboquant_4bit_nc |
#     turboquant_k8v4
#   - SPEC=0 drops the drafter (target-only arms)
#
# Usage:
#   TARGET_DIR=/path/to/qwen3.8-27b-fp8 DRAFTER_DIR=/path/to/drafter-fp8-v5 ./serve.sh
#   SPEC=0 TARGET_DIR=... ./serve.sh                                  # no drafter
#   KV_DTYPE=turboquant_4bit_nc TARGET_DIR=... DRAFTER_DIR=... ./serve.sh
#   SPEC_METHOD=mtp KV_DTYPE=turboquant_4bit_nc TARGET_DIR=... ./serve.sh   # MTP k4
#   VLLM_XPU_MTP_EAGER_HEAD=0 SPEC_METHOD=mtp TARGET_DIR=... ./serve.sh     # stock A/B
set -euo pipefail
TARGET_DIR="${TARGET_DIR:?set TARGET_DIR}"
SPEC="${SPEC:-1}"
SPEC_METHOD="${SPEC_METHOD:-mtp}"
if [ "$SPEC" = "1" ] && [ "$SPEC_METHOD" = "dflash" ]; then
  DRAFTER_DIR="${DRAFTER_DIR:?set DRAFTER_DIR (or SPEC=0 / SPEC_METHOD=mtp)}"
fi
IMAGE="${IMAGE:-llm-scaler-vllm-adv:v26}"
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
COMP_MODE="${COMP_MODE:-FULL_DECODE_ONLY}"   # FULL_DECODE_ONLY|PIECEWISE|FULL|eager
EXTRA_ARGS="${EXTRA_ARGS:-}"      # extra vllm serve flags (space-separated)

ARGS=(serve --host 0.0.0.0 --port "$PORT"
  --model /models/target
  --served-model-name qwen3.8-27b
  --quantization fp8
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
  --trust-remote-code
)
[ -n "$KV_DTYPE" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
if [ "$COMP_MODE" = "eager" ]; then
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${COMP_MODE}\"}")
fi
if [ "$SPEC" = "1" ]; then
  if [ "$SPEC_METHOD" = "mtp" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_K}}")
  else
    ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/drafter\",\"num_speculative_tokens\":${SPEC_K}}")
  fi
fi
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

DRAFTER_MOUNT=()
if [ "$SPEC" = "1" ] && [ "$SPEC_METHOD" = "dflash" ]; then
  DRAFTER_MOUNT=(-v "${DRAFTER_DIR}:/models/drafter:ro")
fi

# Forward vllm env knobs only when explicitly set (empty-string injection
# would defeat in-code defaults like os.getenv(..., "1") == "1").
ENV_ARGS=()
for _v in VLLM_XPU_TQ_SAFE_WARMUP VLLM_XPU_SPEC_SAFE_WARMUP \
          VLLM_TQ_MQ_VERIFY VLLM_ALLOW_TQ_SPEC \
          VLLM_TQ_VERIFY_GRAPH_FIX VLLM_ESIMD_F8_SCALE_FIX \
          VLLM_SPEC_TIMING VLLM_SPEC_TIMING_FLUSH \
          VLLM_DFLASH_FULL_GRAPH VLLM_DFLASH_TQ_DRAFT_KV \
          VLLM_DFLASH_DRAFT_KV_DTYPE \
          VLLM_XPU_MTP_EAGER_HEAD VLLM_XPU_MTP_LOCAL_ARGMAX \
          VLLM_XPU_SPEC_DRAFT_BARRIER VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX; do
  if [ -n "${!_v:-}" ]; then ENV_ARGS+=(-e "${_v}=${!_v}"); fi
done

exec docker run --rm --name "$NAME" \
  --device /dev/dri/card1 --device /dev/dri/card2 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly \
  --network host --shm-size 32g --ipc host \
  -e VLLM_ALLOW_LONG_MODEL_LEN=1 \
  "${ENV_ARGS[@]}" \
  -v "${TARGET_DIR}":/models/target:ro \
  "${DRAFTER_MOUNT[@]}" \
  --entrypoint /opt/venv/bin/vllm \
  "$IMAGE" \
  "${ARGS[@]}"
