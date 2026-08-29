# qwen38-dflash-v22 — MTP with cudagraphs on XPU (eager MTP head)

`llm-scaler-vllm-adv:v22` = validated v21 image + two env-gated overlays that
make `--speculative-config {"method":"mtp",...}` boot **with**
`--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}` on Intel XPU TP=2.

Rollback: `VLLM_XPU_MTP_EAGER_HEAD=0` restores stock behavior (crash).
Default: `1` (fix on) — XPU-gated, non-XPU platforms untouched.

## The bug (KNOWN_ISSUES #09)

MTP + graphs (any k, any KV dtype, v21 and intel images) died during the
eagle_head torch.compile warmup:

```
sycl invoke_barrier -> ccl allgatherv_large_su_ring<half> -> ... ->
c10d ProcessGroupXCCL::allgather_into_tensor_coalesced ->
all_gather_into_tensor -> pythonFallback (dynamo_eval_custom_code)
!!!!!!! Segfault encountered !!!!!!   (both TP ranks) -> VllmWorker died
```

The MTP head classes (`Qwen3_5MTP`, inner `Qwen3_5MultiTokenPredictor`) are
`@support_torch_compile`-decorated; with graphs mode the drafter is forced to
PIECEWISE (`llm_base_proposer.initialize_cudagraph_keys`) and the head gets
torch.compile'd. The head's sampling path issues oneCCL allgathers (full-vocab
fp16 logits gather via `LogitsProcessor._get_logits/_gather_logits`, or the
padded gather in `get_top_tokens`) from inside dynamo-evaluated code — and
that segfaults. The exact same collectives eager are fine (#05 family: eager
collectives x compiled regions on oneCCL/xe).

v21 isolation matrix: graphs crashed at k=4 AND k=1; `--enforce-eager` booted
at both (E[len] 3.38 @ k4). Raising `CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD`
did not help.

## The fix (VLLM_XPU_MTP_EAGER_HEAD, default 1 on XPU)

1. `qwen3_5_mtp.py` — module-tail `ignore_torch_compile()` on
   `Qwen3_5MultiTokenPredictor`, `Qwen3_5MTP`, `Qwen3_5MoeMTP`: the head
   never enters torch.compile. Both decorated classes must be opted out
   (the decorators machinery only skips the exact class carrying the ignore
   key; the inner `self.model` is separately decorated).
2. `llm_base_proposer.py` — `initialize_cudagraph_keys` forces the drafter to
   `CUDAGraphMode.NONE` (instead of PIECEWISE) when `method=="mtp"` on XPU:
   an eager head has no compiled subgraphs to capture piecewise, and NONE
   selects the `direct_eager_inputs` propose fast path. Method-gated so
   dflash/eagle drafters keep their validated PIECEWISE behavior.

Effect: the TARGET backbone keeps FULL_DECODE_ONLY XPU decode graphs (the
decode-loop win); only the single-layer MTP head runs eager.

## Validation (host 10.20.3.63, 2x B70, TP=2, one boot per cell, 0 resets everywhere)

| cell | image | spec | compile | boot | result |
|---|---|---|---|---|---|
| m1-replica (t1) | v22 | mtp k4 | FULL_DECODE_ONLY | **UP** (v21: CRASH) | coherent text, banner logged, 0 eagle_head compile lines |
| eager A/B (cellA) | v21 | mtp k4 | enforce-eager | UP | E[len] 3.37, identical long-think behavior (attribution) |
| perf (cellB) | v22 | mtp k4 | FULL_DECODE_ONLY | UP | see numbers below |
| user-config (t2) | v22 | dflash k2 | FULL_DECODE_ONLY | UP | no-regression gate vs v21 (see below) |

Canonical workload: "Write a html car game.", temp 0.3 / top_k 20 / top_p 0.95
/ min_p 0 / presence 0 / repetition 1.0, max_tokens 4096, streaming.

### Perf (completions endpoint, same canonical sampling; `bench_completions.py`)

| cell | spec | steady tok/s (last-50%) | tok/event (≈E[len]) | TTFT |
|---|---|---|---|---|
| v22 mtp k4 graphs (B2) | mtp k4 | **72.23** | 4.15 | 0.16 s |
| v22 mtp k4 graphs (B3) | mtp k4 | **74.31** | 4.19 | 0.13 s |
| engine-side windows | mtp k4 | 66–88 (avg ~77) | E[len] 4.17–4.44 | — |
| v21 user config (t21buser, chat) | dflash k2 | 32.93 | 1.77 | — |
| v21 best dflash cell (k2c3, chat) | dflash k2 | 39.78 | 1.92 | — |
| v21 nospec bar (chat) | none | 32.79 | 1.00 | — |

**MTP k4 with graphs ≈ 2.2x the user's live dflash k2 config and ≈ 2.2x
nospec.** The eager head costs little: the decode win comes from the graphed
target verify at E[len] ~4.2.

### Known behavior (pre-existing, NOT introduced by v22)

With the **chat** endpoint + `--reasoning-parser deepseek_r1`, the fork's MTP
verify sampling produces a much longer `<think>` phase than dflash/nospec
(v21 eager m2 replica: 14399 reasoning chars, no close within 4096 tokens;
v22 graphs: same). Text stays coherent throughout — planning content, no
corruption, no loops. The completions endpoint (no chat template) finishes
naturally in ~3200-3500 tokens ("Want me to add any of these features?").
Chat-endpoint content-rate benchmarking against dflash is therefore not
apples-to-apples for MTP; use the completions endpoint or engine-side
windows.

### t2 dflash no-regression (user's exact live config on v22)

See PERF_TUNING.md v22 outcome block for the final numbers (gate: 32.93
t21buser ±3%, acceptance unchanged, 0 resets).

## Files

- `Dockerfile` — FROM v21; COPY the 2 overlays; grep guards + py_compile +
  import checks.
- `qwen3_5_mtp.py` — pristine from v21 image + module-tail ignore block.
- `llm_base_proposer.py` — v21 content (v18-v21 lineage incl. spec_timing) +
  the NONE override in `initialize_cudagraph_keys`.
- `serve.sh` — v21 + `SPEC_METHOD=mtp|dflash`, `COMP_MODE` (FULL_DECODE_ONLY
  default; `eager` maps to `--enforce-eager`), forwards
  `VLLM_XPU_MTP_EAGER_HEAD`.
- `bench_cargame.sh` / `cargame_client.py` — v21 canonical chat bench.
- `bench_completions.py` — same canonical bench via /v1/completions
  (reasoning-parser-free; works for MTP's long-think chat responses).
- `monitor3.sh` — host GPU/reset monitor (v21).

## Usage

```bash
# MTP k4 with graphs (the fixed path)
SPEC_METHOD=mtp KV_DTYPE=turboquant_4bit_nc MAXLEN=262144 MNBT=8192 \
BLOCKSIZE=128 DTYPES=float16 EXTRA_ARGS="--quantization fp8 --reasoning-parser deepseek_r1 --tool-call-parser qwen3_xml --enable-auto-tool-choice" \
TARGET_DIR=/models/qwen3.8-27b-fp8 ./serve.sh

# stock behavior A/B (crashes at boot, for reproduction only)
VLLM_XPU_MTP_EAGER_HEAD=0 SPEC_METHOD=mtp ... ./serve.sh
```
