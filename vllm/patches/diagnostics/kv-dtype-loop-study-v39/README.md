# kv-dtype matrix + thinking-loop study v39 (2026-09-03)

Task: (a) make `--kv-cache-dtype turboquant_4bit_nc` (and the other TQ
types) match or beat `fp8_e4m3` on decode speed, prefill/TTFT and latency;
(b) fix the "AI loops when thinking" reports on `fp8_e4m3` and validate
the other fp8* KV dtypes.

Environment: `llm-scaler-prod:v1` (certified v38 tree), 2× B70, TP=2,
block 512, mtp k4 spec, async scheduling, prefix caching — identical
across every lane; only `--kv-cache-dtype` differs (arg-order override on
`serve_user_nospec.sh`, which hardcodes 4bit). Scripts here run from the
host against `localhost:8000`:

- `dt_matrix.sh` — 4-lane boot+loop+bench matrix (`dt_loop.py` @4096 +
  `dt_bench.py`; chat TTFT + prefill/decode at 2k/16k/65k)
- `dt_loop8.py` — the 4 trapper prompts re-run at mt=8192 (the
  escape-rate differential that settles the loop question)
- `dt_loop8run.sh` / `dt_bootbench.sh` — single-lane boot + loop8 /
  env-knob bench runners used for the follow-up arms

## (a) The matrix (steady numbers, all four dtypes)

| metric            | auto (fp16) | fp8_e4m3 | tq_4bit_nc | tq_k8v4 |
|-------------------|-------------|----------|------------|---------|
| chat TTFT s       | 0.130       | 0.135    | 0.166      | 0.153   |
| prefill 2k tok/s  | 1154        | 1184     | 742        | 776     |
| prefill 16k tok/s | 2322        | 2034     | **2397**   | 2191    |
| prefill 65k tok/s | 1475        | 1167     | **1685**   | 1655    |
| decode 2k tok/s   | 33.11       | **34.42**| 32.68      | 32.01   |
| decode 16k tok/s  | 26.65       | **30.75**| 25.27      | 21.82   |
| decode 65k tok/s  | 16.03       | **22.40**| 14.32      | 10.58   |

TQ 4bit WINS deep prefill (+18% @16k, +44% @65k — smaller KV traffic)
and is within 5% on shallow decode; it LOSES 2k prefill (−37%) and deep
decode (−18%/−36%).

## (b) Loop verdict: model behavior, NOT a KV-dtype defect

@4096 (8-prompt battery, all four dtypes): exactly the SAME four prompts
(P2 LIS proof, P4 domino tiling, P5 DB schema, P6 Monty-Hall-with-policy)
hit `finish=length` with zero output on every dtype — fp16 baseline
included. Zero periodic text loops detected anywhere (word- AND
char-periodicity detectors over reasoning and content tails). Trapped
tails are coherent productive reasoning, mid-derivation.

@8192 (the differential): **fp16/auto baseline: 4/4 still trapped,
0 tail-loops. fp8_e4m3: 3/4 trapped, 0 tail-loops — P6 ESCAPES at 7883
tokens and produces a full correct answer.** The unquantized baseline is
(if anything) MORE trapped than fp8. Conclusion: "thinking loops" =
budget-limited long thinking on certain problem types; identical with no
KV quantization at all. No dtype fix exists to make; the Hadamard-wrap
surface mapped for fp8 store (`flash_attn.py:1407` + FA call sites) is
therefore NOT needed for this. Mitigation is operational: raise
`max_tokens` for thinking prompts, or `enable_thinking: false` /
thinking-budget control per request.

Other fp8* types: `turboquant_k8v4` (fp8 keys) — identical trap set
@4096, clean. `fp8_e5m2` — not a servable KV dtype on this stack at all
(KNOWN_ISSUES #19: upstream hard reject with fp8 checkpoints; even the
guard bypass hits a graph-capture fault and a missing e5m2 FA kernel).

## Decode gap: why it exists and what was tried

The TQ decode stage1 (`triton_turboquant_decode.py`) is a 1-warp,
BLOCK_KV=4 serially-chained tiled kernel; per key element the MSE path
does two byte loads + int32 shift/mask + a centroid-LUT gather + a norms
multiply, per split, 32 fixed splits (cudagraph constraint), ~508 serial
iterations/split at 65k. fp8_e4m3 decode does NOT go through it — it
runs vLLM's native ESIMD flash-decode kernels. The gap is architectural
(triton-vs-ESIMD), which is why it grows with depth.

- v39a nibble-split unpack: bit-exact, REJECTED — decode −9.5/−34/−53%
  at 2k/16k/65k (`failed/tq-nibble-unpack-v39/`). The unpack loads were
  never the cost — L1 makes the second byte load free; `tl.interleave`
  lowers expensively on XPU.
- `VLLM_TQ_STAGE1_STAGES=2` (software pipelining, never swept before):
  NO effect — 32.66/25.21/14.31 vs 32.68/25.27/14.32.
  (`VLLM_TQ_STAGE1_STAGES` invalidates the triton cache key → ~6.5s
  recompile on first request; the env-knob A/B boots also show a
  depressed 2k-prefill number — treat that lane's prefill as noise.)
- `VLLM_TQ_BLOCK_KV=8` + stages=2: WORSE — 32.33/24.28/12.56
  (consistent with the header's BLOCK_KV=16/32 −72%/−83% measurements).
- Env-tunable space is exhausted; shipped defaults (4/1/1) are this
  kernel's optimum on B70. Closing the remaining deep-decode gap needs
  an ESIMD-class rewrite of TQ decode (port the MSE dequant into the
  native flash-decode path) — follow-up project.

## 2k prefill gap (−37%): attribution still open

Refuted: store-path fp32 chain (~40ms total, arithmetic), triton autotune
sweeps (there are none), first-use JIT (the chat phase preceding the 2k
measurement compiles the same kernels; chat TTFT penalty is only +31ms).
The deficit pattern (lose 2k, win 16k/65k) implies a ~3s fixed per-call
cost that amortizes — unidentified. A store/attention timing probe is the
next step if this lane is pursued; decode is the bigger deficit.
