# Qwen3.8-27B FP8 + DSpark Drafter — B70 XPU Serving Stack

Achieves **72.2 tok/s median (85.9 peak)** on the isolated C1 benchmark
(temp=0, pp=0, n=16) on 2× Intel Arc Pro B70 (TP=2), vs 32.4 tok/s FP8
no-spec and 54.67 tok/s FP8+MTP2 (dspark image, graphs=0).

End-to-end single-stream serving (adv:v9, graphs on, spec k=4): a
512-token greedy generation completes in **8 s wall** (~64 tok/s
including prefill) — **2.0x the rmacy v14/v15 recipe** (15 s), which
serves the same drafter eager. Full matrix below.

## Contents

- `Dockerfile` — retrofit build: copies the five fixed DSpark/DFlash files
  into the `qwen36-b70-vllm:b3-maxperf-final-v7` base image.
- `qwen3_dflash.py`, `registry.py` — DSpark draft model classes
  (`DSparkDraftModel`, markov + confidence heads).
- `dflash.py`, `utils.py` — **kernel readout fix** + eager-loop allocation
  fix (see below).
- `gpu_model_runner.py` — adaptive-mode list handling.
- `serve.sh` — reference serving command (image-conditional, see below).

All five python files are byte-identical to the validated
`llm-scaler-vllm-adv:v8` image contents.

## Validated configurations (2x Arc Pro B70, TP=2, drafter-fp8-v5)

| Image | graphs | gmu | len | seqs | dtype | Result |
|---|---|---|---|---|---|---|
| `llm-scaler-vllm-adv:v8` | 1 | 0.8 | 64000 | 64 | bf16 | coherent greedy; acceptance 59.5-65%, mean accepted length 3.38, ~63 tok/s single stream (512 tok / 8.1 s) |
| `llm-scaler-vllm-adv:v9` | 1 | 0.8 | 64000 | 64 | bf16, no spec | coherent greedy; 512 tok / 15 s, 1536 / 47 s (~34 tok/s) |
| `llm-scaler-vllm-adv:v9` | 1 | 0.8 | 64000 | 64 | bf16 + dflash k=4 | coherent; 512 tok / 8 s, 1536 / 23 s; mean accepted length 2.77-4.17; greedy byte-identical to no-spec |
| `llm-scaler-vllm-adv:v9` | 1 | 0.8 | 64000 | 64 | bf16 + dflash k=6 | coherent; 512 tok / 7 s, 1536 / 23 s; greedy byte-identical to k=4 |
| `llm-scaler-vllm-adv:v10` | 1 | 0.8 | 64000 | 64 | bf16 + dflash k=4 | coherent; 512 tok / 8 s, 1536 / 24 s; greedy byte-identical to v9 k4; acceptance 2.82-3.29 |
| `llm-scaler-vllm-adv:v10` | 0 | 0.8 | 64000 | 64 | bf16, no spec (compile mode, graphs OFF) | coherent — the v4-v9 silent-garbage cell, fixed by 28ff055; 512 tok / 41 s (~12 tok/s; graphs are still 2.7x faster) |
| `llm-scaler-vllm-adv:v12` | 1 | 0.8 | 64000 | 64 | bf16 + dflash k=4 | coherent; 512 tok / 8 s, 1536 / 23 s; acceptance 2.81-3.57; greedy byte-identical to v9 k4 AND v10 k4 (cross-image) |
| `llm-scaler-vllm-adv:v12` | 1 | 0.8 | 64000 | 64 | bf16, no spec | coherent; 512 tok / 18 s, 1536 / 48 s |
| `llm-scaler-vllm-adv:v12` | 0 | 0.8 | 64000 | 64 | bf16, no spec (compile mode, graphs OFF) | coherent — gate cell re-validated; 512 tok / 36 s, 1536 / 106 s (graphs still ~2x faster) |
| `qwen38-fp8-dspark:v8` | 0 | 0.90 | 8192 | 1 | bf16 | coherent greedy (rmacy serve.sh @6e63e9e verbatim) |

`serve.sh` picks the row matching `IMAGE` automatically.

The `llm-scaler` image additionally supports XPU graphs (PIECEWISE) with
this stack: the dflash eager drafting loop reuses a grow-only bias scratch
buffer (`dflash.py`) so it no longer churns the XPU caching allocator and
wedges the xe engines during piecewise capture — the historical reason
`VLLM_XPU_ENABLE_XPU_GRAPH=0` was required no longer applies to it.
Note the opposite constraint applies to TARGET-only bf16 serving on
adv:v4-v9: compile mode + `VLLM_XPU_ENABLE_XPU_GRAPH=0` silently corrupts
TP output there in BOTH dtypes (KNOWN_ISSUES #04, fixed by 28ff055 /
adv:v10) — keep graphs on, or use adv:v10/v12+. adv:v12 re-validates the
full battery (v10 semantics restored after the reverted v11 experiment):
graphs+spec k=4 is 8 s/23 s with greedy output byte-identical to BOTH v9
and v10 (acceptance 2.81-3.57), the compile+graphs-off gate arm is
coherent, and the TQ champion is unregressed (15 s @512, grid=256). Do
NOT use adv:v11: its out-of-place AR op clone corrupted PIECEWISE
graphs-on serving (word salad, 0% spec acceptance) — the alias return is
load-bearing under PIECEWISE; see KNOWN_ISSUES #04 for the experiment
record.

## Measured speed matrix (v9 vs rmacy v14, "Write a html car game." prompt, greedy, ignore_eos, wall clock)

| serving stack | 512 tok | 1536 tok | tok/s @512 |
|---|---|---|---|
| v9 graphs, bf16, no spec | 15 s | 47 s | 34 |
| **v9 graphs, bf16, dflash k=4** | **8 s** | **23 s** | **64** |
| v9 graphs, bf16, dflash k=6 | 7 s | 23 s | 73 |
| rmacy v14 eager, dspark spec | 15 s | 45 s | 34 |
| rmacy v14 eager, no spec | 40 s | 114 s | 13 |

XPU graphs + dflash spec is **2.0x** the rmacy always-spec recipe and
**~2.6x** rmacy target-only. The two stacks carry the same vllm build
(0.21.1.dev0+gad7125a43); the delta is graphs (rmacy serves eager) —
their recipe always runs spec decode (target M=k+1), which both masks
the KNOWN_ISSUES #04 corruption cell and never executes the M==1 decode
paths the ESIMD fusions target.

### k-sweep (v9, graphs, single stream)

| k | 512 tok | 1536 tok | mean accepted length |
|---|---|---|---|
| 4 | 8 s | 23 s | 2.77-4.17 |
| 5 | 10 s | 32 s | 2.85-3.31 |
| 6 | 7 s | 23 s | greedy identical to k=4 |

k=4 and k=6 tie within run-to-run noise (byte-identical greedy output);
k=5 is dominated — drafter-fp8-v5 acceptance does not grow past 4 draft
positions, so the extra verify cost is pure overhead. Recommend k=4
(default; matches the C1 benchmark optimum).

## `--kv-cache-dtype` support matrix (adv:v14, b19d92f)

Validated with the dflash k=4 recipe (2× B70, TP=2, graphs on,
44-64k len, "Write a html car game." greedy gens; 2026-08-26 battery):

| kv-cache-dtype | graphs | result |
|---|---|---|
| `none` (bf16) | 1 | **validated** — 512 tok/8 s, 1536/23 s, acceptance 0.69, peak 68 tok/s @1-5k depth |
| `fp8` | 1 | **validated** — coherent greedy; ~2× KV capacity; peak 36 tok/s under build load (idle-host numbers match baseline within noise) |
| `float16` | any | **rejected** — `reshape_and_cache_flash` raises `Unsupported data type of kv cache: float16` at first capture (v13 + v14); no memory win vs bf16 — use the default |
| `turboquant_3bit_nc` | 1 | **init + serve OK** (stride fix 3e8af63/b19d92f: the padded-page `view(-1)` store crash is gone). Coherent 512/1536; peak 55 tok/s @1-5k, mean 40 to 25k depth, acceptance 0.51 |
| `turboquant_k8v4` | 0 (`--enforce-eager`) | **functional eager-only** — healthy in ~3 min, 10k gen completed COHERENT head+tail at 7.3 tok/s; draft acceptance ~1-2% (bf16 drafter vs TQ target KV divergence) so no spec speedup |
| `turboquant_k8v4` | 1 | **hangs** post-capture in sampler warmup (worker spin at `make_dummy` H2D, xe GuC engine resets) — KNOWN_ISSUES #05(b) |
| `turboquant_k3v4_nc` | 0 (`--enforce-eager`) | **functional eager-only** — healthy in ~2.5 min, 10k gen completed COHERENT at 6.9 tok/s |
| `turboquant_k3v4_nc` | 1 | **hangs** same as k8v4 (graphs-mode warmup wedge) |
| `turboquant_4bit_nc` | 1 | **hangs** during capture itself at 63% (12/19 sizes) — pre-existing grow-branch issue, unaffected by the stride fix |
| `turboquant_4bit_nc` | 0 (`--enforce-eager`) | **not usable** — serves and starts coherent, but the 10k gen died at ~4.3k depth |

The `turboquant_*` presets route KV traffic through the triton unified
attention + TurboQuant store kernels, which are now stride-safe on the
padded per-layer pages this hybrid (linear_attn + self_attn) model
unifies to 4 MiB (b19d92f: `torch.as_strided(kv_cache, (numel,), (1,))`
flat pointer + explicit `stride_cache_block/pos/head`). The remaining
graphs-mode hang is the capture-context warmup interaction (KNOWN_ISSUES
#05(b)), not addressing. The drafter always keeps `auto` (bf16) KV —
dflash.py:418 falls back automatically and that is benign.

**Long single streams:** no graphs-mode configuration completed an
`ignore_eos` 40k single-stream gen on this driver stack — dflash arms
(bf16/fp8/TQ alike) die with `UR_RESULT_ERROR_DEVICE_LOST` at the
acceptance-event sync at 20-32k, and a target-only (no-spec) control arm
wedges at just 2.9k (`RPC call to sample_tokens timed out`). Only the
eager k8v4/k3v4_nc arms completed their 10k gens. Short gens (≤1536) are
rock-solid in every healthy mode. See KNOWN_ISSUES #05(a).
Depth-vs-rate: peak 68 (bf16) / 55 (3bit) / ~36+ (fp8) tok/s in
the 1-5k band, decaying to ~34/28 by 30k as attention cost grows — the
"30-45 tok/s" you see in practice IS the expected deep-generation band;
60-73 only holds for the first few thousand tokens.

## The kernel readout fix (why this matters)

SpecForge DSpark trains output position j to predict token anchor+j+1
(LM-style). The stock vLLM dflash kernel sampled draft outputs at query
offsets 1..k (BERT-style), so every draft was off by one position and
acceptance collapsed to ~24% for any SpecForge-trained drafter.

Fix (3 lines):
- `dflash.py`: `num_query_per_req = k` (was `k+1`)
- `utils.py`: `is_sample = is_query` (was `is_query & (query_off > 0)`)
- `utils.py`: `sample_out_idx = req*k + off` (was `req*k + off - 1`)

Effect on the released RadixArk drafter: 24% → 66% pos-0 acceptance.
With our fine-tuned drafter-fp8-v5: per-position acceptance 0.838 / 0.676 /
0.541 / 0.324 at k=4 (draft-position conditional rates), overall acceptance
59.5-65%, mean acceptance length 3.38.

## ESIMD page-attention gate (bf16 / eager correctness)

The XPU paged-attention decode shortcut
(`custom_esimd_kernels_vllm.eagle_ops.page_attn_decode`, inserted in
`vllm/v1/attention/backends/flash_attn.py`) is only correct for fp16
queries under XPU graphs on this build:

- bf16 query: `TORCH_CHECK(query.scalar_type() == torch::kHalf)` at
  `custom-esimd-kernels-vllm/csrc/eagle/eagle.sycl:444` — bf16 target-only
  (no drafter) serving crashes on the first decode step.
- fp16 query: coherent in every mode re-tested on v9 (true eager, graphs
  on/off). An earlier "eager fp16 garbage" observation on v8 is
  superseded — all reproducible silent garbage on the v4-v9 lineage
  turned out to be an unrelated TP all_reduce defect triggered by
  compile mode + graphs off in BOTH dtypes (KNOWN_ISSUES #04; the
  instrumented 4-arm run showed fp16 word-salad too, so the earlier
  "fp16+compile coherent" exoneration was a classify() false positive
  on multilingual garbage), and it corrupts serving regardless of the
  page-attn insert.
- Speculative decoding hides the crash: the verify pass runs with
  `max_query_len = k+1 > 1`, which bypasses this code path entirely.

Fixed in `vllm/patches/vllm_for_multi_arc.patch` (`PAGED_ATTN_ESIMD_INSERTED_v1`
gate): the insert now additionally requires `query.dtype == torch.float16`
and `VLLM_XPU_ENABLE_XPU_GRAPH` enabled; everything else routes to
`flash_attn_varlen_func`. `DISABLE_ESIMD_PAGE_ATTN=1` remains a manual
opt-out.

Related, same lineage: with the page-attn insert gated off, TRUE-eager
bf16 (`--enforce-eager`) instead trips a different, correctly-guarded
fp16-only fusion: `esimd_norm_gemv_fp8_blockscale: norm inputs must be
fp16` (M==1 GDN out-proj norm+GEMV). That one is loud, not silent —
work around it with `-e DISABLE_ESIMD_GDN_OUTPROJ=1`, or just serve
with graphs on. The silent compile-mode bf16 corruption itself is
KNOWN_ISSUES #04, fixed by 28ff055 (adv:v10).

## Quality

Greedy spec decode is lossless by construction. Verified earlier: 4/5
byte-identical outputs vs target-only baseline; the one divergence is a
dtype tie-break (fp16 vs bf16), not a drafter effect. Cross-checked
against an independent bf16 reference endpoint: drafter adds zero
divergence.

## Requirements

- 2× Intel Arc Pro B70 (Battlemage G31, 8086:e223) or equivalent XPU
- intel/level-zero driver stack matching the image
- Qwen/Qwen3.8-27B-FP8 checkpoint (block FP8 e4m3, [128,128])
- `/models/drafter-fp8-v5` (1.36B DSpark drafter, HF format)

## Known limitations

- On builds without the ESIMD gate fix (this repo prior to the fix commit,
  and stock dspark images): bf16 target-only serving crashes at the ESIMD
  kernel assert; use the fixed image, spec decoding, or
  `DISABLE_ESIMD_PAGE_ATTN=1`.
- On adv images v4-v9 (07827c0..c10c7b2): target-only serving with
  compile mode and `VLLM_XPU_ENABLE_XPU_GRAPH=0` silently returns
  garbage in BOTH dtypes — KNOWN_ISSUES #04. Keep graphs on, use
  `--enforce-eager` with `-e DISABLE_ESIMD_GDN_OUTPROJ=1`, or use
  adv:v10/v12+ (28ff055). Do NOT use `--dtype float16` as a workaround
  — fp16 corrupts the same way, just with a different garbage flavor.
  Also avoid adv:v11 (reverted experiment): its AR op clone corrupted
  PIECEWISE graphs-on serving itself.
- Adaptive block truncation (DSPARK_ADAPTIVE_BLOCK=1) runs but is slower
  than fixed k=4 on single-request workloads.
