# llm-scaler v1.2.5 — Deep Audit, Live Baselines & Fix/Improvement Plan

Date: 2026-09-07 · Host: 10.20.3.65 (2× Intel Arc Pro B70, TP=2, oneCCL, xe driver)
Image: `llm-scaler-exp:v1.2.5` (certified bits = v1.2.5t5, sha256:a522bf15…, image-ID verified on host)
Model: qwen3.8-27b-fp8 (hybrid GDN: 48 linear-attn + 16 full-attn layers, GQA 24q/4kv × hd256)

All numbers below are **measured today on the live host** unless marked *(from ledger)*.
Harness: `serve_bench.sh / run_lane.sh / master125.sh / post125.sh` + `dt_probe2 / dt_ctxscan / dt_ctxscan2 / dt_conc` (pushed to `/root/build/`, archived in `.tmp-tq/`). Long-context prompts use VARIED 16-word fillers (KNOWN_ISSUES #13: repetitive fillers trigger instant-EOS ≥32k).

---

## 0. Executive summary

1. **v1.2.5 is healthy and certified-reproducible**: all 5 lanes booted on the *user's exact* flags (GMU 0.9, maxlen 262144 — no fallbacks), battery SHAs match certified references bit-exact, 0 wedges / 0 strikes / 0 engine deaths across ~3.5 h of sustained 2k→227k single- and multi-client load. The v52 crash-4 fixes hold.
2. **Speculative decoding is NET-NEGATIVE at deep context.** At 128k/227k the nospec+tq4 lane beats every spec lane (17.7/9.6 tok/s vs mtp4 11.7/6.1, dflash7 4.2–7.3). Cause is measured, not speculative: per-position acceptance collapses with depth (0.73/0.71/0.14/0.02 @65k) while each step pays (k+1) full-KV scans; MTP draft KV is additionally fp16 (2× bytes, defect D4).
3. **fp8_e4m3 + spec collapses 1.7-2.6× at GMU ≥ 0.9** (16-17 vs 35-42 tok/s @16k). Convicted by a 7-boot A/B: the cause is the GMU×spec×fp8 interaction (maxlen/parser/order/env all ruled out). Mitigation validated: GMU 0.85 for fp8+spec lanes. Root mechanism = open profiling item P-1.
4. **fp8 + nospec (GMU 0.9, user flags) is the surprise champion** for long-context and concurrency: 25.6 tok/s @128k and **21.2 @227k** (vs 17.7/9.6 for tq4+nospec, +45%/+121%), conc8 200.6 agg — while keeping bit-exact certified accuracy posture. The only loss: 2k-class interactive (23.5 vs mtp4+tq4 61.8).
5. The other AI's two code-level findings are **confirmed at code level and quantified**: ragged fp8 verify misses the fast kernel (`flash_attn_v51.py:1343-1349` uniform-q gate), and TQ continuation prefill re-dequants the whole cached prefix per chunk (`turboquant_attn_v52.py:1486-1652`). Measured verdicts differ from intuition: the TQ O(n²) is *second-order* (TQ prefill is still 1.5× faster than fp8's), while the deep-ctx **decode** kernel is the dominant lever — and the fp8-vs-TQ decode gap at 227k (21.2 vs 9.6 nospec) proves the TQ decode path leaves ≥2× on the table.
6. Deep decode numbers are steady-state, not JIT noise: LEG1 warm re-run reproduced cold decode exactly (11.7/6.1 tok/s) and prefix-cache re-runs give 3.3-4.8 s TTFT for 136k/243k prompts.

---

## 1. What v1.2.5 actually runs (verified)

Composition (md5-verified in-image on host, matching `vllm/patches/diagnostics/dflash2-fullfix-v51/`):

| Layer | Content |
|---|---|
| Scheduler | v52l geometric boundary-clamp (2048-token mamba blocks) |
| AsyncScheduler | v52m 3-strike zero-emission zombie abort (`VLLM_V52M_STRIKES`) |
| GMR | v52n/o draft-input sentinel clamp + nospec getattr guard |
| Attention (full-attn) | `flash_attn_v51.py` — v51 fp8 MQ Triton two-stage kernel (SPLITS 32, ceiling 64), v33 MQ3D fallback, C++ FA2 branch1 |
| Attention (TQ) | `turboquant_attn_v52.py` — graph-fixed KV splits grid=256 w/ ctx tier; MQ splits fixed 32 (ladder-follow refuted 2026-09-06) |
| Serving | F9 fire-and-forget abort + aclose companions; F8v2 placeholder park timer |
| Warmup | `dt_warmup_v51.py` p1-p9 (greedy bs1+2; chunk buckets 18k/36k/70k) |

KV pools measured at boot (GMU 0.9 / maxlen 262144):
- tq4 + mtp4: **1,298,151 tokens** (12.3 GiB)
- fp8 + mtp4: **707,980 tokens** (hbs=1024)
- fp8 + dflash7: **373,507 tokens** (hbs=1024)

---

## 2. Live baseline matrix (2026-09-07, all on user's exact serve flags)

### 2.1 Single-client decode throughput vs context (256 tok out, temp 0, VARIED fillers)

| Lane | 2k | 16k | 32k | 65k | Phase-V ref (same image) |
|---|---|---|---|---|---|
| **mtp4 + tq4nc** | **61.8** | **46.6** | 29.3 | 21.0 | 61.8/40.4/28.7/25.1 |
| **nospec + fp8** | 23.5 | **33.5** | **32.1** | **29.6** | (nospec fp8 v51: 34.4/30.7/22.4 @0.85) |
| **nospec + tq4nc** | 33.7 | 30.6 | 27.7 | 23.2 | 33.7/30.7/28.0/23.5 |
| df7 + tq4nc | 46.9 | 32.8 | 18.3 | 14.3 | 47.0/35.3/20.0/14.3 |
| mtp4 + fp8 | 31.6 | 16.1 | 14.1 | 9.6 | 31.7/**41.9**/**27.5**/**22.3** |
| df7 + fp8 | 23.2 | 14.5 | 14.8 | 7.0 | — |

*(nospec+fp8 row measured post-matrix at GMU 0.9, user flags — see §3.1.1; mtp4+fp8 @0.85 recovers to 37.2/23.4/22.2, C-warmed 42.1/34.0/22.3.)*

- tq4 lanes reproduce Phase-V bit-consistently (±noise; df7 65k exactly 14.3). The other AI's "nonspec TQ 23.5 vs DFlash 14.3 @67k" is reproduced: 23.2 vs 14.3.
- "DFlash 110.6 tok/s" is the *short-arith battery* (p2); at 2k ctx the same lane does 46.9 — confirmed not representative of real contexts.
- **fp8+spec lanes are broken at depth under user flags** (2.0-2.6× below same-image Phase-V refs) — D1.

### 2.2 Deep context 128k / 227k (nwords 115000/205000 → ptok 136191/242761)

| Lane | 128k decode | 227k decode | 128k prefill t/s | 227k prefill t/s |
|---|---|---|---|---|
| **nospec + fp8** | **25.6** | **21.2** | 1430 | 1733 |
| **nospec + tq4nc** | 17.7 | 9.6 | 1780 | 2232 |
| mtp4 + tq4nc | 11.7 | 6.1 | 1491 | 2055 |
| df7 + tq4nc | **4.2** | 7.3 | 1537 | 2280 |
| df7 + fp8 | 9.7 | 4.0 | 1084 | 1396 |
| mtp4 + fp8 | 6.7→12.5 warm | 4.4→5.1 warm | 1095 | 1374 |

Warm-re-run (post125 LEG1/LEG2, fresh container, cold vs warm pass): mtp4 11.7/6.1 cold == 11.7/6.2 warm; df7 4.0/7.3 cold == 4.2/7.3 warm → **all deep numbers are steady-state, none JIT-contaminated**. Warm TTFTs: mtp4 3.3/4.8 s, df7 1.9/2.3 s (prefix cache; effective 41-105k tok/s).

- **Spec ≤ nospec at every depth ≥65k** for both drafters and both dtypes.
- LEG1 cold/warm (fresh container, mtp4+tq4): decode 11.7/6.1 cold == 11.7/6.2 warm; warm TTFT 3.3 s @128k / 4.8 s @227k (prefix cache absorbs re-prefill; effective 41-50k tok/s).
- Spec acceptance at depth (mtp4, 65k): per-position 0.730/0.714/0.143/0.016 → mean accepted ≈1.4 tok/step while paying 5 full-KV scans (target+drafts, draft KV fp16). dflash7 keeps 0.808/0.692…@65k but pays 8 scans of a *separate* fp16 draft window plus the slowest verify path (D2-adjacent for tq4).
- fp8 prefill is uniformly ~1.5× slower than TQ prefill at depth (1084-1396 vs 1491-2280) — the TQ O(n²) dequant (D3) does **not** make TQ the slow one; fp8's prefill path is the worse one.

### 2.3 Multi-client

| Test | nospec fp8 | nospec tq4 | mtp4 tq4 | df7 tq4 | mtp4 fp8 | df7 fp8 |
|---|---|---|---|---|---|---|
| conc8 @2k agg / per-client | **200.6** / ~31 | 160.5 / ~26.5 | 117.5 / 17-38 | 93.4 / ~20 | 61.4 / ~13 | 90.1 / ~14.8 |
| conc4 mixed 2k+16k agg | **71.1** | 65.7 | 61.4 | 35.5 | 29.5 | 23.0 |
| conc2 @65k per-client | 4.8 / 29.0 | 17.6 / 5.6 | 10.0 / 3.5 | 3.8 / 13.4 | 7.4 / 3.2 | 10.1 / 3.0 |

- nospec scales best under concurrency; **fp8+nospec is the aggregate champion (200.6 @8×2k, fair 31/client)**. Spec lanes lose 27-45% of their single-client advantage immediately (KNOWN_ISSUES #16/#17: eager MTP draft host chain, 52-71% host samples).
- conc2@65k shows a **decode-starvation asymmetry** in all lanes (one client at 2-4× the other while TTFT p50 == p95 == both prefilled together) — D8.

### 2.4 Correctness & health

- Battery SHAs: mtp4_tq4 == certified refs (p1 3cecc747, p2 c87e27c4, p5 f61457ef==v51 ref for fp8 lane); nospec refs bit-exact (c87e27c4 / 1f9c461b).
- Health counters across all lanes: STALLS 0 (tq4 lanes) / 17-20 (df7 JIT class), STRIKES 0, engine ERRS 0, F8 1 benign firing (nospec lane, engine survived), v52f PRE-COPY warnings at ≥65k only (mamba state copy, informational).
- Acceptance: mtp4 0.57-0.61 mean len 2.6-3.0 short ctx; healthy at depth for fp8 too (3.78 @65k) → D1 is *speed*, not draft quality.

---

## 3. Confirmed defects & root causes

### D1 — fp8_e4m3 + spec collapses at GMU ≥ 0.9  *(convicted; see §3.1)*
- Symptom: mtp4_fp8 @0.9 = 16.1/14.1/9.6 tok/s @16k/32k/65k vs **35.5-42.1/23.4-34.0/22.2-22.3** @0.85 (same image/env/harness). df7_fp8 same class.
- Cause axis convicted by 7-boot A/B: **GMU ≥ 0.9 × spec × fp8 three-way interaction** (maxlen/parser/order/JIT ruled out; tq4+spec and fp8+nospec both 0.9-safe; deterministic).
- Penalty is per-step and vanishes at ≥128k (both GMUs converge 12.4-12.5/5.1 warm); acceptance unchanged → step-time, not draft quality.
- **Mitigation (validated)**: `--gpu-memory-utilization 0.85` for any fp8+spec lane. Root mechanism open → P-1 profiling item.

### D2 — ragged fp8 verify misses the fast MQ kernel
- `flash_attn_v51.py:1343-1349`: route to the v51 two-stage Triton kernel requires `num_actual_tokens == max_seqlen_q * batch` (**uniform q**) plus `1<q≤8`, causal, paged, hs≤256. Any ragged batch (mixed finished/running, chunked tail, spec verify with mixed seq lens) falls to v33 MQ3D (`:1388`) or C++ FA2 `is_mix_batch=False` (`:1409/:1433`) — no SPLITS, pathological at depth.
- Confirms the other AI's finding. Most damaging under concurrency at depth (conc tests above show fp8 spec worst-in-class).

### D3 — TQ continuation prefill re-dequants the full cached prefix per chunk (O(n²))
- `turboquant_attn_v52.py:1486` `_continuation_prefill`: per chunk, `alloc_len = ceil(cached_len/512)*512` (`:1516`) → `_tq_full_dequant_kv[grid=(alloc_len,Hk)]` (`:1552`) → Pi-rotation matmul → concat → FA varlen over full seq. At 227k with MNBT 8192: ~13.7× dequant amplification (~3.1M token-dequants) + ~1.9 GiB transient fp16 workspace per chunk.
- **Measured verdict: second-order.** TQ prefill at depth is still 1.5× *faster* than fp8's (2055-2280 vs 1374-1396 tok/s @227k); the dominant deep-ctx costs are decode-side. Worth fixing for 262k multi-chunk and VRAM-spike reasons, not for headline tps.

### D4 — MTP draft KV is always fp16 (no draft-KV dtype policy)
- `llm_base_proposer.py` has zero `kv_cache_dtype` references (grep-verified in image); dflash has `VLLM_DFLASH_DRAFT_KV_DTYPE`. At depth the eager MTP draft loop scans fp16 draft KV = 2× bytes of a tq4nc target KV. Combined with D5, this is why mtp4 loses to nospec at ≥65k.

### D5 — fixed-k speculation is net-negative at deep context
- Measured acceptance curve (0.73/0.71/0.14/0.02 @65k, mtp4) → k=4 pays 5 scans for ~1.4 accepted tokens at ≥128k. Nospec wins 128k by 1.5× and 227k by 1.6×. No ctx-adaptive k exists at runtime (EMIT_K knob exists but default-inert, range 1..num_spec_tokens; EMIT_K=4 was v42-refused, EMIT_K=5 stalled).

### D6 — spec concurrency collapse (#16/#17, known)
- eager MTP draft loop = 52-71% host samples; acceptance-gated scheduling exposes ~54 ms/step host chain. Real fixes are P2 (worker-resident draft loop) / P3 (graph capture) — not yet done. Interim: route high-concurrency traffic to nospec (measured: 160 vs 117 agg @8×2k).

### D7 — first-request JIT at deep context (warmup stops at 70k)
- `dt_warmup_v51.py` p7/p8 buckets reach 70k only. First 128k/227k request compiles new shapes: df7_tq4 cold 128k decode 4.2 vs 7.3 warm (+STALLS 17-20); mtp4_fp8 JIT'd `_fp8_mq_stage1/2` during ctxscan. Steady-state unaffected (LEG1: warm == cold for mtp4), but first-token latency after idle-at-depth is 60-115 s.

### D8 — conc2@65k decode starvation asymmetry
- Both clients prefill together (p50==p95 TTFT) then one decodes at 2-4× the other in *every* lane (e.g. nospec 17.6 vs 5.6). Consistent with step-level interleave favoring the first-scheduled request's chunk boundaries (2048-token mamba blocks) rather than fair round-robin. Not a correctness bug; hurts p99 tail.

### D9 — minor / informational
- F8 guard fired once in nospec lane (placeholder overflow park; engine survived, 0 asserts).
- v52f PRE-COPY warnings at ≥65k (mamba state copy across 2048-blocks) — informational.
- `df7 2k ttft 1.5s` vs mtp4 1.1s: dflash drafter boot cost.

### D10 — df7+tq4 depth curve is NON-MONOTONIC: 128k ≈ half of 227k
- Measured steady-state (cold==warm, two containers): 32k 18.3 → 65k 14.3 → **128k 4.0-4.2** → **227k 7.3**. Deeper context is *faster* from 128k→227k, i.e. something specific to the ~136k-token point (ptok 136191) breaks dflash7 decode. Suspects: KV-splits tier threshold (`_effective_kv_splits` ctx ladder, `turboquant_attn_v52.py:574`), dflash sliding-window draft interacting with 2048-token mamba block boundaries, or a graph-capture bucket gap at that seq-len class. Not present in mtp4 or nospec (both monotonic).

---

## 3.1 fp8 A/B verdict — **CONVICTED: GMU ≥ 0.9 × spec × fp8 interaction** (7 controlled boots, all same image/env/harness)

| Boot | GMU | maxlen | spec | 16k | 32k | 65k | 128k deep | 227k deep |
|---|---|---|---|---|---|---|---|---|
| A-lane (warm) | 0.9 | 262144 | mtp4 | 16.1 | 14.1 | 9.6 | 6.7 cold / **12.5 warm** | 4.4 / **5.1** |
| A2 (fresh, no warmup) | 0.9 | 262144 | mtp4 | 17.2 | 16.3 | 12.7 | — | — |
| bisect | 0.9 | 245760 | mtp4 | 17.1 | 14.0 | 12.6 | — | — |
| B (Phase-V boot, no warmup) | 0.85 | 245760 | mtp4 | 35.5 | 25.2 | 22.2 | 12.4 cold==warm | 5.1 |
| C (0.85 bisect, warmed) | 0.85 | 245760 | mtp4 | **42.1** | **34.0** | **22.3** | 12.4 / 12.4 | 5.1 / 5.1 |
| fixed-GMU | 0.85 | **262144** | mtp4 | 37.2 | 23.4 | 22.2 | — | — |
| **control** | 0.9 | 262144 | **nospec** | **33.5** | **32.1** | **29.6** | **25.6** | **21.2** |

**Rulings:**
1. **GMU is the axis**: 0.9 slow (both maxlens), 0.85 fast (both maxlens) → maxlen exonerated; parser/chat-kwargs exonerated (C uses user parser); run-order/warmup-state exonerated (A2 fresh-boot ctxscan-first is still slow); env byte-identical (diffed `serve_boot_var.sh` vs `serve_bench.sh`); deterministic across ≥3 boots per cell.
2. **Spec is required for the penalty**: fp8+nospec@0.9 is the *fastest* lane measured (see §2). tq4+spec@0.9 is Phase-V-identical (61.8@2k). Only fp8×spec×0.9 collapses.
3. **Depth-dependent magnitude**: 2k 1.7-3.8× → 65k ~1.8× → **≥128k fully converges (12.4-12.5/5.1 both GMUs)**. So the penalty is a per-step cost that matters when steps are fast (shallow), i.e. a fixed-ish overhead per decode step, not a bandwidth-rate change.
4. Acceptance healthy at 0.9 (mean 3.78 @65k, same as 0.85) → steps are slow, not drafts rejected.
5. **Deep-128k/227k lane numbers (6.7/4.4) were first-request JIT** — warm A == B == C (12.4-12.5/5.1). Phase-V "fp8 collapse" comparisons were ctxscan-class; deep was never collapsed.

**Mechanism (open, fix-plan item P-1)**: pool at 0.9 = 705-708k tokens (12.3 GiB) vs ~600k at 0.85; graphs captured 37/37 at 0.9 (clean); `VLLM_FP8MQ_SPLITS` env-fixed 32, kernel scratch grow-only cached, no pool-size dependence in `triton_fp8_mq.py`; hbs=1024 both. Suspects: spec-verify path taking a different route at larger pool (instrument the `_V51_FP8_MQ` gate hit-rate), draft-KV pool sizing squeeze at 0.9, or allocator-layout effect on the two-stage kernel. Needs a profiling session (step-time breakdown at 0.85 vs 0.9, 16k ctx, bs1 spec).

**Immediate mitigation (validated today)**: fp8_e4m3 + spec → `--gpu-memory-utilization 0.85` (maxlen 262144 fine). fp8 + nospec and all turboquant lanes are 0.9-safe.

### 3.1.1 BONUS FINDING — fp8 + nospec is the deep/concurrency champion (GMU 0.9, user flags)

| Metric | fp8+nospec | tq4+nospec | vs |
|---|---|---|---|
| 16k / 32k / 65k | **33.5 / 32.1 / 29.6** | 30.6 / 27.7 / 23.2 | +9/16/+28% |
| 128k / 227k decode | **25.6 / 21.2** | 17.7 / 9.6 | **+45% / +121%** |
| 227k prefill | 1733 | 2232 | −22% (only loss) |
| conc8@2k agg | **200.6** (31/client, fair) | 160.5 | +25% |
| conc4 mixed agg | **71.1** | 65.7 | +8% |
| conc2@65k | 4.8 / 29.0 | 17.6 / 5.6 | same D8 asymmetry, worse ratio |

fp8's in-kernel dequant decode (SPLITS two-stage) outperforms the TQ dequant-to-gmem path by >2× at 227k — direct evidence that the TQ decode kernel rewrite (L1) has ≥2× headroom at depth. Caveats: fp8 KV scale = 1.0 default (v51 certified accuracy posture), pool 705k tokens ⇒ max ~2.7 concurrent max-len requests (vs 4.9 for tq4), first-deep-request JIT (L7 covers).

---

## 4. Improvement plan (ranked)

| # | Change | Where | Expected effect | Effort | Pass/fail test |
|---|---|---|---|---|---|
| **L0** | **Ops routing (zero code, validated today)**: ≥16k / deep / high-concurrency traffic → **nospec+fp8** (GMU 0.9); ≤8k interactive → mtp4+tq4nc; fp8+spec (if used) → **GMU 0.85** | serve flags | Deep 128k/227k: 11.7/6.1 → **25.6/21.2** (2.2×); conc8: 117→200 | None | numbers above, reproduced |
| **P-1** | **Root-cause the GMU×spec×fp8 penalty (D1)**: step-time breakdown 0.85-vs-0.9 (16k, bs1, spec): instrument `_V51_FP8_MQ` gate hit-rate, draft-step vs verify-step times, kernel durations | `flash_attn_v51.py`, `triton_fp8_mq.py`, proposer | Explains 1.7-2.6×; unlocks fix so user flags work unmodified | Med | identified dominant delta ≥50% of the gap; then 0.9 == 0.85 ±5% after fix |
| **L1** | **Persistent/fused TQ decode kernel** (in-kernel dequant, no gmem roundtrip; v28dbg: 26 GB/s effective vs 2 orders below roofline; fp8-nospec @227k proves ≥2.2× available) | `turboquant_attn_v52.py` decode path | ≥2× TQ decode at 131-262k; tq4-nospec 9.6 → ≥19 @227k-class | High | ctxscan 131k/262k ≥2×; parity with fp8-nospec 21.2 @227k |
| **L2** | **Deep-ctx spec policy: route ≥65k decode to nospec / ctx-adaptive k** (EMIT_K knob exists, default-inert, range 1..num_spec_tokens) | scheduler/spec config | Kills the measured net-negative spec regime (acceptance 0.14/0.02 at pos 3/4 @65k) | Low (config) / Med (adaptive) | spec deep ≥ nospec ×0.95 at every depth |
| **L3** | **MTP draft-KV dtype policy (C3/D4)**: wire `kv_cache_dtype` into `llm_base_proposer` draft pool (mirror `VLLM_DFLASH_DRAFT_KV_DTYPE`) | `llm_base_proposer.py` | Halve draft scan bytes at depth; mtp4 deep gap narrows | Med | mtp4 deep ≥0.8× nospec; battery SHAs bit-exact |
| **L4** | **fp8 ragged-verify gate fix (D2)**: pad/mask ragged batches into uniform shape for `_v51_fp8_mq_fn`, or extend kernel to varlen | `flash_attn_v51.py:1325-1435` | Removes v33/C++-branch1 cliff under concurrency at depth | Med | gate hit-rate ≥99% on ragged conc8; fp8+spec conc8 ≥100 |
| **L6** | **TQ continuation-prefill chunked dequant (D3)**: dequant only [prev_end, chunk_end), reuse rotated workspace across chunks | `turboquant_attn_v52.py:1486-1652` | Removes O(n²) at 262k (~32 chunks) + 1.9 GiB transient; prefill +10-20% ≥128k | Med | 227k prefill ≥2500 t/s; prefill VRAM spike <200 MiB |
| **L7** | **Warmup buckets p10/p11 @131k/227k** (extend dt_warmup_v51 p7/p8) | warmup script | First-deep TTFT 60-133 s → <10 s; df7/fp8 first-pass STALLS → 0 | Low | first-deep TTFT <10 s on cold container |
| **L8** | **Spec concurrency (D6)**: P2 worker-resident draft loop / P3 graph capture | proposer/engine | conc8 mtp4 117→160+ | High | conc8@2k agg ≥150 with mtp4 |
| L9 | conc2@65k fairness (D8): round-robin chunk interleave at 2048-block boundaries | scheduler | p99 tail at deep conc | Med | per-client tps ratio ≤1.5 @conc2@65k |
| L10 | **df7 128k anomaly (D10)**: bisect ptok 115k-150k band on df7+tq4; inspect `_effective_kv_splits` tier boundary + dflash draft window vs 2048-blocks | attn/tier config | Removes df7 128k cliff (4.2→~7 class) | Low-Med | df7 deep monotonic within 15% |
| L11 | (parked) prefix-cache 4096-token granularity (#10); MNBT 16384 rejected; KV splits already optimal (MQ 32, graph 256+tier; `VLLM_FP8MQ_SPLITS=64` optional fp8 long-ctx) | — | — | — | — |

**Ordering rationale:** L0 is free and validated; P-1 unblocks the user's unmodified flags; L1 has the largest kernel ceiling (measured ≥2.2× via the fp8 lane); L2/L3 make spec viable at depth; L4 fixes concurrency cliffs; L6/L7 are latency hygiene; L10 is a small config-class fix.

---

## 5. Recommended configuration per workload (measured, today's image)

| Workload | Config | Why (measured) |
|---|---|---|
| Interactive single-client ≤8k | **mtp4 + turboquant_4bit_nc**, user flags (0.9) | 61.8 @2k — best measured anywhere; battery bit-exact |
| Single-client 16k-65k | **nospec + fp8_e4m3** (0.9) | 33.5/32.1/29.6 — beats every lane incl. mtp4-tq4 (29.3@32k, 21.0@65k) |
| Deep 128k-262k, decode-heavy | **nospec + fp8_e4m3** (0.9) | 25.6 @128k / **21.2 @227k** — 2.2× the tq4-nospec lane; caveat: pool 705k tok ⇒ ≤2.7 concurrent max-len reqs |
| Deep 128k-262k, prefill-heavy / many long reqs | **nospec + turboquant_4bit_nc** (0.9) | best prefill at depth (2232 t/s @227k) + 1.3M-token pool (4.9× concurrency headroom) |
| High concurrency short-ctx | **nospec + fp8** (200.6) or **nospec + tq4** (160.5, if capacity matters) | fair per-client; spec loses 27-45% |
| fp8_e4m3 + spec (if required) | **GMU 0.85** (maxlen 262144 OK) | 0.9 collapses 1.7-2.6× (D1); 0.85 == Phase-V refs 41.9/27.5/22.3-class |
| dflash7 | eval/experiment only | 3rd at everything; 128k cliff (D10); loop-risk ledger (Phase F greedy 4/4) |

Env knobs: keep defaults (TQ MQ splits 32; graph grid 256 w/ tier; `VLLM_XPU_FP8_MQ=1`). `VLLM_FP8MQ_SPLITS=64` only for fp8 long-ctx once D1 fixed (v51: +34% @32k). Graceful `docker stop -t 30` before rm (KNOWN_ISSUES #03 — xe wedge).

---

## 6. Validation harness (reusable regression suite)

On host `/root/build/` (source archived in repo `.tmp-tq/`):
- `serve_bench.sh <kv> <specjson|nospec> <gmu> <maxlen> <log> <env> <flags> <image>` — parameterized boot, health-gated, dead-container log salvage.
- `run_lane.sh <tag> …` — boot→warmup→battery→ctxscan(2k-65k)→deep(128k/227k)→conc(8@2k, 4 mixed, 2@65k)→health tail (STALLS/STRIKES/F8/JIT/ERRS).
- `master125.sh` — the 5-lane matrix; `post125.sh` — warm deep re-runs + fp8 A/B (legs A/B/C).
- Regression gates: battery SHA set, ctxscan ±10% of this report's table, deep decode ±10%, conc8 agg ±10%, 0 ERRS/STRIKES.

---

## 7. Loop / zombie / memory posture

- **Zombies/crashes**: v52m+F8v2+F9 held through ~3.5 h sustained matrix: 0 strikes, 0 engine deaths, 1 benign F8 park (nospec), hard-close not re-tested today (v51 dt_disc certified 0.54 s dual-endpoint).
- **Loops**: none observed at temp 0 in any lane today (VARIED fillers; #13 mitigated in harness; #20 loops are budget-limited model behavior, dtype-independent).
- **Memory**: pools as §1; D3's ~1.9 GiB transient at deep prefill is the only unmanaged spike (L6 removes it). No expandable-segment regressions observed (env kept `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`).

---

*Appendix: raw lane reports on host under `/root/build/bench125/lane_*/report.txt`; post A/B in `/root/build/bench125/post.log`.*
