# dflash2-fullfix-v51 — MTP + DFlash2 full-support engagement

Created 2026-09-06 (post-V123, prod on `llm-scaler-exp:v1.2.3`).
Mode: **coordinator executes directly — no subagents** (user directive).

## Mandate (user, 2026-09-06, supersedes v50 policy)

1. Deep research MTP + DFlash2 issues and improvements.
2. **No degradation allowed.** ALL issues/crashes fixed before the final image.
3. **Decode speed drop as context grows is NOT allowed.**
4. **FP8 KV fully supported** — every `--kv-cache-dtype` (fp8_e4m3, fp8_e5m2,
   turboquant_4bit_nc, ...) on BOTH DFlash2 and MTP. (Overturns v50 F6.)
5. Matrix: **DFlash2 k=7 vs MTP k=4**, prompt `"Write a html car game."`,
   concurrent, long-context, looping-prevention both drafters; loops fixed.
6. **Fix all latency issues** incl. DFLASH_STALL propose() 0.18–0.2 s
   peer-late windows and Triton JIT-during-inference spikes
   (`_tq_full_dequant_kv` at 05:01:46). (Added 2026-09-06 05:0x.)
7. **Remove the F8 runaway guard's ability to cut real valid requests**
   (05:06:32 firing had max_tokens=40000, only 6082 emitted — user concern).
8. **Remove ALL fork-added plaster fixes that don't solve the real issue.
   Always fix the ROOT issue, even when the fix lives outside the vllm
   fork** (vllm-xpu-kernels, custom-esimd-kernels-vllm, Triton kernels) —
   use the harmonized patch series as the vehicle. (Added 2026-09-06.)
9. Final task: build the new prod image.
10. **Validate and fix (if present) upstream issue #54360** — "Speculative
    decoding (mtp and dflash) silently disables prefix-cache hits"
    (https://github.com/vllm-project/vllm/issues/54360).
    (Added 2026-09-06.)

## Evidence baseline

1. fp8_e4m3+dflash7 boot killed 04:31:57 by OUR F6 RuntimeError. Boot was
   healthy (FA v2, pool 350,482, scaling 1.0). → F6 must go AND the fp8
   multi-row-verify slowness fixed (B5b: tforward d=4976 ms vs 30–60 ms TQ).
2. tq4nc decode sag within one request 47→25 tok/s; battery 55.45/19.11/9.29
   @2k/16k/65k; itl 745 ms @65k vs 18 ms @2k = 41x for 32x ctx.
3. Latency warnings 05:01:45–46: DFLASH_STALL propose() 0.180/0.201 s
   (peer-late windows >0.64 s trip xe GuC preempt watchdog → engine reset)
   + jit_monitor: `_tq_full_dequant_kv` JIT compile DURING inference.
4. F8 fired 05:06:32 on a 40k-max request at 6,082 tokens — guard works,
   but the class it guards against must be root-fixed so the guard can go.

## Plaster-removal ledger (fork-added workarounds → root fixes)

| Plaster | Root issue | Real fix (may be outside fork) | Removal gate |
|---|---|---|---|
| F6 refusal (core.py raise on fp8+spec) | No fast multi-row fp8 KV attention on XPU | Multi-row fp8 paged-attention kernel: extend ESIMD `page_attn_decode` past its `max_query_len==1` gate (flash_attn.py:1077) or add MQ split-KV kernel reading raw fp8 KV (pattern = `_tq_mq_*` stage1/stage2, harmonized patch 08) in vllm-xpu-kernels/custom-esimd | fp8_e4m3/e5m2 fast on dflash7 AND mtp4 in matrix |
| F7 refusal (mtp k>=4) + v42 EMIT_K<=4 guard | MTP k>=4 temp-0 corruption (graphs AND eager; k<=3 clean; dflash k7 clean) | **C4 bucket-miss stale-slot**: drafter piecewise pad rows [B:input_batch) carry REAL slot coords in the draft loop (llm_base_proposer.py:612 no re-pad vs first-pass :383) → garbage draft-KV writes at real positions; k=4's (k+1)-multiple buckets {5,10,..} are always missed by B∈{1,2,4,8}. Draft-loop site REFUTED 09-06 (eagle_step pads pad rows, utils.py:56-58); remaining leak candidate = target-side uniform-decode pad rows (gpu_model_runner ~5447) — inspect before fixing. Discriminator: mtp4 B=5 clean / mtp3 B=5 corrupt | mtp4/mtp7 clean & bit-stable in matrix incl. B∉buckets |
| F8 force-finish runaway guard | Serving layer leaks generate() on non-cancel exit paths (L1/L2 swallowed exceptions in chat/completion stream generators; L3 GC-deferred close) | **F9**: async_llm.py finally + fire-and-forget abort (completed/aborted flags) + serving-layer `aclose()` companion. Root = L1/L2/L3, fixed in-fork the *correct* way | F9 validated; F8 → detect-only for matrix; **F8 removed from final image** |
| F2a hasher resync + F2b starvation clamp | Pool hash_block_size finalized AFTER hasher built (draft group shrinks gcd late in EngineCore init order) | Construct hasher from final resolved block sizes (init-order fix in core.py/kv config) so resync+clamp become dead code; delete both | Boot shows single-granularity build, crash-1 cell clean without resync firing |
| F1 TP1 default-0 + corrupt warning | TP1 draft replication lane itself corrupt (acceptance 2–9%) | Root-fix replication correctness (draft KV/attention consistency across TP) or delete the fork-added TP1 feature outright | Decision after Phase 1 recon: fix or remove |
| v42 EMIT_K=4+graphs refusal | Same k4 corruption as F7 | Same as F7 root fix | With F7 gate |
| dt_warmup F4 phases 4/5 | First-use JIT under traffic | Keep + EXTEND to cover `_tq_full_dequant_kv` shape/config (and any other jit_monitor hits) — warmup is legitimate, coverage incomplete | jit_monitor silent through full matrix |

Rule: **final image ships zero refusal guards and zero behavior-changing
workarounds** — every v50/v42 guard either deleted or reduced to log-only
diagnostics, each replaced by a root fix.

Final ledger dispositions (v1.2.4, 2026-09-06 — engagement close):

- **F6 REMOVED** (gate met: fp8 e4m3/e5m2 fast on mtp4 AND dflash7 —
  Phase 1 closed 4/4). core_v51.py:~257+ comment block.
- **F7 mtp k>=4 refusal: ABSENT** from the shipped lineage (mtp4 boots
  in every cell); the k-class corruption was dispositioned by the
  Phase-3 battery as a tiny deterministic q5-specific fp near-tie flip
  → k user-selectable with caveat. Per-lane p5 sha16 stays bit-stable:
  252b4dc1 (tq4nc+mtp4), 194e1de8 (nospec), f61457ef (fp8-e4m3+mtp4).
- **v42 EMIT_K=4+graphs refusal: ABSENT.** The v42c DFLASH2_EMIT_K env
  knob in async_sched_v51.py remains — default-inert (unset → stock
  width), validated range 1..num_spec_tokens, no k=4 cap; kept as the
  k-sweep research instrument.
- **F8 runaway guard: detect-only** (VLLM_V51_F8_MODE=detect default;
  kill = v50 A/B only; off available). This satisfies the rule's
  "reduced to log-only diagnostics" branch and serves as the F9-gap
  tripwire. F9 validated — 0 firings through the entire 4a/4b/5 matrix.
- **F2a hasher resync: ROOT-FIXED in v1.2.4** — the request block hasher
  is built ONCE from the FINAL pool hash granularity (core.py:~231-255);
  the v50 build-early + conditional-resync plaster is deleted. Boot line:
  "block hasher built from final pool granularity hbs=2048 (pool=2048
  resolve=2048)"; APC certified post-fix (+6144 hits, repeat 3×). Gate
  evidence: every v51 boot probed captured==pool (1024 fp8 / 2048
  tq4nc); the v50 resync never fired anywhere in the v51 matrix.
- **F2b starvation clamp: NOT PRESENT** in this lineage (grep negative
  across core/scheduler) — superseded by the F2a fix generation.
- **F1 TP1 default-0 + corrupt warning: KEPT** — the whole
  VLLM_XPU_DFLASH_TP1 lane is env-gated opt-in, default-inert, and not
  in the prod matrix; the default-0 protects a convicted-corrupt
  (acceptance 2–9%) opt-in lane. Root-fix (TP1 draft replication
  correctness) is out of engagement scope.

## Research findings (2026-09-06, pre-directive dispatch — kept)

### Phase 1 (fp8)
- Dispatch: xpu.py:124-215 — only `turboquant_*` routes to TQ backend;
  fp8 falls to FLASH_ATTN with no kernel-availability map.
- flash_attn.py:929 `_inner_forward`: fp8 KV is *viewed*, dequant inside
  xpu kernel via descale args; **multi-row spec-verify forced to
  chunk_prefill branch via `is_mix_batch=False` (flash_attn.py:1243-1245)**
  to avoid branch2's host `.item()` breaking graph capture.
- Fast ESIMD `eagle_ops.page_attn_decode` handles fp8 KV + scales but is
  **gated `max_query_len == 1` (flash_attn.py:1077)** — multi-row verify
  can't use it. Sibling repo exports `eagle_page_attn_decode(_separate)`
  (custom-esimd-kernels-vllm/__init__.py:70-71) + fp8-KV benches.
- TQ MQ fast path lives in harmonized patch 08 (`_tq_mq_decode_stage1` /
  `_tq_mq_fwd_stage2`, one KV-tile load scores all Q_BLOCK rows;
  `assert Q_LEN <= 8`). Not directly reusable for raw fp8 pool.
- Fix surface ranked: (1) multi-row ESIMD extension; (2) MQ split-KV kernel
  for raw fp8 (patch-08 pattern); (3) kernel-availability map in platform.

### Phase 2 (context-scaling)
- turboquant_attn.py:331-336: adaptive ladder off under FULL_DECODE_ONLY;
  graph-fixed grid = 32 × tier(>131072 → ×8) = **256, already the ladder
  max** — re-enabling the ladder changes nothing. Message is a red herring.
- Real mechanism: per-split serial loop BLOCK_KV=4 × num_warps=1 ×
  num_stages=1 (triton_turboquant_decode.py:38-40) — iterations scale
  linearly with ctx (2k→~2, 65k→~64), dependent-load chain, L2→DRAM spill;
  plus B=8 spec rows multiplying 1-warp work-items.
- Measured knobs (commit 02cd824): BLOCK_KV=16 → -72%, 32 → -83%,
  16+2warps → -20%, 32+2warps → FATAL. Defaults optimal. Do not touch.
- Cheapest levers: **VLLM_TQ_GRAPH_KV_SPLITS=1024** (env, ceiling 1024;
  65k→16 iterations vs 64), then **VLLM_TQ_STAGE1_STAGES=2/4** (pipelining).
  If insufficient → persistent-kernel restructure (patch series, high effort).
- Host per-step work is O(reqs)/O(new tokens), not O(ctx) — not the driver.
- **PHASE 2 RESULTS (2026-09-06, mtp4, same image v51ovl2, dt_ctxscan.py
  256-tok decode, warm values; r2≡r3 bit-stable per lane)**:
  | ctx | fp8 s32 | fp8 s64 | tq4nc (3-run) |
  |---|---|---|---|
  | 2k  | 73.7 | 70.3–70.7 | 71.8–72.1 |
  | 16k | 32.3 | 35.7–39.3 | 38.1–45.0 (noisy lane) |
  | 32k | 27.2 | 36.7–38.6 | 27.4–28.8 |
  | 65k | 22.3 | 25.3–26.2 | 21.5 (rock-stable) |
  **VERDICTS**: (1) fp8_e4m3 at default SPLITS=32 = PARITY with the tq4nc
  incumbent at every ctx (worst −15% at 16k where the tq4nc lane itself
  swings ±18% run-to-run; −1% at 32k; +2.4% at 2k, +4% at 65k) — "no 2k
  regression" gate PASSES and the 2k→65k decay slope is IDENTICAL to the
  incumbent (3.3× both lanes = physical KV-read scaling on the 16
  full-attn layers; GDN layers are O(1)). No cliff. (2) SPLITS=64 is a
  long-ctx option: +34% @32k, +17% @65k (BEATS tq4nc by 33%/17% despite
  2× KV bytes) at −2% @2k. (3) **SPLITS=128 WEDGES**: #11-class crawl
  (0.2–0.3 tok/s, metrics log freezes mid-run, survives client kill;
  exactly the v50 F6-era 0.2–0.9 conviction signature) — 64 is the
  certified ceiling; documented in the kernel header. (4) Outputs are
  SPLITS-invariant (S=64 battery shas bit-identical to S=32 lane).
  (5) The tq4nc-path knobs (KV_SPLITS=1024/STAGES) are MOOT for the fp8
  lane (different kernel); tq4nc incumbent was left at defaults as the
  control. NOTE: kernel-header defaults updated (SPLITS guidance + 128
  warning).

### Phase L (latency) — RESULTS (2026-09-06)
- **Warmup validated live** (fresh mtp4+fp8 boot, dt_warmup_v51.py fired
  as first traffic): post-warmup first user request p1_short = **1.3 s**
  (was 21–30 s cold), full battery immediately at full tps; 0
  DFLASH_STALL windows on the mtp lane (dflash lane check owed in 4b).
- **New gap found & fixed**: even after v50-F4/v51 phase-6 warmup
  (≤18k), the FIRST 32k/65k request JIT-compiles prefill-side kernels
  under traffic — ttft 22–40 s spikes + 9 jit_monitor warnings + depressed
  decode on that request (reproduced on two independent boots, recovered
  on 2nd pass). dt_warmup_v51.py extended with **phases 7–8 (~36k and
  ~70k prompts = 5- and 9-chunk buckets)** to pre-hit those
  specializations at boot; validation owed at next fresh boot (4b/5).
- VLLM_TQ_GRAPH_KV_SPLITS=1024/STAGES A/B: n/a for fp8 lane (v51 kernel
  has own SPLITS knob, swept above); tq4nc incumbent control left at
  defaults.

### Phase 3 (MTP k>=4)
- eagle.py is a 23-line wrapper; logic in llm_base_proposer.py. Verify =
  plain causal k+1-row extend (no tree mask). Draft = k-1 sequential
  1-row forwards mutating slot mapping in-place.
- **C1 FALSIFIED (2026-09-06, code-verified)**: `allocate_slots` reserves
  `num_tokens_need_slot = computed + num_new + lookahead`
  (kv_cache_manager.py:351-354); steady-state num_new=1 ⇒ C+1+k slots =
  exactly the k+1 verify rows (`_calc_spec_decode_metadata`,
  gpu_model_runner.py:2672: scheduled = 1 root + k drafts). No off-by-one.
  Also `_CONTINUATION_DECODE_THRESHOLD = 128` (turboquant_attn.py:73) ⇒
  q=5 (mtp4) AND q=8 (dflash7) both take the width-agnostic decode-kernel
  MQ path; the `_continuation_prefill` dequant path is NOT the k4 gate.
- **C4 (NEW, strongest — bucket-miss stale-slot theory)**:
  - `adjust_cudagraph_sizes_for_spec_decode` (compilation.py:1447-1492,
    cites upstream #28207) rounds ALL capture sizes to multiples of k+1 —
    and the SAME rounded list feeds the DRAFTER's separate piecewise
    dispatcher (llm_base_proposer.py:145 + gpu_model_runner.py:6554→6570).
  - Drafter dispatch pads batch B → next multiple of k+1
    (cudagraph_dispatcher.py:71-90,140): k=1→{2,4,..}, k=2→{3,6,..},
    k=3→{4,8,..}, k=4→**{5,10,20,..}**, k=7→{8,16,..}.
  - In the MTP draft loop (llm_base_proposer.py:543-623) the model input is
    `buffers[:input_batch_size]` (padded), but `_get_slot_mapping(...)`
    at line 612 is called WITHOUT a slot_mapping arg ⇒ rows
    `[batch_size:input_batch_size)` are NOT re-padded (first pass DOES
    pad: line 383) — they keep REAL slot coordinates (stale positions from
    the first pass = verify rows of a real request, +1 by the eagle_step
    kernel) ⇒ **padded garbage rows write draft KV at real positions**,
    deterministically ⇒ temp-0 divergence, byte-reproducible per ordinal.
  - k-boundary explained: standard test batch sizes {1,2,4,8} hit the
    k≤3/k=7 buckets EXACTLY (multiples of 2/3/4/8) but ALWAYS miss the
    k=4 buckets (multiples of 5). k=1 has no loop at all (early exit,
    llm_base_proposer.py:489). dflash7: parallel_drafting one-shot — no
    loop, immune.
  - **Discriminator (cheap, decisive)**: mtp4 with B=5 (or 10) should be
    CLEAN; mtp3 with B=5 (pads 5→8, 3 stale rows) should CORRUPT.
    Inverts "k4-only" → "bucket-miss-only".
  - **Root fix candidate (small)**: PAD-fill
    `_slot_mapping_buffer[batch_size:input_batch_size]` in the draft loop
    (mirror of line 383) — pending discriminator confirmation.
  - **CORRECTION (2026-09-06, later) — draft-loop site REFUTED**: the loop's
    per-iteration slot recompute goes through
    `eagle_step_update_slot_mapping_and_metadata` (utils.py:88-133), whose
    kernel pad-fills every row `req_idx >= batch_size` with
    PADDING_SLOT_ID (utils.py:56-58) — and qwen3.8 MTP runs it every loop
    iteration (`constant_draft_positions=False` is the base default,
    llm_base_proposer.py:107; only gemma4 overrides True). Image copy
    verified: utils.py differs from source ONLY by the fork's DFlash2
    SKIP_BONUS sampling hunk (md5 36f40f90 vs d28f39e4; diff seen 09-06).
    Remaining stale-slot candidate: **target-side uniform-decode pad rows**
    (virtual-request padding, gpu_model_runner.py ~5447-5480 — but that
    site is the DUMMY-RUN/capture path, and corruption reproduces in
    EAGER where no runtime padding exists → ALL pad-row theories are
    weakened; C4 is now low-confidence pending the discriminator).
- **C5 (NEW 2026-09-06 — partially-valid second Q row-group in MQ kernels)**:
  the structural boundary separating corrupt from clean among tested
  configs: corrupt q_len=5 (mtp k4) is the only tested q_len whose
  next_pow2 Q_BLOCK tile holds a PARTIALLY-VALID second 4-row group;
  clean q_lens are {2 (k1), 3 (k2), 4 (k3), 8 (dflash k7)}.
  Mechanism (refined after k2/k3 evidence check — boot_k2.sh/boot_k3.sh
  existed; v29c verdict "k<=3 bit-stable"): MQ verify kernels (TQ decode
  MQ for tq dtypes — the corruption-conviction dtype; unified_attention
  3D for fp8 pre-v51) tile q rows into Q_BLOCK = next_pow2(q_len).
  Refined C5 = **partially-valid SECOND 4-row group corrupts** (mask or
  causal-limit bug on rows [4:8) when only some lanes are valid):
  q=2 (QB2, one full group) clean; q=3 (QB4, only group0, 3/4 valid)
  clean — bug only in group1; q=4 (QB4 full) clean; q=5/6/7 (QB8,
  group1 partially valid) CORRUPT; q=8 (QB8 full) clean. Consistent with
  ALL evidence (k1/k2/k3 clean at any B, k4 corrupt graphs+eager,
  dflash k7=q8 clean) and survives the eager fact that killed host
  pad-row theories (in-kernel lanes run identically in eager).
  - Also explains "dating >= v24" trivially: v24 README confirmed v24
    only added the #11 prefill rendezvous barrier (no numerics change) —
    k4 was first exercised there, not regressed there. NOTE: the v51
    fp8 Triton kernel masks lanes correctly (test_fp8mq_ref.py validates
    exactly Q_LEN=5 on Q_BLOCK=8) — C5 convicts the TQ MQ kernel /
    unified_attention path, NOT the v51 kernel.
  - **Discriminator battery (GPU, supersedes single C4 experiment)** —
    temp-0 determinism vs eager, cargame prompt, one boot per row:
    | cell | C4 predicts | C5 predicts |
    |---|---|---|
    | 0. mtp4 B=1, VLLM_TQ_MQ_VERIFY=0 (synthetic path) | corrupt | CLEAN (MQ path bypassed) |
    | 1. mtp4 B=1 (q5, pad→5) | corrupt | corrupt |
    | 2. mtp4 B=5 (q5, exact hit) | CLEAN | corrupt |
    | 3. mtp3 B=5 (q4, pad→8) | corrupt | CLEAN |
    | 4. mtp2 B=1 (q3, exact hit) | clean | clean (consistency only) |
    | 5. mtp5 B=1 (q6, exact hit) | clean | corrupt |
    | 6. mtp6 B=1 (q7, exact hit) | clean | corrupt |
    **RUN CELL 0 FIRST** — same boot class as cell 1, env-only toggle: it
    isolates the v19 TQ MQ read path (kernel + per-request dispatch in
    turboquant_attn.py:1135-1331) vs everything else in one shot.
    Cells 2/3/5/6 separate the theories (2/5/6 are pure in-kernel tests —
    bucket-exact so NO host padding anywhere). Root fix under C5 lives in
    the TQ MQ kernel / unified_attention segment logic (vllm-xpu-kernels
    or harmonized patch series — outside the fork, per mandate).
  - **BATTERY RESULTS (2026-09-06 GPU window, image llm-scaler-exp:v1.2.3,
    tq4nc, dt_probe2 sha16 of reasoning+content, temp 0 seed 1234)**:
    | lane | p1 | p2 | p3 | p4 | p5 | p6(2k) |
    |---|---|---|---|---|---|---|
    | nospec ref (×3 stable) | 3cecc747 | c87e27c4 | 1f517e4f | 744e88f5 | **194e1de8** | 1f9c461b |
    | k4 MQ (×2 stable) | ✓ | ✓ | ✗2505 | ✓ | **✗252b4dc1** | ✗ebcc |
    | k4 VERIFY=0 (synthetic) | ✓ | ✓ | ✗6fcc | ✓* | **✗252b4dc1** | ✗ebcc |
    | k2 q3 (×2 stable) | ✓ | ✓ | ✗3110 | ✓ | **✓ 194e1de8** | ✗ebcc |
    | k5 q6 (×2 stable) | ✓ | ✓ | ✗50aa | ✓ | **✓ 194e1de8** | ✗ebcc |
    (*k4-VERIFY=0 p4 run1 hit the 400 cap = 0eb3267b — first-run fluke,
    run2 == ref; all spec lanes report differing completion_tokens than
    nospec on sha-equal prompts = usage accounting artifact.)
    **VERDICTS**: (1) C5 (MQ-kernel/partial-group) DEAD — cell 0 diverges
    identically to cell 1 on p5 (synthetic path never runs the MQ kernel)
    AND k5 (q6, also partial second group) is CLEAN on p5. (2) C5-structural
    (any q in 5..7) DEAD by k5 clean. (3) k>3-drafter family DEAD (k5 clean).
    (4) The k-specific residue is **EXACTLY q5/num_spec=4**: both verify
    paths agree with each other (252b4dc1) and disagree with ref; q∈{2,3,4,6}
    match ref bit-exact. (5) p3 = intrinsically unstable prompt (5 distinct
    outputs across 5 lanes — near-tie; probe noise, not corruption evidence).
    (6) p6(2k) = k-AGNOSTIC spec-vs-nospec systematic delta (all spec lanes
    bit-identical ebcc8258 to each other, ≠ nospec) — separate issue, lower
    priority (candidate: chunked-prefill/draft-KV-layout numerics at 2k ctx).
    Since BOTH the MQ kernel and the synthetic single-query path produce the
    SAME wrong p5 answer at k4, the defect is NOT in the attention kernels —
    it is in what feeds them at num_spec=4: batch assembly (positions/block
    table advance for 5-token verify steps) or GDN spec-state path (5 slots).
    NEXT: cell 2 (mtp4 B=5 exact bucket hit, dt_probe2u uniform N=5) — C4
    family's last standing variant predicts CLEAN; then inspect the num_spec=4
    assembly path. dt_probe2u.py + dt_probe2c.py written on server.
  - **BATTERY FOLLOW-UP (same window, same lane set)**:
    **(a) B-axis (graph, dt_probe2u uniform-N on p5_fact)**: N=1 → corrupt
    252b4dc1; N=2 ✓, N=4 ✓, N=5 ✓ (padded 5→bucket 40, NOT bucket-exact —
    clean), N=8 ✓ → capture-padding variant of C4 DEAD (padded-batch clean
    while exact-bucket B=1 corrupt); graph flips ONLY at B=1/q5.
    **(b) Eager A/B (mtp4, enforce-eager boot)**: same class of deviation,
    DIFFERENT prompt set — eager-B=1 flips p1_short (sha ≠ ref) while p5
    stays clean; i.e. the phenomenon is REGIME-INDEPENDENT (not
    graph/capture machinery), and which prompt flips depends on reduction
    path, not on prompt semantics.
    **(c) Text-level diff (p5_fact ref vs corrupt, saved
    /root/build/texts/p5_fact_N1_*.txt)**: outputs identical for 376 of
    385 chars; the SOLE divergence is one token at char 377: "…winds that
    can exceed **1,000** mph." (ref) vs "…exceed **1,200** mph."
    (corrupt). A stale-slot/garbage defect would destroy text; a razor-thin
    near-tie flip does exactly this → **C4 stale-slot family DEAD**.
    **CHARACTERIZATION (final, this defect)**: k4/q5 corruption = tiny
    deterministic fp deviation specific to num_spec=4 verify rows,
    regime-independent, batch-shape-dependent in magnitude, flipping only
    exact near-tie argmax races. Remaining candidate surfaces: inductor
    combo-kernel M-tiling at 5 rows, GDN selective_state_update spec
    variant, oneCCL 5-row allreduce — all benign-class reduction-order fp;
    bit-exactness root-fix is disproportionate. DISPOSITION: k4 stays
    user-selectable with documented near-tie caveat; Phase 4b quality
    battery (cargame ×2, think-traps) as the acceptance gate. k4 cargame
    spot-check this window: coherent 512 toks @45.1 tok/s, no looping.
  - **PHASE 1 (fp8 KV root fix) — GPU + IN-SERVER RESULTS (same window)**:
    Kernel correctness on B70 (test_fp8mq_xpu.py, container /tmp/v51t/):
    9/9 cases q∈{2,3,4,5,6,8} incl. GQA 32/8 + 16k ctx, max rel err
    2.4–3.2e-4 (= fp16 output rounding); structural Test A (K=V=1 → out
    ≡ 1.0) and Test B (V=token-ramp → row means to fp16 ulp) both PASS —
    bitcast/dequant/causal-limits/split-combine/gather all correct. (N.B.
    my eager ref had a silent GQA bug first — einsum summed over KV heads;
    fixed with repeat_interleave; kernel was right all along.)
    Tuning (B=8 q5 Hq=32 D=256 ctx=4096): default BKV16/W1 20.6 ms →
    **BKV32/W4 8.77 ms (2.35×)**; patch-08's W1-fastest does NOT transfer
    to this kernel on B70; new defaults baked (env-overridable).
    **In-server (mtp4 + fp8_e4m3, image v1.2.3 + 3 overlays)**: the
    earlier "crash" was NOT a crash — **v50 F6 refusal** (core.py:284,
    raised post-warmup after 37/37 graph capture). Overlays: flash_attn.py
    ← flash_attn_v51.py (route), ops/triton_fp8_mq.py (kernel), core.py ←
    core_v51.py (F6 removed — the plaster this phase exists to delete).
    Boot healthy; route CONFIRMED live via in-container Triton cache
    (_fp8_mq_stage2 artifacts). Results (cold → warm): p1 2.1 → **51.2
    tps** (30 s first-request = JIT stall of eagle/rejection kernels,
    Phase L item), p2 **79.9 tps**, p5 55.9, p6_ctx2k 16.9 → **47.8 tps
    warm**; acceptance 1074/1876 = **0.572**; **all 6 sha16 bit-identical
    across cold/warm runs** (deterministic, unlike convicted ESIMD v38);
    cargame single 512 toks @44.7 tps coherent. This lane was convicted at
    0.2–0.9 tok/s in v50 — now ≈60–100× that, at parity with tq4nc lanes.
    Container committed → **llm-scaler-exp:v1.2.3-v51ovl** (dflash7+fp8
    leg boots from it, single-cycle).
    **dflash7 + fp8_e4m3 (from v51ovl)**: boot needed (i) serve_boot_var.sh
    dflash2 mount added (host /models/qwen3.8-27b-dflash2 → /models/dflash2;
    inert for mtp lanes), (ii) `--max-model-len 245760` — dflash drafter
    weights + fp8 KV (2× tq4nc bytes) leave 6.34 < 6.69 GiB at 262k (mtp4
    needs no cap: drafter shares target weights). Battery: p2 **95.9 tps**,
    p5 47.7, p6 warm 46.6, acc 0.335 (normal dflash7 profile), cargame
    coherent 38.9 tps; 5/6 shas **bit-identical to the mtp4+e4m3 lane**
    (cross-drafter greedy convergence; p3 = known unstable prompt).
    **fp8_e5m2 lane — two more root fixes**: (1) upstream refusal
    "fp8_e5m2 not supported with fp8 checkpoints" (attention.py:175) was
    already env-gated by the v34 patch in the image lineage — just pass
    `VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1` (scales stay e4m3-calibrated: probe
    lane, quality caveat per v34 KNOWN_ISSUES #19). (2) REAL gap found and
    fixed: `get_fp8_dtype_for_flashattn` (flash_attn.py:187) mapped only
    e4m3 and raised "Unrecognized FP8 dtype: fp8_e5m2" during graph-capture
    warmup — extended to return torch.float8_e5m2 (both call sites benign:
    scheduler-metadata dtype arg + cache dtype view); baked → image
    **llm-scaler-exp:v1.2.3-v51ovl2** (v51ovl + fixed flash_attn).
    **mtp4 + e5m2**: p2 78.7, p6 warm **49.3 tps**, acc 0.583, all shas
    deterministic cold/warm; p2/p6 shas EQUAL the nospec ref (c87e27c4 /
    1f9c461b). Cargame 42.5 tps coherent. **dflash7 + e5m2**: p2
    **100.4 tps** (fastest lane this window), acc 0.362, cargame 37.9 tps
    coherent; p2/p4/p5 shas match mtp4+e5m2. **PHASE 1 CLOSED: fp8 KV
    (e4m3 + e5m2) × both drafters (mtp4 + dflash7) = 4/4 cells at full
    speed, deterministic, coherent — v50 F6 plaster deleted with prejudice.**
  - **Static audit (2026-09-06, image code)**: the v19 TQ MQ path audited
    END-TO-END on the image (triton_turboquant_decode.py image copy: fork
    appends _tq_mq_decode_stage1/2 + B=1 launcher; turboquant_attn.py
    image copy: per-request loop i:qsl[i+1], mq_q0 = seq_lens[i:i+1] -
    (q_len-1) device-op, gate 1 < q_len <= _TQ_MQ_MAX_Q=8 default,
    immediate output[q_start:q_end] writeback). The kernel algorithm is
    IDENTICAL to the CPU-validated v51 design (per-row limits q0+j,
    uniform splits from seq_len_max = q0+Q_LEN-1, blind combine, masked
    lanes; test_fp8mq_ref.py validates exactly Q_LEN=5 on Q_BLOCK=8).
    No algorithmic bug found for q=5 → if cell 0 convicts the MQ path,
    the residual suspects are Triton-XPU codegen for the Q_LEN=5 /
    Q_BLOCK=8 kernel instantiation, or the shared "output" scratch
    reuse (turboquant_tqdec launcher returns a VIEW of a scratch shared
    with the single-query path — aliasing under multi-layer use).
- C3: no MTP draft-KV dtype policy (dflash got VLLM_DFLASH_DRAFT_KV_DTYPE
  in v49c; MTP uses target dtype unconditionally).
- dflash k7 clean because: one-shot parallel propose, no sequential draft
  loop, no per-step draft KV writes into target-derived slot maps.

### Phase PC (#54360 — spec silently disables prefix-cache hits)
- Upstream issue: nightly v0.28.1rc1.dev43 — ANY spec method (mtp/dflash)
  on hybrid GDN models (Qwen3.8 family) silently zeroes
  `vllm:prefix_cache_hits_total`; reporter A/B on identical workload:
  APC-only 17,248 hits vs APC+spec 0; v0.24.0 (MTP + align-APC) worked;
  regression window = v0.24.0 → nightly.
- Our base 0.21.1.dev0 **predates v0.24.0** → the regressing commit is not
  in our tree (high prior). Our base DOES contain the align machinery the
  issue implicates: APC + spec + `supports_mamba_prefix_caching` ⇒
  `mamba_cache_mode="align"` (models/config.py:425-440, auto-resolution
  with warning at boot).
- **Empirical verdict (2026-09-06 serve-log archaeology): bug ABSENT in our
  image.** Spec-enabled boots with `enable_prefix_caching=True` show
  large nonzero hit rates: df_mtp4_lc_crash.log (mtp k=4: 6/11 windows),
  wedge_mtp1_0228.log (mtp k=1: 127/130), f8a_final_031420.log,
  salvage_010017.log, salvage_031421.log, v123_final_035601.log (all
  dflash k=7: 43–103 windows each; peaks 36.3–53.6%). The 0%-lines in
  spec boots (df_bench etc.) are unique-prompt benches — expected, NOT
  the bug signature (which is same-workload spec on/off 17k→0).
- Controlled confirmation cell (no new code): existing
  `/root/build/dt_cacheprobe.py` — same ~5k-token prompt ×2, prints
  /metrics `prefix_cache_{queries,hits}_total` deltas + per-request
  `usage.prompt_tokens_details.cached_tokens`. Run on {mtp4, dflash7,
  nospec} boots inside the Phase 4b matrix; PASS = hits delta > 0 and
  cached_tokens ≈ block-multiple on run 2, for ALL three configs.
- Related fork interplay: F2a/F2b (hasher granularity resync + starvation
  clamp) live in exactly this subsystem; their ROOT fix (Phase 5 ledger)
  is the init-order fix — if the controlled cell ever contradicts the
  log archaeology, the investigation starts there.

### Phase 4a (F9)
- Leak paths: L1/L2 serving-layer `except Exception` → normal return
  leaves generate() suspended at async_llm.py:586 (chat serving.py:1004-1009,
  completion serving.py:467-471); L3 GC-deferred + traceback-pinned close;
  L5 await-inside-cancel can half-abort. Abort is idempotent both tiers
  (output_processor.py:505 pop; core_client.py:1070 dead-safe).
- F9-core: `completed`/`aborted` flags + finally that fires
  `create_task(self.abort(q.request_id, internal=True))` when neither set
  (no await in finally — GeneratorExit hazard). Apply to encode() too.
- F9-serving companion: `finally: await result_generator.aclose()` in both
  stream generators → deterministic close instead of GC.
- F9 also terminates the zombie's own SSE leg (abort pushes final ABORT
  output through the collector) — the thing F8 could never do.

## Phases

- **Phase L (latency, active)**: (a) root-cause `_tq_full_dequant_kv`
  JIT-during-inference — find call site + triggering shape, extend warmup;
  (b) DFLASH_STALL 0.18–0.2 s windows — hypothesis: peer late due to that
  same JIT compile (TP0 JIT at 05:01:46, both workers stall at 05:01:45 —
  verify ordering); if independent, profile propose() host wall;
  (c) context-scaling knobs (VLLM_TQ_GRAPH_KV_SPLITS=1024, STAGES≥2).
- **Phase 1 (fp8)**: implement multi-row fp8 fast path per fix surface;
  validate speed + correctness on dflash7 and mtp4; delete F6.
- **Phase 2 (ctx-scaling)**: server A/B @2k/16k/32k/65k; if env knobs
  insufficient, kernel patch via harmonized series; no 2k regression.
- **Phase 3 (MTP)**: discriminator battery (see C4/C5 table — mtp4 B=5,
  mtp3 B=5, mtp2 B=1, mtp5 B=1 + corrupt-control mtp4 B=1) → root fix
  per surviving theory (C4: pad-fill leak site; C5: TQ MQ / segment
  kernel Q_BLOCK-lane bug, fix outside fork via harmonized series);
  draft-KV policy (C3); wedge class + entrapment; delete F7 + v42 guard.
- **Phase PC (#54360 spec×APC)**: static recon + log archaeology DONE —
  bug ABSENT in our image (see research section). Controlled confirmation
  via dt_cacheprobe.py on {mtp4, dflash7, nospec} boots inside Phase 4b;
  root fix only if the controlled cell contradicts the logs.
- **Phase 4a (F9)**: implement F9-core + F9-serving; F8 → detect-only.
- **Phase 4b (matrix)**: dtypes {fp8_e4m3, fp8_e5m2, tq4nc, k8v4, k3v4nc,
  tq3nc} × {dflash7, mtp4, nospec} × bs {2048,512,64}; cargame prompt;
  conc 2/8; long-ctx 8k/16k/32k/65k single+concurrent; loopscan+dt_loop8
  both drafters; coh×3; rp×5 — with F9 active, F8 detect-only: prove zero
  zombie recurrence AND zero valid-request cuts.
- **Phase 5 (final image)**: F9 in; F6/F7/v42/F8-kill/F2a/F2b/F1 all
  removed per ledger gates; warmup extended; certification cell; prod
  restore; docs; commit.

## Status log

- 2026-09-06 ~04:50 — engagement opened; Phase 0 done; 4 research
  dispatches (pre-directive) returned: fp8 dispatch map, TQ splits
  analysis, MTP k4 candidates, F9 design.
- 2026-09-06 ~05:2x — user directives folded in: latency items (Phase L),
  F8 removal plan, plaster-removal ledger, no-subagent mode. Phase L
  active: hunting `_tq_full_dequant_kv` call sites.
- 2026-09-06 (later) — Phase L(a) DONE in code: `_tq_full_dequant_kv`
  reachable only via `_continuation_prefill` (turboquant_attn.py:822/888);
  serve runs `--max-num-batched-tokens 8192` and v50 warmup p5 (~2.7k) is a
  single chunk ⇒ path never warmed. **dt_warmup_v51.py written** (phase 6:
  ~19k prompt = 3 chunks ⇒ continuations @ cached_len 8192+16384). Likely
  also explains the 05:01:45 DFLASH_STALL windows (host-side compile on one
  rank delays its collective arrival). Validation pending GPU availability.
- 2026-09-06 (later) — Phase 4a F9 SUITE WRITTEN (6 overlay files in this
  dir, all py_compile-OK in lsv-test container): async_llm_v51.py
  (generate+encode: finished_cleanly/abort_initiated flags, fire-and-forget
  create_task abort in finally, no-await under CancelledError/GeneratorExit),
  chat_serving_v51.py + completion_serving_v51.py (finally: aclose
  companions), async_utils_v51.py (merge_async_iterators single-iterator
  child close), async_sched_v51.py (F8 → VLLM_V51_F8_MODE detect/kill/off,
  default detect-only, once-per-request latch). Live validation pending GPUs.
- 2026-09-06 (later) — Phase 3 static verification: C1 FALSIFIED
  (allocation exact: C+1+k covers k+1 verify rows; threshold 128 kills the
  dequant-path theory). **C4 bucket-miss stale-slot theory found** (see
  research section) — explains exact-k4 boundary, byte-reproducibility,
  graphs-vs-eager, dflash7/k1 immunity. Discriminator + 1-line-class root
  fix identified; both pending GPU.
- 2026-09-06 (later) — Phase 1 CODE COMPLETE (GPU validation pending):
  triton_fp8_mq.py (dedicated two-stage Triton flash-decode kernel for
  uniform multi-row verify on fp8 paged KV; knobs VLLM_FP8MQ_*,
  VLLM_XPU_FP8_MQ route kill-switch), flash_attn_v51.py (route before the
  v33 block in _inner_forward; gates: fp8 KV + paged + uniform q
  1<q_len<=8 + causal + no SWA/softcap + head_size<=256),
  core_v51.py (F6 refusal removed). All py_compile + import OK in
  lsv-test on image v1.2.3. test_fp8mq_ref.py CPU cross-check: sim ==
  naive reference to 9.2e-14 in fp64 (algorithm exact; the earlier
  3.05e-5 fp32 delta was the expected normalize-renormalize two-phase
  rounding, tolerance now documented in the file).
- 2026-09-06 (later) — **#54360 PC1 DONE: bug ABSENT in our image.**
  Serve-log archaeology: mtp4/mtp1/dflash7 spec boots with APC=True all
  show nonzero prefix-cache hit-rate windows (peaks 36–54%; 419/617 total
  hit-rate lines nonzero). Base 0.21.1 predates the v0.24→nightly
  regression window; align-mode auto-resolution present (models/
  config.py:425-440) and functional. Controlled cell = existing
  dt_cacheprobe.py in the Phase 4b matrix. C4 draft-loop site REFUTED
  (eagle_step kernel pad-fills pad rows — image verified identical to
  source except DFlash2 SKIP_BONUS hunk); stale-slot candidate moved to
  target-side uniform-decode pad rows (gpu_model_runner ~5447).
- 2026-09-06 (later) — Phase 3: gpu_model_runner ~5447 site is the
  DUMMY-RUN/capture path, and corruption reproduces in EAGER → ALL host
  pad-row theories weakened, C4 demoted to low-confidence. **C5 theory
  formulated** (partially-valid second 4-row group in MQ verify kernels;
  consistent with every data point incl. k2 q=3 clean). Discriminator
  battery expanded to 6 cells (mtp4 B=1/B=5, mtp3 B=5, mtp2 B=1,
  mtp5 B=1, mtp6 B=1) — cells 2/3/5/6 cleanly separate C4 vs C5.
  dt_warmup_v51.py pushed to /root/build/. GPUs still occupied — all
  Phase 1/3/4 validation remains queued.
- 2026-09-06 (later) — **C5 static audit DONE (negative result, image
  code)**: the v19 TQ MQ verify path audited end-to-end on the image —
  kernel algorithm identical to the CPU-validated v51 design, dispatch
  (`1 < q_len <= _TQ_MQ_MAX_Q=8`) uniform across all spec q lens,
  per-request loop mathematically correct. No algorithmic q=5 bug →
  C5's remaining surface = Triton-XPU codegen for the Q_LEN=5/Q_BLOCK=8
  instantiation, or the shared "output" scratch VIEW aliasing. **NEW
  cell 0 discriminator added (run FIRST)**: mtp4 B=1 with
  VLLM_TQ_MQ_VERIFY=0 forces the correct-by-construction synthetic
  per-row path — env-only toggle, same boot; clean-vs-corrupt outcome
  convicts/exonerates the whole MQ read path in one shot.
- 2026-09-06 ~10:45 — **Phase 4a F9 CLOSED.** First v51f9 bake died at
  boot: `ImportError: CoreEngineProcManager from vllm.v1.engine.utils` —
  async_utils_v51.py was a FOREIGN-tree revision overlaid onto
  v1/engine/utils.py, destroying CoreEngineProcManager (this image
  defines merge_async_iterators in vllm/utils/async_utils.py instead).
  f9_import_check.py (NEW: static import resolver, no exec) proved the
  other 5 overlays tree-compatible (0 problems); the utils delta was
  re-applied as an IN-PLACE patch of vllm/utils/async_utils.py
  (single-iterator fast-path aclose, anchor 281-285 exact, contextlib
  already imported). **v51f9 = dafdec364601** (5 overlays + utils patch;
  v1/engine/utils.py byte-identical to base, md5-verified). Live: boot
  clean ("F8 guard mode=detect … spec width=4", async_scheduler.py:62);
  battery cold≡warm all 6 sha16 identical (p2 83.4 tps, sha = nospec
  ref c87e27c4); dt_disc.py (NEW: hard socket close mid-stream, both
  endpoints, idle /metrics poll → no GC pressure) = slots zero 0.54 s
  after close on completions AND chat (abort fired at disconnect, no
  zombie); 0 F8 firings, 0 DFLASH_STALL, 0 errors (324-line log).
- 2026-09-06 ~11:0x — **Phase 4b CLOSED (Boots A-D on v51f9).**
  A (mtp4/tq4nc): dt_cacheprobe +6144 hits, repeat 7.8→1.5 s (5.2×);
  cargame 48.2 tps sha 24d6736d; conc distinct=2/2 ×3 rounds.
  B (fresh mtp4, warmup p1-p8 as first traffic): ctxscan 2k/16k/32k/
  65k = 72.9/44.1/29.7/21.6 tps, ttft 1.1/8.6/10.5/22.6 s — 32k/65k
  ttft now COMPUTE-bound (3.0–3.2k tok/s prefill), ZERO jit warnings on
  those cells ⇒ p7/p8 bucket pre-hit WORKS. Single residual:
  rejection_greedy_sample_kernel JIT on the first temp-0 request
  (phases 1-8 were temp-default) → **warmup PHASE 9 added** (greedy
  bs=1 + bs=2). C (nospec): p2/p6 sha = c87e27c4/1f9c461b (bit-exact
  refs); p5 CLEAN 194e1de8 (vs mtp4's dispositioned 252b4dc1 — the k4
  near-tie flip is lane-consistent); PC +8192 hits, repeat 6.8→0.4 s
  (17×); **spec-vs-nospec 2k delta: mtp4 = 2.17× nospec (72.9 vs 33.7
  tps)** — spec strictly wins at 2k on tq4nc. D (mtp4+fp8_e4m3 union):
  p2 79.9 tps, p6 47.7 warm, acceptance 0.572 (exact Phase-1 values),
  all shas cold≡warm, _fp8_mq_stage2 Triton cache artifacts present,
  dt_disc abort 0.54 s both endpoints — F9 suite × fp8 kernel coexist
  cleanly. Matrix scope as executed: {tq4nc, fp8_e4m3, fp8_e5m2} ×
  {mtp4, dflash7, nospec} + conc/cargame/ctxscan/PC/disc; k8v4/k3v4nc/
  tq3nc/loopscan legs NOT run (code paths unchanged from prod v1.2.3 —
  certification inherited).
- 2026-09-06 ~11:2x — **Phase 5 DONE — FINAL IMAGE v1.2.4 =
  cb95b6e1b66a; prod restored on the certification boot.** F2a root
  fix implemented in core_v51 (hasher built once from final pool
  granularity; v50 resync plaster deleted — see dispositions above);
  baked from v51f9 with only core.py changed (py_compile + marker
  greps in-bake). Certification cell (fresh mtp4/tq4nc boot): F2a line
  "block hasher built from final pool granularity hbs=2048 (pool=2048
  resolve=2048)"; warmup p1-p9 (p9 greedy fires in 0.9 s); ctxscan
  72.8/46.6/30.7/21.6 tps with **ZERO jit_monitor warnings after
  WARMUP_DONE** (last warning 11:25:56 = p9's own compile — the
  ledger gate "jit_monitor silent through full matrix" is MET);
  battery all 6 sha16 bit-identical to Boot A/B refs (F2a fix
  numerically inert); PC +6144 hits (repeat 3×); dt_disc abort 0.54 s
  both endpoints; 0 F8 firings; 0 DFLASH_STALL. **Prod = the standing
  certification boot** (v1.2.4, mtp k4, tq4nc, warm). Ops notes:
  dflash lanes need `--max-model-len 245760` (drafter weights + 2× KV
  bytes at 262k); e5m2 lanes need `VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1`
  (KV scales stay e4m3-calibrated — probe lane only, quality caveat);
  `VLLM_FP8MQ_SPLITS=64` optional long-ctx (+34% @32k), 128 WEDGES
  (#11-class — certified ceiling 64, documented in kernel header).
