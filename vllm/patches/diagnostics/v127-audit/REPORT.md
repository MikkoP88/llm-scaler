# llm-scaler v1.2.5 — v127 Deep Audit: Codebase Findings, Live Matrix, Fix & Test Plan

Round: v127-audit (2026-09-07). Scope per charter: deep codebase research of the
running image (`llm-scaler-exp:v1.2.5` = v52 final, image a522bf15be2b), all
issue classes (non-optimized pipelines, memory handling, token generation,
long-context degradation, looping), all serve families (nospec / MTP / DFlash2
spec × `fp8_e4m3` / `turboquant_4bit_nc`), optimal env/params, bugs, fixes &
improvement plan with a real-scenario test plan (single + multi client, 128k
and 262k).

Companion artifacts in this dir: `master127.sh` (matrix driver), `run_lane127.sh`
(lane runner), `p1_disc.sh` (P-1 discriminator), `logkeep.sh` (log preservation).
Baselines live in `vllm/patches/diagnostics/v125-audit/REPORT.md` (D1–D10) and
`vllm/patches/v125-fixes/PLAN.md` (v53 round: L2 conviction, D7 warmup verdict,
16k-dip observation). Host bench root: `/root/build/bench127/`.

**Sibling audit merged into this plan**:
`vllm/patches/diagnostics/latest-image-audit/REPORT.md` (2026-09-07, image
inventory + N01–N16 backlog + validation staging; its evidence set includes
`live-inventory.txt`, `latest-image-packages.txt`, `torch_utils.py` dtype
mapping, `bench127-partial.log`). Its findings, corrections and backlog are
merged below (§3.9 corrections register, §4.5 dtype/memory model, §5 unified
backlog, §7 protocol). Evidence labels used here follow its convention:
**Fresh** (measured this round on the live host), **Historical** (archived),
**Code** (inspected path, impact unmeasured), **Proposed**.

**Baseline decision (adopted)**: v1.2.5 (image a522bf15be2b) stays the
comparison and production baseline. v1.2.6t2 (006791e5f95c) is an
observability/guarded-feature build — its flash_attn overlay adds a ragged
**front-padding route default-ON** plus optional routing counters
(`VLLM_V125_P1`), the proposer adds ctx-width calc default-DISABLED
(`VLLM_V53_L2_UNSTABLE`), TQ adds split-tier logging; no demonstrated perf
upgrade. Deep warmup (`dt_warmup_v53.py`) is host-side, not baked. Priorities
adopted from it: (1) trustworthy measurement + loop diagnosis, (2)
scheduler-owned speculative widths (never proposer-only narrowing — v53
recorded a worker wedge), (3) tiled compressed-KV attention, (4)
memory/admission accounting.

**Three more companion plans merged** — two under
`vllm/patches/diagnostics/long-context-plan/`, one under
`vllm/patches/diagnostics/bug-remediation-plan/`:
- `PLAN.md` ("Fixing long-context degradation: nonspec, MTP and DFlash2",
  2026-09-07; LC-0..LC-6 plan + M0–M6 milestones + ms/token cost model;
  evidence copies of `dflash.py`/`dflash2.py`).
- `LATENCY_THROUGHPUT_PLAN.md` ("Long-context latency and maximum token
  throughput plan", 2026-09-07; three-profile operating points, workstreams
  1-8, Stages A–D benchmark design, speculation decision rule; evidence
  `latency-throughput-host-snapshot.txt` captured the bench127 df7_tq4 deep
  cells + TTFT 88.7/128.7 s live).
- `../bug-remediation-plan/PLAN.md` ("Bug remediation plan: serving,
  speculation, cache/state and failure handling", 2026-09-07; BUG-01..12
  register + R1–R8 review units; evidence includes captured
  `async_utils.py`/`async_scheduler.py` and a CPU-only reproduction of
  BUG-01 with result JSON). Its two new code findings are verified this
  round and recorded as C11/N17/N18; the rest map onto existing items
  (§5 trailing note).
Their corrections are applied in place — C5 rewritten (DSPARK knob inert on
dflash2, grep-verified), C4 memory math fixed, C1 discriminator confound
documented, C9 items 9–11 — and their items map onto the N-backlog (§5
trailing note; §6 profiles; §7 cost model/milestones).

---

## 0. Executive summary

v1.2.5 audited on the four user baseline lanes @0.9/262144 (bench127, fresh
single-draw matrix). Verdict on evidence so far:

1. **Health/stability is now the top finding, not perf**: the PROD lane
   (mtp4_tq4) died with a level_zero DEVICE_LOST during warmup — p7 slow
   (0.7 tok/s) → TQ kernel JITs → ccs engine reset on BOTH GPUs → RPC
   timeout → EngineDead (§4.9, N04). Second event of its class in lineage
   and the first on the exact prod config; host shows a repeating
   engine-reset pattern in the kernel journal (4 events/8 h). Rebooted per
   §8/BUG-10; the single relaunch then completed ALL phases cleanly with
   **all six battery SHAs bit-exact to the prod refs** — attempt-1
   classified driver-state-dependent, not workload-deterministic; N04 stays
   P0 (GuC firmware drift + xe/level-zero escalation recommended).
2. **P-1 ROOT-CAUSED at axis level (C1 closed by the P-1a/P-1b axis-split)**
   (C1): mtp4_fp8 @0.9 SLOW (16.0/14.6/12.6 vs 42/34/22 @0.85 refs) while
   mtp4_tq4 @0.9 healthy (60.6/37.8/30.7/20.9). Dose-response on pool BYTE
   footprint (margin held at 0.9): 599k→FAST 32.0/35.4/27.2/22.3,
   708k→SLOW 16, 1,120k→PATHOLOGICAL 0.4 — **driver = KV pool bytes ÷
   device-memory headroom (XPU/L0 allocator pressure under spec transients);
   GMU margin exonerated**. Ops mitigation defined (cap pool to 0.85-class);
   override-validation defect filed as N19 (units ≈ 956.73 tok/unit, honored
   unvalidated vs the 12.3 GiB profile — P-1a booted 19.5 GiB/rank).
3. **D10 root-caused to dflash2-drafter acceptance collapse at 136k, not a
   device cliff** (C2, §4.7): df7 acceptance 2.1-2.5 @136k vs 5.5-8.0
   @262k-class; MTP on the SAME prompts+KV shows no collapse (11.7 @136k =
   2.8× df7) — drafter-specific, dtype-independent; DFLASH_STALL and the TQ
   ladder both exonerated. Residual discriminator: df7 seed-swap boot.
4. **Decode at depth is poor everywhere except the prod lane's 136k cell**
   (spec lanes 4.2-6.8 @136k and 4.7-6.7 @262k; mtp4_tq4 11.7/5.6; nospec
   refs 34.4→22.4 @32k) — the tiled compressed-KV / continuation-kernel
   workstream (N06/N07, LC-1/LC-2) remains the structural fix; 262k-class
   TTFT 1.9-3.6 min on every lane. Post-phase: 16k is a stable ~35-42 band
   (P1), and **no cold-vs-warm asymmetry on the prod config** (P2, 10.3 vs
   10.9 on two distinct 128k prompts) — the 4.2→7.3 asymmetry is D10-tied,
   not a first-request property.
5. **D8 conc2@65k order-lottery reproduced on ALL FOUR lanes with ORDER
   FLIP** (C3, 2.6-3.2× asym either direction) — PrefillAdder head-of-line,
   not dtype/drafter.
6. Code findings verified this round: BUG-01 (merge_async_iterators
   multi-input cleanup, CPU-reproduced) and BUG-02 (unconditional
   DFLASH2_EMIT_K read on MTP/nospec lanes) → N17/N18; C5 corrected
   (DSPARK inert on dflash2); **loop8 `"!!!!"`×128 periodic tails are
   KV-dtype-correlated (tq4 6/8 vs fp8 1/8), not drafter-caused** (§4.4) —
   quantization-induced repetition is the new N02 primary suspect.
7. All fixes route through the unified backlog (§5, N01-N19) sequenced by
   R1-R8; measurement integrity (N01) and failure preservation (N04) gate
   everything else.
8. **R-LC directive adopted as the governing performance requirement (§7A)**:
   only a MINOR decode-tps drop 64k→262k. Deep research (code + web) verdict:
   the user's suspects are second-order — the checkpoint is a documented
   hybrid (16 full-attn + 48 linear-attn of 64 layers; SSM/linear layers
   O(1)), and decode at depth runs 2-6× ABOVE the memory floor (floors
   themselves only fall −20% 64k→262k: 41.5→33.4 tps nospec, more with
   spec), while PCIe binds prefill/TTFT, not decode. The falloff is fixable
   in software: W1-W7 workstreams, gates G1-G5 (e.g. 262k ≥0.5×2k),
   sequencing §7A.7. No hardware change implied.

Post-matrix discriminators COMPLETE: **D10 order-swap** (df7_tq4 deep cells
reversed: 221000-first = 6.7 tps / acceptance 5.5-8.0, 115000-second = 4.2 /
acceptance 2.1-2.5 — the collapse follows the PROMPT, not position/state;
even a full prefix-cache-hit cell stayed 4.2) ⇒ C2 closed at classification
level, residual = length-class vs prompt-text mechanism (cheap decisive cells
designed in C2). **P-1 axis-split** (P-1a/P-1b) ⇒ C1 closed, see item 2.

## 1. What v1.2.5 runs (unchanged from v125 §1, verified this round)

Image `llm-scaler-exp:v1.2.5` = v52 final bake (fp8 MQ two-stage kernel, TQ
v52 attention, F8/F9 guards, async abort fixes, v51 warmup p1–p9). The four
user baseline lanes all serve `--gpu-memory-utilization 0.9
--max-model-len 262144`, `FULL_DECODE_ONLY` whole-step graphs,
`--async-scheduling`, TP=2 on 2× Arc Pro B70, model `qwen3.8-27b-fp8`:

| lane | kv dtype | spec |
|---|---|---|
| df7_fp8 | fp8_e4m3 | dflash k7 (`/models/dflash2`) |
| df7_tq4 | turboquant_4bit_nc | dflash k7 |
| mtp4_fp8 | fp8_e4m3 | mtp k4 |
| mtp4_tq4 | turboquant_4bit_nc | mtp k4 (prod lane) |

Boot observation (this round): **df7_fp8 KV pool = 373,507 tokens** — the
dflash2 drafter (a second full model) consumes roughly half the fp8 KV capacity
headroom vs mtp4_fp8's 705–708k (v125 §3.1). Capacity consequence: at
maxlen 262144 the df7_fp8 lane can hold **~1.4 max-len requests** — deep
multi-client on this lane is capacity-bound before it is throughput-bound.

## 2. Methodology — bench127 matrix

Per lane (run_lane127.sh): boot (+0.85/245760 fallback) → warmup v51 (p1–p9) →
battery (dt_probe2, 6 SHA cells + acceptance) → ctxscan 2k–65k (varied fillers)
→ **deep 128k + 262k-class** (dt_ctxscan2 sizes 115000/221000 words ≈
136.2k/261.7k ptok, 256 out, temp 0) → **loop8 think-trap scan** → conc
(8@2k, 4 mixed 2k/16k, 2@65k) → health tail (STALLS / v52m strikes / F8 /
jit_monitor / ERRS / DEVICE_LOST).

Post phase (master127.sh, on the live mtp4_tq4 container): P1 = 16k N=3 with
DISTINCT prompts (classifies the v53 46.6→38.5 dip: noise vs systematic); P2 =
D7 cold-cold deep pair (two DIFFERENT 128k prompts back-to-back — first-vs-
second decode_tps isolates per-prompt vs first-request/JIT-pool effects);
P3 = xpu-smi memory snapshot.

Baseline deltas are computed against v125 numbers (same harness, same image
family; deep 227k cells there were nwords 205000 ≈ 242.8k ptok — the v127 deep
cell is 221000 ≈ 261.7k ptok, i.e. the true 262k envelope edge).

<!-- TBD-matrix: lane tables + verdicts -->

---

## 3. Code-level findings (new this round; file refs = installed tree)

### C1 — P-1 (fp8 × spec × GMU 0.9 per-step penalty): three suspects narrowed to one axis

Carried from v125 D1: mtp4_fp8 @0.9 decodes 1.7–2.6× slower at ≤65k than @0.85
(same everything else); penalty is per-step, vanishes ≥128k; acceptance healthy;
deterministic. New eliminations from source this round:

1. **Attention route change at pool size — EXONERATED.** The v51 fp8 MQ gate
   (`flash_attn_v51.py:1337-1356`) keys only on query-shape metadata
   (`max_seqlen_q`, `num_actual_tokens == max_seqlen_q × batch`, causal,
   head_size, descale) — none of which depends on pool size for a fixed bs=1
   workload. Batch composition cannot change with pool size at bs=1.
2. **Kernel grid/scratch pool-dependence — EXONERATED.** `triton_fp8_mq.py`:
   stage-1 grid = (B, Hq, SPLITS=const), stage-2 grid = (B, Hq); scratch `mid`
   is grow-only keyed on (pool tensor identity, B, Hq, SPLITS, Q_BLOCK, D)
   (`triton_fp8_mq.py:357-388`). No per-step pool-scaled work exists in the
   kernel path.
3. **Wider block-id gather spread — EXONERATED by the tq4 control, with a
   byte-vs-token refinement (below).** tq4 pools are LARGER in TOKENS
   (1.3M ≈ 81k blocks vs fp8's 705k ≈ 44k) and tq4+spec@0.9 is healthy; if
   block-id address spread were the cost, tq4 would be worse. (Note: this
   exonerates token-count-scaled costs specifically — pool BYTE footprint
   is a different axis, see the P-1a dose-response.)

**RESOLVED this round — P-1a/P-1b axis-split executed (`p1_disc.sh`, `p1b.sh`;
reports under `bench127/p1/`, `bench127/p1b/`).** The planned
confound analysis was invalidated by a premise discovery, which turned into
the decisive dose-response:

- **Unit discovery:** `num_gpu_blocks_override` units in this fork are
  ≈ **956.73 tokens/unit**, NOT 512-token blocks — natural 0.9 = 740 units ⇔
  707,980 tokens; override 1171 ⇔ **1,120,330 tokens** (1.58× GROWTH, not a
  shrink); override 626 ⇔ 598,912 tokens (0.85-class). Boot logs confirm each.
- **P-1a (0.9 + override 1171 → pool 1,120,330):** PATHOLOGICAL — warmup
  p2-conc 0.2–0.3 tok/s, `dt_warmup` died phase 3, ctxscan 2k = **0.4 tps**
  (682 s; verified uncontaminated — no warmup process alive). The override was
  honored WITHOUT validation against `Available KV cache memory: 12.3 GiB`
  (1,120,330 tok × 18.65 KB/tok/rank ≈ 19.5 GiB/rank requested) — the pool
  consumed essentially all post-weights headroom ⇒ near-OOM allocator thrash.
- **P-1b (0.9 + override 626 → pool 598,912):** **FAST** — WARMUP_OK full
  p1–p9, ctxscan 2k/16k/32k/65k = **32.0/35.4/27.2/22.3**. Margin axis (GMU
  0.9) held constant, pool shrank ⇒ margin alone does NOT produce SLOW.
  Host py-spy during ctxscan: EngineCore idle in `shm_broadcast.dequeue`
  (normal step path — no pathological spin), PYSPY_ON_HOST technique proven.

**Dose-response (fp8 KV, mtp4 spec, GMU 0.9 unless noted):**

| Pool tokens | Pool bytes/rank | Headroom after weights | 2k tps | Class |
|---|---|---|---|---|
| 598,912 (0.85 natural; P-1b forced @0.9) | ~10.4 GiB | ~8 GiB | 32–42 | FAST |
| 707,980 (0.9 natural) | ~12.3 GiB | ~6 GiB | 16 | SLOW |
| 1,120,330 (0.9 + override 1171) | ~19.5 GiB (!) | ~0 | 0.4 | PATHOLOGICAL |

Monotone in pool BYTE footprint at fixed margin; reconciles the tq4 control
(1.3M TOKENS but 4-bit ≈ small BYTES → healthy). **Verdict: P-1 driver = KV
pool byte footprint ÷ device-memory headroom (torch XPU / L0 caching-allocator
pressure under spec's per-step transients), amplified by fp8's 2× token
density per byte budget. GMU margin per se is exonerated** (P-1b FAST at 0.9).
Residual mechanistic confirmation (allocator slow-path stack, was P-1c) is
optional — axis conviction is complete without it.

**Ops mitigation (immediate, no code):** for fp8-KV × spec boots, cap the pool
to the 0.85-class size — either run GMU 0.85 (natural 599k) or GMU 0.9 +
`--num-gpu-blocks-override 626` (proven FAST above). Cost: ~15% pool tokens;
benefit: 1.7–2.6× decode at ≤65k.

**New config-integrity finding (folded as N19):** `num_gpu_blocks_override`
is honored without any validation vs profiled available KV memory — it can
request ~1.6× the budget, silently degrade to near-OOM thrash (P-1a), and
report the inflated pool as healthy. Upstream parity check + clamp/warn
belong in `kv_cache_utils.py` override handling.

Status: **P-1 ROOT-CAUSED at axis level (C1 closed)** — pool-bytes/headroom
dose-response 599k→FAST, 708k→SLOW, 1,120k→PATHOLOGICAL; margin exonerated;
mitigation defined; N19 filed.

### C2 — D10 (df7+tq4 128k non-monotonic cliff): tier ladder exonerated by source

v125 D10 measured df7_tq4 decode 14.3 @65k → **4.0-4.2 @128k** → 7.3 @227k
(deeper = faster). Suspects were the TQ splits tier ladder, dflash draft
window vs 2048-blocks, or a capture bucket gap. Source ruling:

- Under `FULL_DECODE_ONLY` the TQ grid is a **compile-time constant**
  (`turboquant_attn_v52.py:515-545`): maxlen 262144 ⇒ tier=8 ⇒
  `graph_kv_splits = min(base×8, 256) = 256` at **every** depth — the ladder
  cannot produce a 128k-specific step. Graph capture buckets are batch-size
  keyed (no seq-len buckets) — bucket-gap also structurally excluded.
- **Live suspect: DFLASH_STALL firings.** `dflash.py:157-171` logs
  `DFLASH_STALL propose() took %.3fs` for host-side drafter walls >150 ms
  (peer-late oneCCL windows; the >640 ms tail trips the GuC watchdog). v125
  df7_tq4 lane logged STALLS=17-20. Each 0.2-0.5 s stall inside a decode
  phase directly inflates wall-clock decode_tps; the depth-conditioned
  trigger (why ~136k and not ~243k) remains open — candidate: draft-KV pool
  pressure/watermark interactions at the 373k-token (df7_fp8) / draft-share
  boundary, or chunked-prefill tail shapes unique to the 115000-word cell.
- bench127 evidence incoming: df7_tq4 deep cells + per-lane STALLS count +
  `b_df7_tq4.log` stall-timestamp × decode-phase correlation. Caveat (LC
  plan, adopted): lane-total STALLS ≠ in-decode proof — stalls landing in
  prefill/warmup/conc phases inflate the count without touching deep
  decode_tps; only timestamp × phase correlation convicts. Fresh so far:
  STALLS=15 with deep 128k 4.2 / 262k-class 6.7; correlation pending (logs
  preserved via docker cp into `/root/build/bench127/logs/`).
- **CORRELATION VERDICT (Fresh, preserved log 17:00:04–17:15:34 covers the
  full deep window): DFLASH_STALL EXONERATED for D10.** All 11 preserved
  stall firings (0.17–6.1 s propose walls) land in the WARMUP phase
  (17:02:47–17:05:47, draft-model chunk-prefill first-touch); the deep
  window (17:07:35–17:12:52) has ZERO (the 15-vs-11 delta = 4 firings in
  loop8/conc, outside the preserved range). Decode-window WARNINGs are only
  per-chunk prefill bookkeeping (DEF-PP/PRE-COPY pairs, both ranks
  lockstep, sched=6144/chunk) + one-shot DISCARD-GAP tripwires at chunk
  boundaries; engine-reported gen tps 4.0–5.0 confirms the 4.2 measurement
  is real step time × acceptance.
- **NEW LEAD (Fresh): acceptance-length inversion between the deep cells.**
  10s-window SpecDecoding metrics: 136k cell mean acceptance length
  ≈2.1–2.5 (draft rate 16–21%) vs 262k cell ≈5.5–8.0 (windows at 100%).
  DFlash2 verify cost is width-fixed (native 8-row block always runs, C5),
  so tps ≈ acceptance/step_time — the 4.2-vs-6.7 inversion is substantially
  an acceptance collapse on the 115000-word prompt class. BUT mtp4_tq4 on
  the SAME prompts shows the opposite depth trend (Historical 11.7 → 6.1,
  slower deeper) ⇒ prompt-entropy alone cannot flip the sign; a
  drafter×depth interaction remains. Discriminators already in flight:
  lane-4 mtp4_tq4 deep cells (same prompts, MTP drafter) + post-phase P2
  (different-text 134.9k/137.3k) ; follow-up (cheap): df7_tq4 boot with
  seeds swapped at 115000/221000 to test prompt-conditioning directly.

- **ORDER-SWAP VERDICT (Fresh, `d10swap/`)**: df7_tq4 boot, deep cells
  REVERSED (221000 first, 115000 second; prompts bit-identical via the same
  dt_ctxscan2.py): 262k-class = 6.7 decode / acceptance 5.5-8.0 (windows at
  100%) at FIRST position; 136k = 4.2 decode / acceptance 2.1-2.5 at SECOND
  position — **bit-exact per-prompt reproduction of lane 2's original order
  (6.7/4.2 both times, positions swapped)**. D10 is deterministic and
  PROMPT-TIED: order, first-decode state, JIT, and pool state are excluded
  (the 136k cell even landed a full prefix-cache hit — TTFT 1.9 s — and
  still decoded 4.2). D10 = a dflash2-drafter × prompt(-class) interaction:
  the drafter's drafts are rejected ~3× more on the 136,191-ptok prompt than
  on the 261,757-ptok prompt, reproducibly.

Status: **D10 ATTRIBUTED (C2 closed at classification level)** —
DFLASH_STALL exonerated (timestamp×phase), ladder/bucket exonerated
(source), position/state exonerated (order-swap), MTP shows no collapse on
the same prompts+KV (lane 4). Residual open question (downgraded to a
focused N-item): WHY the dflash2 drafter's acceptance inverts between the
two prompt classes — length/bucket-boundary mechanism vs prompt-text
mechanism. Cheap decisive next cells (one boot): same-length different-seed
136k prompt + same-seed different-length prompt; if collapse follows
length class → draft-KV/chunk-boundary conditioning; if it follows the
specific text → selector/entropy interaction. No device-side hunting before
that lands.

### C3 — D8 (conc2@65k starvation asymmetry): root = PrefillAdder head-of-line chunk monopoly

Mechanism (now fully attributed): with 2 concurrent ~65k prefills and MNBT
8192, the v1 scheduler's prefill adder grants the **whole per-step token
budget to the first waiting request** — request A monopolizes ~170 s-class
prefill, finishes, and decodes at co-decode speed while request B still
prefills; B's decode interleaves at ~1.4 tok/s-class under A+B mixed steps.
Measured asymmetries: nospec 17.6 vs 5.6; fp8-nospec 4.8 vs 29.0 (order
flipped). Not a correctness bug; a p99/fairness defect.

**Fix design (L9, scheduler, no numerics risk):** round-robin prefill chunking
— when >1 chunked-prefill request is waiting, split the per-step budget
(e.g. MNBT/2 each, floor 1024) instead of head-of-line fill. Touch point:
`v1/core/sched/scheduler.py` prefill-adder loop (budget application), gated
behind `VLLM_V1_PREFILL_ROUND_ROBIN=1` default-off. Effects: TTFT p95 at
matched conc improves (second client's prefill no longer serialized); decode
phase overlap equalizes per-client tps (target ratio ≤1.5 vs today's 2-6×);
single-client throughput unchanged (monopoly case identical when only one
prefill is resident).

### C4 — D3 (TQ continuation prefill O(n²)): only viable fix is a fused paged kernel

Structure confirmed at `turboquant_attn_v52.py:1486-1652`: per continuation
chunk, per full-attn layer: `_tq_full_dequant_kv[grid=(alloc_len, Hk)]` over
the **entire cached prefix** → Pi-rotation matmul → materialized
`k_full/v_full` (seq_len × Hk × D fp16, ~2 GiB transient at 262k) → FA varlen.
At 262k/MNBT 8192 ≈ 32 chunks ⇒ ~17× dequant amplification.

A persistent dequantized fp16 prefix cache is **infeasible**: 16 full-attn
layers × 262k tokens × 2 KV-heads/rank (TP2) × D 256 × 2 B × (K+V) ≈
**8 GiB/rank** (corrected by the LC plan; an earlier 17 GiB figure here used
the pre-TP head count) — the per-layer
shared WorkspaceManager exists precisely because per-layer persistence does
not fit. The viable fix is the L1-class **fused paged TQ continuation
kernel** (dequant-on-chip while streaming attention, same pattern as the TQ
decode kernel) — eliminates the gmem dequant materialization AND the O(n²)
re-reads. Stays ranked below L1/L4 on impact (measured verdict: TQ deep
prefill is still 1.5× faster than fp8's; the dominant deep costs are
decode-side), but it also removes the ~2 GiB transient spike (memory-margin
hygiene at GMU 0.9 — see C6). Scope note (LC plan): the fused kernel kills
the quadratic dequant-BYTE traffic and the materialization; attention
against the cached prefix remains O(n²) per chunk by definition — the win
is bytes and transients, not FLOPs.

### C5 — DFlash2 width policy: knob machinery is INERT on the dflash2 path (corrected)

**Correction this round (convicted by source + LC plan; an earlier draft of
this finding claimed "a KNOB, not new code" — wrong for the serving path):**

- DFlash2 is a **native 8-row block drafter** — one anchor row + 7 masked
  positions in a single forward, walked by the tree selector
  (`dflash2.py:234` propose, `dflash2.py:285` `_greedy_sample` override) —
  **not** k7 autoregressive draft forwards (also corrects C7's old aside).
  Emitted-width reduction saves target VERIFY work; the native 8-row draft
  pass always runs.
- `DSPARK_ADAPTIVE_BLOCK` truncation gates on `self._draft_confidence is not
  None` (`dflash.py:174`), and `_draft_confidence` is populated **only in
  the base DFlash Markov path** (`dflash.py:360`) — dflash2's own
  `_greedy_sample` never populates it ⇒ **the knob can never fire on
  dflash2**. Verified by grep on `long-context-plan/evidence/dflash{,2}.py`.
- Internal draft width is checkpoint-tied (block = 8); narrowing emitted
  width is feasible, narrowing the internal block is not.
- `VLLM_XPU_DFLASH_TP1` stays 0 (v50 corruption history).
- `DFLASH2_EMIT_K` exists (default-inert, range 1..num_spec_tokens) but both
  tested settings are convicted (EMIT_K=4 v42-refused, EMIT_K=5 cargame
  stall) — not a promotion path as-is.

Measured ground (unchanged) for wanting width-down at depth: acceptance
decays with context (mtp4: 0.73/0.71/0.14/0.02 @65k positions 0-3; dflash k7
battery acceptance 0.336 @2k-class this round vs mtp4's 0.572-0.613) while
every draft/verify row costs a full-depth KV scan — fixed-k is net-negative
≥65k (v125 D5). **What real width-down requires (LC-4, folded under N03):**
wire actual selector confidence (the dflash2 walk already computes edge
scores) or a measured-acceptance ctx→k policy into the **scheduler-owned**
width protocol (N03/L2b — never proposer-only; v53 wedge), plus full
determinism/battery/crash gates (v50/v53 history: this surface bit twice).
Speculation decision rule (adopted, §7): enable spec per
(mode, KV, ctx-bucket, resident-batch, output-workload) only when
step-time/emitted-token beats nonspec by >10% margin.

### C6 — Memory handling inventory (this round's findings)

| item | size class | note |
|---|---|---|
| df7_fp8 KV pool | **373,507 tok** (~1.4 max-len reqs) | dflash2 drafter halves fp8 capacity; df7 deep-concurrency is capacity-bound |
| mtp4_fp8 pool | 705-708k tok @0.9 | v125; margin axis under P-1 study |
| tq4 pool | ~1.3M tok | best capacity headroom (4.9 max-len reqs) |
| `_DEQUANT_BUFS` fallback + WorkspaceManager bufs | grow-only, ~2 GiB @262k/rank | per-device persistent; sized by deepest continuation seen |
| `k_full/v_full` transients | ~2 GiB per chunk-layer call @262k | cached-allocator reuse; spike risk at GMU 0.9 (C4 fix removes) |
| fp8 MQ `mid` scratch | B×Hq×32×(q+1)×(D+1) fp32 | tiny; grow-only |
| prefix-cache retention | 7-17% residue observed | shrinks effective capacity; v52 eviction posture unchanged |
| MTP draft KV | ~23% of request capacity | inherited dtype from target (v125 D4 correction) |

Ops consequence: **df7_fp8 cannot serve 2 concurrent 262k requests at all**
(pool < 2×262144) — admission serializes (TTFT ladder, PERF_TUNING
multi-stream table); if dflash deep-concurrency ever matters, tq4 is the only
df7 lane with pool headroom (or maxlen must drop).

### C7 — Host-side token-generation chain (recap, unchanged)

#16/#17: spec exposes a ~54 ms/step host chain (acceptance of step N gates
scheduling of N+1): eager MTP drafter 7-20 ms, GDN metadata ~4.4 ms, block
commit H2D ~2 ms, ~30 ms IPC/wakeup latency (EngineCore idle in
shm_broadcast). Real fixes are P2 (worker-resident acceptance loop) /
P3 (draft graph capture, blocked upstream on #11-class wedges). Interim
routing: high-concurrency traffic → nospec (v125 §5). DFlash's drafter is a
separate model with its own propose() (host walls logged by DFLASH_STALL,
C2) — same class of host exposure; its draft is a single 8-row block forward
+ selector walk (C5 correction), so the wall is propose/select host time,
not per-row forwarding.

### C8 — Looping/degenerate-output ledger (cross-doc, closed classes)

- #20: "thinking loops" on fp8 = budget-limited long thinking (not dtype);
  mitigation max_tokens / `enable_thinking:false`. Verified all 4 dtypes.
- #13: ≥32k repetitive-filler degeneracy = distribution collapse on
  low-entropy filler (probe artifact); correctness gates must use natural
  text — loop8/ctxscan cells measure THROUGHPUT only.
- #12: graphs+MTP-k4 temp-0 corruption — k4+ is user-selectable with the
  standing caveat (prod = mtp4_tq4 ships with it; battery SHAs + acceptance
  are the regression gate; any k4 lane keeps the #12 risk on low-margin
  prompts at SHORT ctx — deep ctx masks it).
- dflash7 greedy loop-risk ledger (v125 Phase F: 4/4 greedy loops under
  sustained load class) — df7 stays eval/experiment posture.
- loop8 think-trap scan now runs per-lane in this matrix (fresh N=4 evidence
  incoming).

<!-- TBD-loop8 -->

### C9 — Corrections register (sibling-audit §3.3 merged; supersedes selected v125/earlier takes)

1. **Adaptive spec width is an end-to-end protocol change** — the v53 trial
   narrowed only proposer output and wedged a worker (silent 300 s RPC
   timeout; async placeholder width fixed at boot). Accounting/graph mismatch
   is the code-backed explanation; exact blocked device/collective not
   isolated. `VLLM_V53_L2_UNSTABLE` stays closed. Any width work is
   scheduler-owned (N03/L2b); assess upstream Dynamic SD (MRv1 = piecewise
   only; full graphs need MRv2) before inventing an API.
2. **Ragged routing is a latent path, not the GMU-cliff cause** — t2 route
   counters observed 2000 fixed-width MTP/fp8 verify calls, ALL uniform
   (Fresh). Do not project ragged-padding speedups onto the measured P-1
   lanes; L4/N11 stands on its own (conc-at-depth cliff) with hit-rate
   measurement FIRST.
3. **Deep warmup did NOT solve deep decode** (v53 L7 verdict re-confirmed):
   different-text deep results reproduce slow values after extended warmup;
   same-prefix TTFT gains show cache reuse only. Never promise cold-262k
   TTFT <10 s from warmup (N13 A/B before keeping warmup phases).
4. **TQ host tier ladder excluded ≠ DFlash depth anomaly closed** — matches
   C2: ladder/bucket structurally exonerated (constant 256 grid), remaining
   candidates: device-side re-partition, acceptance, draft windows,
   addresses, request history. v127 Fresh data: df7_tq4 128k = 4.2
   (bit-reproduces v125's 4.2) vs 262k-class 6.7 / 227k 7.3 → **D10 is
   deterministic and boot-stable**; anomaly localization band (120k–144k vs
   240k–262k) is the N-item follow-up.
5. **Retained per-layer dequantized prefix is NOT a cheap fix** (C4 math:
   shared scratch is per-layer-overwrite by design) — tiled in-attention
   dequant (N06/L1+L6) is the fix class.
6. **Health posture not universally clean** — v125 restore logged DEVICE_LOST
   during warmup (GuC-reset class, distinct from the L2 silent wedge); later
   zero-error runs don't exclude intermittency (N04 forensics).
7. **Draft dtype inheritance is real** (MTP inherits cache_config; dflash
   defaults to target dtype with override) — and "K+1 full scans" is a
   costing heuristic, not a measured bandwidth equation (fused multi-query
   verify can reuse KV tiles). v125 D4/D5 stand with that caveat.
8. **Workload-boundary corrections**: mtp4_tq4 @16k (46.6) BEATS nospec+fp8
   (33.5) — the v125 §5 "nospec+fp8 for 16k-65k" row overstated the range;
   crossover is prompt-dependent between 16k and 32k (§6 corrected). TQ pool
   1.3M vs fp8 0.705M = **1.84× capacity ratio** (4.9 was concurrent
   max-len request count, not an improvement ratio).
9. **DFlash2 architecture** (LC plan, grep-verified): native 8-row block
   drafter (anchor + 7 masks) with selector walk — not k autoregressive
   forwards; draft cost ≈ one block forward + walk, and emitted-width
   narrowing saves only target verify work (the native pass always runs).
   Corrects C7's old "7 draft forwards" aside and any per-row costing.
10. **`DSPARK_ADAPTIVE_BLOCK` is inert on dflash2** (C5): confidence is
    populated only in the base DFlash Markov `_greedy_sample`
    (dflash.py:360); dflash2's override never populates it and the
    truncation gate (dflash.py:174) requires it. Width work = LC-4 under
    N03 — real code, not a knob.
11. **Override/counters are insufficient attribution instruments**: the
    `num_gpu_blocks_override` discriminator moves pool AND margin together
    (C1 confound — ballast/py-spy are the decouplers); lane-total STALLS
    need timestamp × phase correlation before conviction (C2). Also
    (latency-throughput plan): bench127's 221000-word deep cell is 261,757
    ptok + 256 out = **262,013 total — NOT the exact 262,144 boundary**
    (exact-boundary cells are §7 Stage C; log the arithmetic).

### C10 — KV dtype coverage & memory model (from sibling audit; enum has 15 values)

Support gates are separate: argument spelling / tensor representation /
backend routing / boot success / correct generation. The flash backend's
advertised list (auto/fp16/bf16) disagrees with exercised fp8 routes — enum
and class lists are not capability proofs (N15: executable startup
capability check).

| dtype | status here | action |
|---|---|---|
| auto (→fp16 exec) | uncompressed-KV quality reference | keep as reference arm |
| fp8_e4m3 | exercised baseline | P-1 root-cause; scales audit |
| turboquant_4bit_nc | exercised baseline | L1 kernel ceiling |
| turboquant_k8v4 | implemented (fp8 keys/4-bit vals) | intermediate candidate — needs Stage-A validation |
| turboquant_k3v4_nc / 3bit_nc | implemented (3-bit) | aggressive-experimental; mandatory retrieval/reasoning differential gates |
| fp8_e5m2 | checkpoint-reject class (#19) | never bypass the guard only |
| fp8_inc / fp8 (generic) | mapping exists, byte-storage differences | confirm format/scales vs e4m3 before counting as distinct |
| int8_per_token_head / fp8_per_token_head | enum/mapping only | validate scale layout + kernels before any serve |
| fp8_ds_mla / nvfp4 | enum/mapping only | inapplicable (GQA target) / vendor-specific — classify unsupported |
| float16 / bfloat16 | mapping exists, no current matrix | explicit boot smoke (version-specific) |

Packed full-attn KV bytes/token/rank (D=256, Hkv=2/rank, 16 layers):
fp16 32 KiB · **fp8 16 KiB** · k8v4 12.125 · **tq4nc 8.1875** · k3v4 7.1875 ·
3bit 6.1875 — derived lower components only; add scales/padding/draft
pools/GDN state/graphs/scratch/comms/headroom. At 262k/rank: 8 GiB (fp16) →
1.55 GiB (3bit). Continuation-prefill at 262k ≈ 512 MiB cached K+V + 512 MiB
concat + 256 MiB rotated K per layer-call class; O(n²) re-dequant across 32
chunks ≈ 4.06M cached-token visits/layer ≈ 15.5× final length (C4/N06).
`--max-num-seqs 64` is a scheduling ceiling, not capacity; df7_fp8 pool
(373,507 tok) cannot hold two unrelated 262k requests — admission must
reserve prompt+output budget and report queueing (N09).

### C11 — Serving-layer defects (bug-remediation plan; both new code claims verified this round)

- **BUG-01 (reproduced, P1 → N17)**: `merge_async_iterators` multi-input
  cleanup does not wait for children — `async_utils.py:318-320` calls
  `f.cancel()` then immediately `await it.aclose()` under
  `suppress(BaseException)`; cancel schedules, it does not complete. CPU
  AST-extraction repro (`bug-remediation-plan/evidence/
  repro_iterator_cleanup.py`) proves cleanup incomplete at close-return,
  complete later (`repro-result.json`: cleanup_completed_when_merge_close_
  returns=false). Note: this is the SAME function v52's F9 patched — F9
  added the single-iterator fast-path aclose (:294-297); the multi-input
  path kept the ordering defect. Client-disconnect leak class (v50 crash-3
  adjacency). Fix order: cancel all outstanding anext tasks → await their
  outcomes → aclose ALL children (including ones whose task raised) →
  supervised cleanup task if serving needs a timeout; never blanket-suppress.
- **BUG-02 (code-confirmed, P1 → N18)**: `async_scheduler.py:27-37` (v42c
  comment "honor DFLASH2_EMIT_K on the async path") reads the env
  unconditionally — a leftover value silently changes placeholder width for
  MTP (or fails validation on nospec). Fix = one validated config resolving
  (method, native width, emitted width, scheduled width) BEFORE worker
  launch; an irrelevant dflash setting must be a loud error/warning, never
  a silent MTP behavior change.
- **BUG-04 (P0, design arm under N02)**: zero-emissions/sentinel clamps
  (F8/v52 guards) are containment, not diagnosis — required: explicit
  per-step output status (valid-decode / prefill-no-emission /
  grammar-deferred / cancelled-stale / invalid), bounded first-invalid-row
  diagnostics across both ranks, validity masks so substituted tokens are
  never committed, exactly-once terminal delivery on every path.
- **BUG-05 classification discipline (N02)**: repetitive output and budget
  exhaustion must NOT share the label LOOP — loop8 already separates
  `think_trapped` vs `loop_reason_char`; keep them separate in all
  reporting; replay workflow = teacher-forced prefix through
  nonspec/MTP/dflash at identical weights/KV/SSM, locate first divergent
  layer/logit/state, then free-run controls (auto-KV, fp32-SSM).
- Remaining BUG items map: BUG-03=N03, BUG-06=C5 (fix-first: reject/warn
  the unsupported DSPARK-on-dflash2 combination at startup diagnostics),
  BUG-07=N11 catch-narrowing (propagate device/OOM, prevalidate shapes),
  BUG-08=N12, BUG-09=N15, BUG-10=N04, BUG-11=N01, BUG-12=open-experiment
  policy (C1/C2 — no perf patch on an unmeasured causal assumption).

---

## 4. Live matrix (bench127) — results

All Fresh, single draw per cell unless noted; temp 0, 256 out, ignore_eos;
deep cells = nwords 115000/221000 → **136,191 / 261,757 ptok** (note
261,757+256 = 262,013 ≠ the exact 262,144 boundary, C9.11 — exact-boundary
cells remain §7 Stage C).

### 4.1 Per-lane decode curve (tok/s)

| lane | pool tok | 2k | 16k | 32k | 65k | 136k | 262k-class | ACCEPT (2k battery) |
|---|---|---|---|---|---|---|---|---|
| df7_fp8 | 373,507 | 23.3 | 19.3 | 12.9 | 8.0 | 5.4 | 5.2 | 0.336 |
| df7_tq4 | ~1.3M | 46.6 | 29.4 | 19.8 | 14.3 | **4.2** | 6.7 | 0.473 |
| mtp4_fp8 | 707,980 | 32.3 | 16.0 | 14.6 | 12.6 | 6.8 | 4.7 | 0.572 |
| mtp4_tq4 (prod) | ~1.3M | **60.6** | **37.8** | **30.7** | **20.9** | **11.7** | 5.6 | **0.614** |

v125 deltas: df7_tq4 65k 14.3==14.3 and 128k 4.2==4.0-4.2 (**D10
deterministic**); mtp4_fp8 @0.9 = P-1 SLOW class (v125 @0.85 refs 42/34/22
→ 16.0/14.6/12.6, C1). First-ever df7_fp8 262k-class cell: 5.2 (capacity
373k pool holds ONE such request + nothing else). **Prod lane (mtp4_tq4) is
the FASTEST lane at every ctxscan depth AND at 136k (11.7 = 2.8× df7_tq4's
4.2 on identical prompts+KV)**; its curve decays monotonically — only the
df7_tq4 curve is non-monotonic (4.2 @136k < 6.7 @262k), which is the D10
anomaly itself. Lane 4 = attempt-2 boot after the §4.9 DEVICE_LOST crash;
attempt-2 completed all phases cleanly (STALLS=0, ERRS=0, DEVLOST=0).

### 4.2 Deep-cell detail (prefill_tps / ttft / decode_tps)

| lane | 136k prefill | 136k ttft | 136k decode | 262k prefill | 262k ttft | 262k decode |
|---|---|---|---|---|---|---|
| df7_fp8 | 1084 | 125.7 s | 5.4 | 1249 | 209.5 s | 5.2 |
| df7_tq4 | 1536 | 88.7 s | 4.2 | 2034 | 128.7 s | 6.7 |
| mtp4_fp8 | 1098 | 124.0 s | 6.8 | 1225 | 213.7 s | 4.7 |
| mtp4_tq4 | 1492 | 91.3 s | **11.7** | 1847 | 141.7 s | 5.6 |

TQ4-vs-fp8 deep prefill: **1.42× @136k, 1.63× @262k** (C4's "1.5×" verdict
confirmed, grows with depth — the O(n²) dequant amplification). MTP-vs-dflash
prefill parity (fp8 lanes within 2%). TTFT at 262k-class ≈ 2.1-3.6 min on
all lanes (chunked prefill at ~1.1-2.0k tok/s).

### 4.3 Concurrency

| lane | conc8@2k agg (per-client) | conc4 mixed agg | conc2@65k (A/B) | asym |
|---|---|---|---|---|
| df7_fp8 | 100.3 (14.2-15.2) | 19.8 | 3.2 / 9.6 | 3.0× |
| df7_tq4 | 94.4 (18.5-21.7) | 35.8 | 12.3 / 3.9 | 3.2× |
| mtp4_fp8 | 50.0 (7.0-8.7) | 24.7 | 7.3 / 2.8 | 2.6× |
| mtp4_tq4 | **123.2** (18.1-39.2) | **61.6** | 3.6 / 10.2 | 2.8× |

**D8 reproduced on ALL FOUR lanes with ORDER FLIP** (fp8: second client
faster ×3; tq4-df7: first ×3; mtp4_fp8: first ×2.6; mtp4_tq4: second ×2.8)
— confirms the PrefillAdder head-of-line lottery is order-deterministic, not
dtype- or drafter-caused (C3/N14 stands; conc8 per-client spread up to 2.2×
on the prod lane, mixed conc4 also 1.8×+ spread everywhere).

### 4.4 loop8 think-trap scan (8192-budget, preserve_thinking+xhigh)

- df7_fp8: 3/4 think-trapped + **1 genuine periodic tail (`"!!!!"`×128)**.
- df7_tq4: 4/4 trapped, **3 periodic tails** (2 `"!!!!"`×128-class + 1).
- mtp4_fp8: 3/4 trapped, **0 periodic tails** — and P6 produced a natural
  finish (7,822 tok, stop, 1,566 content chars).
- mtp4_tq4: 3/4 trapped, **3 periodic tails** (P2/P4/P5, all `"!!!!"`×128),
  P6 natural finish (7,079 tok, stop, 1,709 content chars).
- **Lane-4 verdict overturns the dflash-only hypothesis**: tails now cluster
  on the tq4 KV lanes (6/8 runs: 3+3) vs fp8 lanes (1/8: 1+0) across BOTH
  drafter families, with the identical `"!!!!"`×128 unit on every lane that
  shows them — the signature is **KV-dtype-correlated (TQ4), not
  drafter-caused** (fresh N02 lead; quantization-induced repetition is now
  the primary suspect — BUG-05's paired-lane replay on tq4-vs-fp8 at
  identical seeds/prompts is the direct instrument).
- Natural finishers are MTP-lane-only so far (P6 on both mtp lanes; 0 on
  df7) — weak signal at N=4/lane, recorded not concluded.

### 4.5 Health tails

| lane | STALLS | F8 | JIT | ERRS | DEVLOST |
|---|---|---|---|---|---|
| df7_fp8 | 20 | 1 | 13 | 0 | 0 |
| df7_tq4 | 15 | 1 | 16 | 0 | 0 |
| mtp4_fp8 | 0 | 1 | 13 | 0 | 0 |
| mtp4_tq4 a1 | — (died in warmup, §4.9) | — | 3 pre-crash | **3 (TP0)** | **1** |
| mtp4_tq4 a2 | 0 | 1 | 16 | 0 | 0 |

All df7 DFLASH_STALL firings whose timestamps are preserved = warmup-phase
(C2 verdict); JIT counts are warmup-only (zero post-WARMUP_DONE expected —
v51 posture holds). Lane-4 attempt-1 counts cover only its ~5.5 min of life
(boot 17:53:24 → death 17:58:53); attempt-2 (post-reboot, 18:48 boot)
completed all phases with a clean tail — see §4.9 for the disposition.

### 4.6 Battery determinism fingerprint (SHAs, temp 0)

- `p2 c87e27c4` **identical on ALL lanes** (drafter- and dtype-independent).
- **Lane-4 attempt-2 = ALL SIX SHAs bit-exact with the prod refs**
  (`p1 3cecc747`, `p2 c87e27c4`, `p3 25058c3d`, `p4 744e88f5`,
  `p5 252b4dc1`, `p6 ebcc8258`) with ACCEPT 0.614 vs prod ref 0.613 —
  the prod config reproduces deterministically on a fresh post-reboot
  boot; this boot IS the §8 prod-restore battery gate passing.
- fp8 lanes share `p5 f61457ef` and `p6 724d7401` (dtype-determined,
  drafter-independent); tq4 lanes share p1/p4 with the prod refs
  (`3cecc747`/`744e88f5`).
- p1/p3/p4 differ on fp8 lanes (`83dd8aaa`/`12b16921`-class/`6765850d`) =
  the known lane-class flips (#12 k4-class for MTP, greedy-class for df7);
  deterministic within lane across boots (v52 posture holds).
- N01 loss event: **df7_fp8 boot log was NOT preserved** (container replaced
  before the docker-cp workflow existed) — per-phase forensics for lane 1
  impossible post-hoc; only the lane report values survive. This is exactly
  the N01 "record container+image ID + preserve raw logs per phase" case.

### 4.7 D10 attribution (fresh, from preserved-log SpecDecoding windows)

Deep-decode acceptance: df7_tq4 **2.1-2.5 @136k vs 5.5-8.0 @262k**;
mtp4_fp8 3.3-3.5 vs 3.5-4.0 (flat) while its tps drops 6.8→4.7 (= device
time ×1.45 growth with depth); df7_fp8 flat 5.4→5.2 (fp8 per-step cost
masks any acceptance gain). Model `tps ≈ acceptance / step_time`: df7_tq4
predicted inversion from acceptance ratio ×~2.8 and mtp-class device growth
×1.45 ⇒ ×1.9 expected vs ×1.6 measured — consistent. **D10 = df7 drafter
acceptance collapse on the 136k-class prompt, NOT a device-side cliff.**
Residual question: why does the dflash2 drafter collapse at 136k but soar at
262k on same-family prompts (draft-window conditioning vs prompt text) →
**discriminator = df7 boot with seeds swapped at 115000/221000** (cheap,
queued). **Lane-4 confirmation landed**: mtp4_tq4 on the SAME prompts + SAME
tq4 KV = 11.7 @136k (vs df7_tq4 4.2) and a monotonic decay curve — the MTP
drafter does NOT collapse at 136k, pinning D10 on the dflash2 drafter's
136k acceptance inversion (drafter-specific, dtype-independent: df7_fp8
5.4-flat vs its own 262k 5.2).

### 4.8 Post phase (on live mtp4_tq4) — COMPLETE (MASTER127_DONE 19:18:59)

- **P1 16k N=3 distinct-prompt dip classification**: 35.5/37.9/42.2 decode
  (ptok 16,097-16,176, ttft 8.2-8.4 s) vs ctxscan 16k single-draw 37.8 —
  the ~35-42 band is **systematic across distinct prompts**, not noise; the
  v53 single-draw 46.6 was the favorable outlier, and the "46.6→38.5 dip"
  is re-classified as draw variance around a stable band. [BUG-11 caveat
  applies: `ctok` values 1,029-1,073 are CHAR counts; decode_tps numerators
  valid under ignore_eos+max_tokens=256.]
- **P2 D7 cold-cold (two DIFFERENT 128k-class prompts back-to-back, no
  prefix overlap)**: seed 71 (135,472 tok): ttft 90.6 s, prefill 1,495,
  decode 10.3; seed 72 (137,446 tok): ttft 92.4 s, prefill 1,488, decode
  10.9 — **first ≈ second within 6%: NO cold-vs-warm asymmetry on the prod
  config**. The documented 4.2→7.3 asymmetry is therefore df7-lane/D10-
  specific (tied to the drafter acceptance inversion), not a first-request
  or JIT/pool-state property.
- P3 memory snapshot: `xpu-smi dump -m` printed nothing in-image (driver
  arg drift — cosmetic; device states captured via `xpu-smi discovery`
  during the §4.9 forensics).

### 4.9 Lane-4 DEVICE_LOST crash forensics (mtp4_tq4 = PROD CONFIG)

Second DEVICE_LOST-during-warmup event in lineage (N04/C9.6 class), now with
a fully preserved chain (`logs/b_mtp4_tq4.log` + `logs/dt_warmup_lane4crash.log`
+ kernel journal):

1. 17:53:24 lane boot (TurboQuant grid=256 live, prod flags 0.9/262144) —
   healthy; dt_warmup p1-p6 normal.
2. 17:56-17:57 **warmup phase 7 (36k) abnormally slow: 0.7 tok/s** (siblings
   ~10-25) — first anomaly, ~90 s before the fault.
3. 17:57:21 `_tq_full_dequant_kv` JIT (phase 8, ~70k chunk bucket); 17:58:49
   `_tq_mq_decode_stage1` JIT; 17:58:50 `_tq_mq_fwd_stage2` JIT — the TQ
   long-context kernel ladder compiling mid-phase.
4. **17:58:53 `RuntimeError: level_zero backend failed with error: 20
   (UR_RESULT_ERROR_DEVICE_LOST)` ×3 on Worker_TP0 (pid=473)**; kernel
   journal: `xe` **ccs engine reset on BOTH devices** at the same second
   (da:00.0 + b1:00.0, guc_id 32/22). TP1 never returned from the collective
   → shm_broadcast 60 s warnings.
5. Zombie window 17:58:53-18:03:53: /health still 200 (v50 F8-era lesson:
   health ≠ alive), request stalled at 88,065 tokens (spec decode tokens
   [-1,-1,-1,-1], kv_cache_usage 0.0817).
6. 18:03:53 `TimeoutError: RPC call to sample_tokens timed out` →
   EngineDeadError → **clean container exit(0)** 18:04; journal: second
   engine reset pair (bcs class, both devices) at 18:03:55 = teardown of the
   wedged collective.

Context: kernel log shows the same paired ccs/bcs engine-reset signature at
10:17, 14:02/14:07, 15:31/15:36 (pre-matrix activity — the host has a
repeating xe engine-reset pattern under this workload class, not a one-off).
Disposition per §8/BUG-10: host rebooted before relaunch (device "normal" in
xpu-smi does NOT mean clean driver state after an engine reset); lane 4
relaunched ONCE post-reboot — a recurrence = deterministic N04 finding and
the lane stops. Fresh leads: (a) **TQ JIT × device-lost timing** — all 3
faults sit ≤4 s after fresh TQ-kernel JIT compilations at a NEW chunk
bucket; if it recurs at the same warmup phase, the discriminator is a
no-JIT warmup (pre-warmed kernels) vs stock; (b) p7 slowness may be the
first symptom of the same fault, not an independent issue.

**Attempt-2 outcome (post-reboot, 18:48:02 boot)**: warmup p1-p9 clean in
3:00 (WARMUP_OK 18:54:05 — vs attempt-1 dying at 5.5 min), then battery →
ctxscan → deep → loop8 → conc → health tail ALL completed
(LANE_DONE 19:11:08, STALLS=0/ERRS=0/DEVLOST=0). Non-recurrence on the
identical config after a driver-state reset classifies attempt-1 as
**driver-state-dependent, not workload-deterministic** — but the host's
repeating engine-reset history (4 events in 8 h) keeps N04 at P0: the
GuC-firmware-drift ops ledger item (#5) and a xe/level-zero escalation
remain the standing recommendations. Crash evidence preserved:
`logs/b_mtp4_tq4_crash1.log` (attempt-1 slice, 668 lines) alongside the
full bind-mounted `logs/b_mtp4_tq4.log` (both attempts, appended).

<!-- v127-audit matrix + post phase COMPLETE; discriminators appended to -->
<!-- §3 C1/C2 (both closed). -->

### 4.10 Post-matrix discriminators — D10 order-swap + P-1 axis-split COMPLETE

- **D10 order-swap (`d10_swap.sh`, df7_tq4 boot 19:29:18, deep cells
  REVERSED)**: CELL_A 221000-first = **6.7 tps, acceptance 5.5-8.0**;
  CELL_B 115000-second = **4.2 tps, acceptance 2.1-2.5** — the collapse
  followed the prompt, not the position (lane order gave the mirror image of
  bench127 lane df7_tq4). CELL_B was even a full prefix-cache hit (TTFT
  1.9 s) and still 4.2 ⇒ prefill-state exonerated alongside position/state/
  JIT. Evidence: `bench127/d10swap/{report.txt,accept_windows.txt,
  b_d10swap.log}`. Verdict folded into §3 C2.
- **P-1 axis-split (`p1_disc.sh` P-1a, `p1b.sh` P-1b)**: unit discovery
  (956.73 tok/unit), P-1a 1,120,330-tok pool PATHOLOGICAL (0.4 tps @2k,
  warmup died phase 3), P-1b 598,912-tok pool @GMU 0.9 **FAST**
  (WARMUP_OK 20:32:15; 32.0/35.4/27.2/22.3). Dose-response + verdict in
  §3 C1; override-validation defect filed as N19. Evidence:
  `bench127/{p1/report.txt,p1b/report.txt,p1b/pyspy_dumps.txt,
  logs/b_p1b.log}`.

## 5. Unified fixes & improvements backlog (v127 = v125 L-items ⊕ sibling N01–N16)

All items Proposed unless marked Fresh/Historical. Rollback switches stay
until correctness+perf+soak gates pass. Order: P0 hygiene first (they gate
trust in everything else), then profiling/root-cause, then kernel work.

| ID (maps) | pri | problem (evidence) | fix / investigation | validation gate |
|---|---|---|---|---|
| **N01** (new) | P0 | measurements can cross container replacement; bench scripts assume 256-out | record container+image ID per batch; abort attribution on change; consume final usage/DONE; distinguish chars/events/tokens | unit fixtures (grouped spec chunks, empty content, missing usage, disconnect) |
| **N02** (extends C8) | P0 | Fresh df7_fp8 periodic tail (`"!!!!"`×128) + 3/4 budget-exhaustion — separate outcome classes; cause unattributed | archive full reasoning/token IDs/sampler; paired nonspec/MTP/dflash same KV/seed/prompt; check EOS masking, rejected-token cleanup, NaNs, state rollback, parser separation | 4k/8k/16k budgets × temp0/0.7/1.0; zero structural token errors; loop-rate vs budget-exhaustion reported separately |
| **N03** (=L2b; = LC-3 MTP + LC-4 dflash arms; C5 correction applies) | P0 | proposer-local narrowing wedges under async+whole-step graphs (v53) | scheduler-owned width at step boundary, serialized to both ranks/runner/proposer/verifier/output accounting; width-bucket captures; drain old-width steps; assess upstream Dynamic SD (MRv1 piecewise-only; full graphs need MRv2) first; LC-4 arm wires dflash2 selector-confidence / measured ctx→k acceptance through the same protocol (real code, not a knob — C5); true K0 = target-only, spec resume needs draft catch-up | width transitions both directions, K∈{0,1,2,4,7}, prefill tails, cancel/preempt, mixed batches, 512/2048 boundaries, 262k; no orphan placeholders |
| **N04** (extends #5) | P0 | DEVICE_LOST vs silent wedge share downstream symptoms | first-error timestamps + per-rank phase IDs; correlate kernel/driver logs; fail lane on device loss before retry | repeated clean boots, deep chunked prefill, teardown; no hidden failed attempts. **Fresh §4.9**: prod-config lane died DEVICE_LOST during warmup, chain fully preserved (p7 slow → TQ JITs → ccs engine reset both devices → RPC timeout → EngineDead); host engine-reset history pre-matrix at 10:17/14:02/15:31 — repeating, not one-off; reboot-then-one-relaunch posture applied |
| **N05** (=P-1; C1 discriminators fold in) | P1 | fp8×spec×GMU0.9 per-step 1.7-2.6× penalty, mechanism unknown | factorial GMU{0.80,0.85,0.90}×spec, fixed model/maxlen; log pool/group/block sizes, allocator peaks, graph buckets, kernel routes/times, draft/target/collective time; `num_gpu_blocks_override` discriminator (p1_disc.sh, deployed; CONFOUNDED — C1) + **LC-5 ballast pair** (pool-fixed margin-shrink; margin-fixed pool-shrink) + py-spy stack diff; inspect device/host timelines, not single counters | ≥3 fresh processes, distinct prompts; dominant delta ≥50% of gap identified; then 0.9==0.85 ±5% |
| **N06** (=L1+L6; = LC-1 decode + LC-2 continuation; C4 design) | P1 | TQ decode latency-bound at depth (26 GB/s effective, 2 orders headroom) AND continuation O(n²)+2 GiB transients | tiled compressed-KV attention: dequant K/V inside tile load, online softmax, preserve rotation+norm-correction; covers decode (q1), verify (q2-8), continuation prefill; FA fallback kept | all TQ layouts × q∈{1..8,prefill} × page tails × GQA × causal/window; logits tolerance; 131k/262k ctxscan ≥2×; prefill VRAM spike <200 MiB |
| **N07** (L1 prerequisite) | P1 | TQ deep decode slower than fp8 despite fewer bytes (−18/−36%) | profile unpack/centroid/rotation/occupancy/spills/bandwidth/split-reduction; specialize q1 vs verify independently; tune tiling vs measured bottleneck | matched-shape microbench + real 128k/262k; no short-ctx/conc regressions |
| **N08** (new) | P1 | hybrid SSM precision/boundary risks on long outputs | fp16-vs-fp32 SSM cache same weights/KV; accepted-token state copy/rollback; dtype-dependent effective block requirements | exact 512/1024/2048±1 boundaries, reject-all/accept-all, resumed prefixes; no cross-request contamination |
| **N09** (extends C6) | P1 | capacity/lifetime under-characterized; admission over-promises | worker allocated/reserved/peak per group; graph/scratch attribution; request lifecycle counters; admission reserves prompt+output; reclaim checks after abort/EOS/preempt | long→short cycles, 100+ cancels, 30-cycle plateau, 24h soak; distinguish cached vs leaked |
| **N10** (=L8/P2) | P1 | spec host/IPC chain erases conc gains (#16/#17) | profile exposed critical path (not stack %); move only proven bottlenecks into bounded worker-resident loop; fuse metadata copies; preserve fairness/cancel | C1-64 open-loop arrivals+aborts; aggregate goodput + p95/p99, not headline t/s |
| **N11** (=L4; C9.2 caveat) | P2 | ragged fp8 verify misses SPLITS kernel (D2) — but 2000/2000 measured calls uniform | MEASURE hit-rate first; then prealloc gather/scatter or ragged kernel via cu_seqlens; catch only recoverable shapes (never swallow device faults) | q_i=0..8, waste threshold, empty partitions, graph replay; gate ≥99% on ragged conc8 |
| **N12** (new, Code) | P2 | `_DEQUANT_BUFS` keyed by device only; length/Hk checked, D/dtype/ownership not | key workspace by device+head_dim+dtype+stream; event-protected reuse; capture lifetime | alternate models/head dims, draft/target paths, concurrent-stream stress |
| **N13** (=L7 follow-up) | P2 | warmup costs boot time; deep transfer unproven (C9.3) | separate compile-warmup / prefix-priming / cold-unique; size buckets by tokenizer IDs; keep only buckets that help | fresh-process A/B, equal-length distinct prompts; readiness time + first-unique TTFT reported |
| **N14** (=L9/D8; C3 design) | P2 | conc2@65k asymmetry (head-of-line chunk monopoly) + mixed-arrival tails | timestamp admission/schedule/prefill/decode per request; THEN `VLLM_V1_PREFILL_ROUND_ROBIN` chunk-share knob; aging only if traces show starvation | mixed 2k/16k/128k/262k staggered, ≥100 reqs; per-client distribution; conc2@65k ratio ≤1.5, conc8 agg −≤5% |
| **N15** (new) | P1 | enum/backend/kernel support mismatch (15-value enum vs exercised routes) | startup capability validation: arch × target+draft dtype × dims × scales × graphs × kernel availability; clean informative rejection pre-device-work | all 15 enum × 3 modes; unsupported = explicit-rejection pass |
| **N16** (new) | P2 | image reproducibility across overlays (27 modified + 7 added wheel files) | bake source SHA256 manifest + dependency lock + imported-path check + loaded-library IDs; reject stale patch bases | rebuild by digest, manifest compare, semantic smoke |
| **N17** (=BUG-01; C11, reproduced) | P1 | merge_async_iterators multi-input cleanup doesn't await children (CPU-reproduced; v52 F9 fixed only the single-input fast path) | cancel outstanding anext tasks → await outcomes → aclose ALL children incl. raised ones → supervised cleanup task on timeout; log iterator/request identity; single-input path + serving finalizers follow same ownership | zero/one/many iterators; child raises; delayed finalizer; early consumer break; cancel during anext AND during cleanup; repeated cancel; disconnect integration: engine request/block release + healthy next request |
| **N18** (=BUG-02; C11, code-confirmed) | P1 | DFLASH2_EMIT_K read unconditionally by async_scheduler (v42c) — leftover env silently changes MTP/nospec placeholder width | central validated config (method, native, emitted, scheduled width) resolved pre-worker; irrelevant setting = loud error/warn; scheduler+proposer consume resolved config, not env | nospec/MTP/dflash1/dflash2 × unset/empty/0/1/2/7/neg/nonint/out-of-range; no worker/collective start on invalid; stale env across sequential lane switches |
| **N19** (new; C1 P-1a, Fresh) | P1 | `num_gpu_blocks_override` honored with NO validation vs profiled available KV memory (`kv_cache_utils.py:2017` logs the override, then reports the inflated pool as healthy): fork units ≈ 956.73 tok/unit — P-1a booted 1,120,330 tok ≈ 19.5 GiB/rank vs `Available KV cache memory: 12.3 GiB`, silently thrashing to 0.4 tok/s near-OOM | validate override against profiled KV budget (clamp + loud warn with both numbers); document fork unit size; upstream-parity check of override semantics | override ∈ {626, 740, 1171, 2000} × fp8/tq4: boot log must show request vs budget; >budget ⇒ hard fail; 626/740 pools decode FAST-class per C1 dose-response |
| **L0** (ops, Fresh) | — | routing per workload | §6 table | standing |

Dynamic width is N03's umbrella with two arms: **LC-3** (MTP —
scheduler-owned width at the step boundary, serialized to
ranks/runner/proposer/verifier/accounting) and **LC-4** (dflash2 — selector
confidence or measured ctx→k policy through the same protocol; C5
correction: the DSPARK knob is inert on this path, this is real code, not
configuration). Both carry the v50/v53 bit-twice history: full
determinism/battery/crash gates before any promotion. Companion-plan
mapping: LC-0=N01/N04 attribution · LC-1=N06/N07 decode kernels · LC-2=N06
continuation · LC-3/LC-4=N03 · LC-5=N05 ballast · LC-6=N02/N08 quality
ladder; latency-throughput workstreams 1-8 map 1=N01, 2=N09/N14,
3=N06/N07, 4=N06, 5=N03, 6=LC-4 selector overhead, 7=N10, 8=N05/N12;
bug-remediation BUG-01=N17, BUG-02=N18, BUG-03=N03, BUG-04=N02 protocol
arm, BUG-05=N02, BUG-06=C5, BUG-07=N11, BUG-08=N12, BUG-09=N15, BUG-10=N04,
BUG-11=N01, BUG-12=open-experiment policy (C1/C2).
**Review-unit sequencing (bug-remediation R1–R8, adopted as the
patch-boundary discipline)**: R1=N17 · R2=N18 + C5-startup-reject + N15 ·
R3=N01 (harness identity/protocol) · R4=BUG-04 output-status protocol ·
R5=N02 replay repair · R6=N11 catch-narrowing + N12 · R7=N03 (only after
R4/R5) · R8=N04 + isolated perf (N05/N06/N07). Separate reviewable
patches/images; existing guards stay enabled until replacements demonstrate
equivalent containment AND correct behavior; no production hot-swap during
comparative runs.

## 6. Recommended configuration per workload (v127 refresh)

Boundary corrections vs v125 §5 (C9.8): **mtp4+tq4 @16k (46.6) beats
nospec+fp8 (33.5)** — spec stays the 16k-class recommendation; the
nospec+fp8 crossover begins somewhere in 16k–32k (prompt-dependent).
Capacity correction: tq4 vs fp8 pool = **1.84×** (not 4.9×).

**Profile framing (latency-throughput plan, adopted)**: minimum latency and
maximum aggregate throughput compete — publish **three validated profiles**
per deployment instead of one "optimal" setting: (1) **first-token latency**
(queue delay + cold unique-prompt TTFT); (2) **interactive decode** (ms per
emitted token, final-answer latency; record time-to-first-FINAL-answer —
`preserve_thinking`+`xhigh` streams reasoning tokens early while delaying
the answer); (3) **concurrent throughput** (quality-passing output tokens/s
under declared latency SLOs; count only requests passing quality/protocol/
latency gates, keep failed/unfinished work in separate load/error counts;
never reward periodic repetition or padded output as useful generation).
Reminder: a prefix hit improves prefill work only — it never removes
full-context attention during subsequent decode.

<!-- TBD-matrix: full refreshed table with v127 numbers (esp. 262k cells + -->
<!-- df7 fp8 373k capacity note). Standing statics: fp8+spec ⇒ GMU 0.85 until -->
<!-- N05 fix (or num_gpu_blocks cap per p1_disc result); df7_fp8 cannot hold -->
<!-- 2×262k; SPLITS knobs stay default; keep user env block verbatim -->
<!-- (CCL settings are host-specific, do not "optimize"); reasoning_effort -->
<!-- xhigh + preserve_thinking = long-thinking default — bound per request -->
<!-- when latency matters (#20 posture). -->

## 7. Comprehensive test plan (real scenarios)

Definitions (adopted): **128k = 131,072 tokens; 262k = 262,144 tokens**
(log decimal lengths separately if 128000/262000 meant). Exact-boundary
cells: 131072+256; **261,888+256 = 262,144 total**; negative boundary
tests beyond-limit / boundary−1 / exact / +1 with documented API behavior
(rejection vs clipping). Long-generation alternative: 258,048+4,096 or
253,952+8,192 = 262,144 (a full-262k prompt leaves no output budget).
Tokenize through the actual server tokenizer and submit token IDs for
exact cells; for chat count the fully rendered template.

**Cost model (LC plan, adopted)**: score deep decode as `cost_per_token =
sum(decode wall time) / sum(emitted tokens)` per (lane, depth) — report
ms/token alongside tok/s. Historical Fresh references at ≈136k/≈243k ptok:
nonspec fp8 **39.1/47.2** ms/tok · nonspec tq4 56.5/104.2 · mtp4 tq4
85.5/163.9 · df7 tq4 238.1/137.0 (D10 anomaly inverted: worse at 136k than
243k). Controller keeps ≥10% margin over the measured floor before any
width change (speculation decision rule, C5).

**Milestones (companion M0–M6 mapped onto N-items)**: M0 measurement/
attribution hygiene (N01/N04) → M1 six-lane exact-depth baseline incl. the
mid-depth arm (Stage B) → M2 decode-first kernels (N07 profile, N06 q1 =
LC-1) → M3 scheduler-owned width (N03 = LC-3/LC-4) → M4 continuation kernel
(N06 prefill arm = LC-2) → M5 allocator/margin root-cause (N05 = LC-5
ballast + P-1c py-spy) → M6 quality ladder + release gates (N02/N08,
Stages C/D; 2h screening / 24h soak). Decode-first ordering (LC-1 and width
before LC-2) adopted: deep cost is decode-dominated (C4 verdict).

**Staging** (from sibling plan, adopted):
- **Stage A — support/lifecycle**: all 15 KV enum values × {nospec, mtp4,
  dflash7} on the pinned image; record supported / expected-reject /
  unexpected-boot-fail / generation-fail / pass (N15). Servable lanes get a
  32/2048-token smoke (natural EOS, one tool call, one cancellation) before
  any long run.
- **Stage B — primary perf matrix**: 6 primary lanes (fp8/tq4 ×
  nospec/mtp4/dflash7); prompt tokens {2048, 16384, 32768, 65536, 98304,
  131072, 261888} (98304 = mid-depth arm for width-decay/acceptance curves,
  LC), out 256, conc {1,2,4,8} (+16/32/64 short/mixed where capacity
  allows); ≥3 fresh processes × 3 unique equal-length prompts per cell; one
  pass in **reverse length order** to expose history effects; cold-unique /
  warm-unique / exact-prefix TTFT measured separately; deep-conc cells run
  **equal-resident AND equal-offered** (TQ4 may win useful throughput by
  avoiding queue/preemption even with slower single-client decode) and
  distinguish resident vs queued clients (capacity shortage IS a finding,
  not a failed test); synchronized bursts AND open-loop arrivals, raising
  offered load until queues/SLOs fail (synthetic mix 80% 2k/16k, 15%
  131072, 5% near-262k + an all-long arm; fixed 256 out first, then
  realistic 256/1024/4096/8192 with natural stops — never extrapolate
  short-output throughput to long reasoning).
- **Stage C — natural application scenarios**: interactive chat
  (multi-turn), coding (compile/test-scored), RAG single-needle at
  1/10/25/50/75/90/99% depth in 128k/262k, multi-hop joins, long-doc
  summary+quotes, tool agent (streamed args, parser equivalence), thinking
  regression (the 4 loop8 prompts + arithmetic/proof/DB), prefix
  reuse/isolation (incl. post-cancel state validity), mixed arrivals
  (80/15/5 short/16k/deep, staggered+burst, SLO goodput), streaming
  failure (cancel pre-first-token/mid-prefill/mid-decode, socket reset,
  slow reader → blocks/placeholders released), long generation (4k-16k
  out), memory pressure (capacity±1 then shorts). Output budgets
  {256,1024,4096,8192} + 16384 reasoning arm; temp 0 + sampled 0.7/1.0.

**Metrics**: request/client/container/image IDs, UTC+monotonic times,
tokenized-prompt hash+count, final usage, finish reason; TTFT to first
nonempty reasoning vs first final content; event-gap percentiles labeled as
such (spec chunks carry multiple tokens); true token ITL from server
timestamps or stated unavailable; aggregate = completed output tokens /
common interval with failures kept in the denominator; p95 needs ≥100
completed, p99 ≥1000 — deep small-N cells report individual values.
Per-rank allocated/reserved/peak, KV occupancy per group, prefix hits,
preemptions, draft proposed/accepted per position (request-window deltas,
not process-lifetime).

**Release gates**: (1) zero silent corruption / leakage / bad IDs /
accounting errors / resets / missing terminal events in the structural
suite; (2) no task-tolerance regression vs same-weights auto-KV reference
(quantized KV not assumed lossless; fp32-SSM compared separately); (3) no
new periodic-loop pattern on the regression corpus, budget exhaustion
reported separately (target-only failure ⇒ model/numerics; spec-only ⇒
rollback/sampling path); (4) perf promotion = ≥10% target-workload median
gain with uncertainty, ≤5% protected-workload regression, no SLO-goodput or
memory failures; (5) memory: no monotonic retention after warm-pool
stabilization across cancel cycles; queue/running recover post-abort; (6)
2h mixed stress for screening, 24h for release including long ctx, natural
stops, cancellations, graceful restarts.

**Execution sequence**: finish/capture the running bench127 job first
(exclusive host lock; no lane interleaving); archive raw outputs; per lane
manifest→boot→health+semantic smoke→cold→warm-unique→perf→semantic/loop→
cancel/memory→teardown→error-log capture. Stop a lane immediately on device
loss or structural corruption; preserve first error; remaining cells marked
not-run. Implement N01/N02/N15/N16 first, then N05/N08/N09 profiling;
N03 and N06/N07 as separate patches/images; N11 only on measured ragged
fixtures; combine patches only after individual A/B gates, then full-matrix
rerun. Original v125 digest stays available; no baseline retagging; no
production hot-swaps during comparative runs.

**Fix-specific regression tests** (unchanged): N05 fix ⇒ 0.9==0.85 ±5%
16k/32k/65k mtp4_fp8; N14 ⇒ conc2@65k ratio ≤1.5, conc8 agg −≤5%; N03
(LC-3/LC-4) dynamic width ⇒ deep decode ≥ nospec ×0.95 AND short-ctx
battery SHAs unchanged; N11 ⇒ gate hit-rate ≥99% ragged conc8; N06 ⇒
ctxscan 131k/262k ≥2× and 227k-class prefill ≥2500 tok/s.

---

## 7A. R-LC directive — flat decode across 64k/128k/262k (deep-research design)

**Governing requirement (user, 2026-09-07, supersedes per-depth expectations
in §6 profiles):** from 64k through 128k and 262k context, only a MINOR
decode-tps drop is allowed. Decode-tps-vs-depth must be near-flat; suspects
named by the user (memory bandwidth, PCIe bandwidth) were deep-researched in
code + web; the deliverable is super-efficient KV-cache handling and data
movement, the fastest GPU↔GPU backend, mamba/jamba/splitting-style
techniques, and minimal inter-GPU traffic.

### 7A.1 Current posture vs the requirement

Prod lane (mtp4_tq4, Fresh §4.1): 2k 60.6 → 32k 30.7 → 65k 20.9 → 136k 11.7
→ 262k 5.6 tps. That is a **10.8× falloff 2k→262k and 3.7× from 65k→262k** —
far beyond "minor". The design below shows why the requirement is
nevertheless PHYSICALLY REACHABLE on this hardware, and where the loss really
lives.

### 7A.2 Hardware ground truth (web-verified 2026-09-07)

Per rank: Intel Arc Pro B70 (Xe2/Battlemage, 148 Xe cores), 32 GB GDDR6,
256-bit bus, **608 GB/s** DRAM; host link **PCIe Gen5 x16** (~63.8 GB/s
theoretical per direction); **no Xe Link / no GPU fabric** — the two B70s are
PCIe peers through the host root complex, so every TP collective is either
L0 P2P over PCIe or host-staged shared memory (oneCCL picks P2P when the
topology permits; falls back to SHM staging — Intel oneCCL docs; see also
arXiv:2409.09874). TDP 230 W. Sources: Intel product pages, Puget Systems
B70 review, TechPowerUp GPU DB.

### 7A.3 Bandwidth ledger — what the hardware actually permits

Measured pool math (§4.5, Fresh): fp8 KV = 12.3 GiB / 707,980 tok =
**18.65 KB/token/rank**. The checkpoint architecture is documented in the
v125-audit header: **hybrid GDN — 48 linear-attention + 16 full-attention
layers of 64, GQA 24q/4kv × hd256** (boot arg `mamba_ssm_cache_dtype: fp16`
confirms). Cross-check: 16 layers × 2×4kv×256 × 1 B / TP2 = 16.4 KB/tok/rank
vs 18.65 measured (+14% = block padding/scales/state overhead) — the pool
bytes ARE the attention minority; the other 48 layers contribute
constant-size state only. This is the Jamba (arXiv:2403.19887) / Qwen3-Next
pattern, and it caps the depth-scaling problem to 16 of 64 layers.

| cost / decode step (bs=1, per rank) | @64k | @136k | @262k |
|---|---|---|---|
| weights read (27B fp8 / TP2 = 13.5 GB) | 22.2 ms | 22.2 ms | 22.2 ms |
| attention-KV read (fp8, 18.65 KB/tok/rank) | 1.2 GB → 1.9 ms | 2.5 GB → 4.0 ms | 4.7 GB → 7.7 ms |
| **memory-bound floor / step** | 24.1 ms | 26.2 ms | 29.9 ms |
| **floor tps (nospec)** | **41.5** | **38.2** | **33.4** |
| floor tps (mtp4, ×~1.6-2 effective) | ~66-83 | ~61-76 | ~53-67 |
| measured prod tps | 20.9 | 11.7 | 5.6 |
| **measured / floor** | **2.0×** | **3.3×** | **6.0×** |

Conclusions (this is the central R-LC finding):

1. **The bandwidth floors falloff 64k→262k is only −20%** (41.5→33.4 nospec).
   A decode path that ran at ≤1.5× floor would show a ~minor~ drop and clear
   every gate in 7A.6 with spec on top. **The user requirement is achievable
   in software; no hardware change is implied.**
2. **DRAM bandwidth is NOT the binding constraint** (2-6× above floor at
   depth). The loss is kernel/step-path inefficiency: non-tiled attention
   over long KV (TQ `_tq_full_dequant_kv` full-history dequant continuation
   — the exact JIT class from §4.9 warmup p6/p7 — and non-split fp8 MQ
   fallbacks), plus fixed per-step host/launch overhead that stops
   amortizing as steps lengthen.
3. **PCIe/GPU↔GPU is second-order for decode**: bs=1 AR payloads are
   KB-class, so collectives are latency- (~2×PCIe RTT + staging per
   collective × layers ≈ 2-4 ms/step) not bandwidth-bound — 10-20% at 2k,
   ~2% at 262k (179 ms/step). Where PCIe genuinely hurts is **prefill/TTFT**
   (GB-class streamed AR: 262k TTFT 1.9-3.6 min on every lane) — a separate
   axis with separate fixes (W2/W5).
4. Even at 2k the engine reaches only ~48% of the nospec+spec step ceiling —
   there is headroom at EVERY depth; depth just amplifies the fixed-cost
   share.

### 7A.4 Workstreams (W1-W7 → backlog mapping)

- **W1 — Tiled/split compressed-KV decode kernel (N06/LC-1, the structural
  fix).** Generalize the proven v51 two-stage fp8 MQ flash-decode (SPLITS
  parallelism; +34% @32k with SPLITS 64) and the TQ v52 tiled path to ALL
  depth buckets; eliminate `_tq_full_dequant_kv` continuation (dequant
  per-tile inside the kernel, never full-history); SPLITS scaled by context
  bucket (64/128 at ≥128k). Gate: attention time ≤2× KV-BW floor at every
  depth (7A.3 table); N06 gate (131k/262k ≥2×) is the mid-milestone.
- **W2 — Fused continuation prefill (N07/LC-2).** Chunked prefill with
  compute/comm overlap and decode-kernel reuse for the tail chunk; target
  227k-class prefill ≥2500 tok/s and 262k TTFT ≤60 s. Prefill is where the
  PCIe AR bandwidth actually binds (7A.3 concl. 3), so overlap/bidirectional
  AR matters here, not in decode.
- **W3 — KV byte-floor program.** (a) Keep tq4 KV as the depth lane
  (10.4 KB/tok/rank measured ≈ 4.4-bit effective); (b) KIVI-style 2/4-bit
  per-channel-K/per-token-V with per-block scales fused into the W1 kernel
  (arXiv:2402.02750); (c) KVCompress-style block eviction / sliding window
  on attention layers only (arXiv:2410.00161 — hybrid makes this cheap:
  16 KV-bearing layers of 64, and linear/SSM layers are O(1) by
  construction). Effect is
  modest for single-stream decode floors but large for the **P-1 headroom
  axis (C1: pool bytes ÷ headroom drives a 1.7-2.6× penalty)**, for
  df7_fp8's 1.4-request capacity ceiling (§1), and for multi-client depth.
- **W4 — Hybrid-SSM leverage (no architecture change; the checkpoint already
  IS mamba/jamba-class).** SSM/linear layers have constant-size decode state
  — no KV growth, no tiling problem; ensure (i) `mamba_ssm_cache_dtype`
  stays fp16 (fp32 fallback would double state traffic — N08 verification),
  (ii) W1/W3/W6 target ONLY the 16 attention layers (bounded work), and
  (iii) no attention-layer assumption leaks into scheduler/admission
  accounting (token-based fine; byte-based must use 18.65 KB/tok/rank, not
  dense-model math — P-1a's 19.5 GiB surprise is the cautionary tale, N19).
- **W5 — Fastest GPU↔GPU backend for THIS topology (no Xe Link).**
  Measure, then choose: (a) microbench oneCCL allreduce latency/bandwidth by
  message size on the actual 2×B70 PCIe topology, L0-P2P vs SHM staging
  (oneCCL bench tool; log which path engages); (b) cut collective COUNT per
  step (fuse per-layer ARs; one-shot AR for KB-class decode payloads);
  (c) draft-model placement: dflash2 drafter resident on one rank with
  replicated embeddings instead of TP-split (removes drafter ARs entirely —
  drafter is small); (d) prefill-only: bidirectional/pipelined AR overlap
  (W2). Gate: per-step collective time ≤1 ms decode; prefill AR utilization
  ≥40% of PCIe peak during 128k+ chunks.
- **W6 — Splitting (flash-decode class).** SPLITS is the decode-side
  split-parallelism knob (folded into W1); add prompt-aware split scheduling
  (larger SPLITS for long-KV cells) and keep the v51 ceiling findings
  (SPLITS 128 = WEDGES-class ceiling 64 on this part — re-verify at 262k).
- **W7 — Adaptive spec width by measured acceptance (N03/LC-3/LC-4).**
  Scheduler-owned width per context band and per drafter health: D10 proved
  acceptance can collapse on specific prompt/length classes (dflash 2.1-2.5
  @136k-class prompts); adaptive width (or nospec fallback) converts an
  acceptance cliff into a minor dip. mtp4's 0.614 acceptance stays the depth
  lane's engine; W1-W3 make its per-step cost depth-flat.

### 7A.5 The "brilliant solution" in one paragraph

The documented architecture (16 full-attn + 48 linear-attn layers) means the
Jamba/Qwen3-Next flat-decode property is sitting in the checkpoint. The
10.8× depth falloff is therefore NOT intrinsic: it is the attention minority
(16 layers) traversing long KV through non-tiled dequant paths plus fixed
per-step costs, running 2-6× above a memory floor that itself only drops
20% across the whole 64k→262k range. Fix the attention path (W1), keep bytes
small (W3), keep headroom healthy (C1 mitigation/N19), and pay only ~1 ms of
interconnect per step (W5) — the ladder lands near-flat at multiples of
today's 5.6.

### 7A.6 Acceptance gates ("minor drop", measurable)

Expressed vs the same boot's 2k cell (lane-relative, temp 0, 256-out,
single stream; conc and TTFT gates separate):

- **G1**: decode@64k ≥ 0.75 × decode@2k
- **G2**: decode@136k ≥ 0.60 × decode@2k
- **G3**: decode@262k ≥ 0.50 × decode@2k
- **G4**: monotone ladder, no cell >15% below BOTH neighbors (D10-class
  collapses excluded by construction once C2's fix (W7) lands)
- **G5**: 2k battery SHAs unchanged + spec acceptance ≥0.55 at every depth
  (quality invariants while chasing speed)
- Cross-check vs floors: on the prod lane (2k = 60.6) the gates demand
  45.5 / 36.4 / 30.3 tps — all below the 7A.3 floor+spec envelopes
  (~66-83 / ~61-76 / ~53-67), i.e. the gates ask for ≤~50% of what the
  hardware already permits once the step path is fixed.

### 7A.7 Sequencing

- **Phase A (ops, no code, standing)**: C1 mitigation for fp8 lanes (pool
  cap 0.85-class or override 626); §6 routing; N19 clamp once coded.
- **Phase B (kernels)**: W1 (N06) then W2 (N07) — single-patch A/B per §7
  discipline, deep-cell gates first.
- **Phase C (bytes + comm + width)**: W5 microbench (cheap, informs W2
  overlap design), W3 INT2/4 + eviction, W7 (N03) last (needs healthy W1
  base for clean attribution).
- **Phase D**: full bench127-style matrix rerun + 7A.6 gates on all four
  lanes + conc/TTFT side-gates; prod promotion only after G1-G5.

### 7A.8 Source register (web, verified 2026-09-07)

- Intel Arc Pro B70 — Intel product pages; Puget Systems B70 review;
  TechPowerUp GPU database (608 GB/s, PCIe Gen5 x16, no Xe Link).
- Intel oneCCL documentation (L0 P2P transport preference, SHM fallback,
  benchmark tool); arXiv:2409.09874 (oneCCL/collectives perf methodology).
- vLLM fp8 KV-cache blog (vllm.ai) — fp8 KV as the deployed baseline W3
  starts from.
- KIVI — arXiv:2402.02750 (2-bit asymmetric K/V quantization).
- KVCompress — arXiv:2410.00161 (paged KV eviction, retrieval-accurate).
- Flash-Decoding — flash-attention split-KV decode parallelism (basis of
  the fork's SPLITS knob).
- Jamba — arXiv:2403.19887 (1:7 attention:SSM hybrid, constant-state
  decode); Qwen3-Next (hybrid Gated-Deliberation + linear attention) — the
  architecture family the audited checkpoint already belongs to.

## 8. Posture

- **Prod lane RESTORED and STANDING (20:47:55, `prod_restore127.sh`)**:
  mtp4 + tq4nc @0.9/262144 on v1.2.5, pool 1,298,151 tokens, HEALTH_OK
  20:43:51, WARMUP_OK 20:46:41; smoke ctxscan 2k/16k/32k/65k =
  61.9/51.0/26.8/21.3 (consistent with §4.1 lane refs). Battery gate
  satisfied by the §4.6 pass this round (6/6 SHAs bit-exact to refs
  3cecc747/c87e27c4/25058c3d/744e88f5/252b4dc1/ebcc8258, ACCEPT 0.614 vs
  0.613 ref) on the identical config+image — no re-batterying on restore.
- Standing ops: reboot after any GPU engine reset (#03/#5); graceful
  `docker stop -t 30` before rm; GuC firmware drift (70.44.1 loaded vs
  70.49.4 recommended) stays on the ops ledger for #5-class recurrences.
  **Applied this round (§4.9)**: prod-config lane-4 DEVICE_LOST + confirmed
  xe ccs/bcs engine resets on both devices → host rebooted before the single
  relaunch; reboot itself took >30 min to return to network (POST/GPU-init
  path — no OOB access from this seat; if the host hangs it needs physical
  intervention).
- Do-not-touch ledger (measured REJECTs): SPLITS=64/128 both dtypes,
  `VLLM_XPU_FP8_MQ_Q1`, EMIT_K=4/5, MNBT 16384, `CCL_ENABLE_SYCL_KERNELS=0`,
  `CCL_ATL_SHM=1`, `CCL_TOPO_P2P_ACCESS=0`, `--async-scheduling` removal,
  L2 narrowing (`VLLM_V53_L2_UNSTABLE` stays closed).

<!-- References: v125-audit REPORT (D1-D10, baselines); v125-fixes PLAN -->
<!-- (v53: L2 wedge conviction, D7 verdict, L2b design); KNOWN_ISSUES -->
<!-- #3/5/6/7/10/11/12/13/14/15/16/17/18/19/20; PERF_TUNING LONG_CONTEXT_ -->
<!-- ANALYSIS (v26/v28dbg/v29b/c); dflash2-fullfix-v51 sources; companion -->
<!-- plans: latest-image-audit REPORT + long-context-plan PLAN.md + -->
<!-- LATENCY_THROUGHPUT_PLAN.md + bug-remediation-plan PLAN.md. -->
