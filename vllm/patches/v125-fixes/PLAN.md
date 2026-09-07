# v125-fixes PLAN — implementing REPORT.md §4 levers (v53, images v1.2.6t1/t2)

Date: 2026-09-07. Base: `llm-scaler-exp:v1.2.5` (v1.2.5t5, cert cb95b6e1…/a522bf15…).
Source audit: `vllm/patches/diagnostics/v125-audit/REPORT.md` (Rev 2).
Host: 10.20.3.65, 2× Arc Pro B70, TP=2, model qwen3.8-27b-fp8.

## STATUS (post-validation, 2026-09-07)

| Lever | Image | Outcome |
|---|---|---|
| L2 ctx-adaptive k | t1→t2 | **CONVICTED UNSOUND under async scheduling** — see §L2 post-mortem. Narrowing gated behind `VLLM_V53_L2_UNSTABLE=1` (default closed, knob = observability only). Proper design = L2b (§L2b). Battery SHAs bit-exact in both inert configurations. |
| L4 ragged fp8 pad | t2 | **No-op on static-width spec lanes (proven by P-1)**: 2000 verify calls on mtp4/fp8 = 100 % `v51_uniform`, 0 ragged — whole-step scheduling makes verify uniform by construction (every request emits width+1 rows). Harmless (0 fires, battery bit-exact). Remains a safety net for variable-width traffic (dflash2 confidence truncation, future L2b). |
| P-1 route counters | t2 | **Working; §3.1 question answered**: ragged does not occur on uniform-width spec lanes. `V125_P1 routes: v51_uniform=2000(181245tok)` both workers. |
| L10 tier logs | t2 | **Hook unreachable in deployed config** (see §L10 post-mortem) — but D10 closed STRUCTURALLY: under FULL_DECODE_ONLY graphs `_tq_adaptive_splits=False` (turboquant_attn.py:523), `_effective_kv_splits` early-returns the constant graph grid (256, device-side re-partition) — the tier ladder never executes at runtime on these lanes, so the df7 128k-vs-227k decode asymmetry cannot be a KV-splits-tier effect. Lane C otherwise clean: 5/6 battery SHAs bit-exact (p3 = #18-class run-to-run knife-edge, 3 distinct draws), deep 4.2/7.3 == refs, STALLS=18 (v125: 17). |
| L7 deep warmup | t2 (host script) | **Calibration bug fixed** (p10 was 400-rejected: generator ~2.97 tok/target-unit, not 8.5 tok/record; calibrated 41000→121,701 tok / 67000→198,004 tok, container-tokenizer-measured). Buckets run at boot (tq4: p10 80.6 s, p11 158.1 s; fp8: 117/225 s; df7: 80/150 s). Benefit mixed: tq4-65k ctxscan 21.0→21.0 (no change), fp8-65k 9.6→12.8 (+33 %, matches post125 warm-class 12.5, N=1 caveat); **deep decode NOT warmed on any lane** (mtp4_tq4 11.7/6.1, fp8 6.6/4.4, df7_tq4 4.2/7.3 — all identical to refs): the post125 "warm deep" numbers were same-prompt prefix-cache-specific or second-deep effects, NOT transferable JIT warmup — D7's first-deep-cold is not warmup-fixable. |

Bottom line: **v1.2.6t2 = v1.2.5 behaviorally** (all levers measured inert or gated off;
battery SHAs bit-exact on both spec lanes; ctxscan/deep/conc within noise of refs) plus
new observability (P-1, L10, L2 armed-logs). No promotion case from this round; L2b is
the follow-up with upside.

## What shipped (v1.2.6t2 = v1.2.5 + 3 overlays + host-side warmup)

| Lever | File (in-image path) | Knob | Default | md5 (t2) |
|---|---|---|---|---|
| L2 ctx-adaptive spec k (observability; narrowing UNSTABLE-gated) | `vllm/v1/spec_decode/llm_base_proposer.py` | `VLLM_SPEC_CTX_K="<ctx>:<k>"` + `VLLM_V53_L2_UNSTABLE` | `""` (inert) / `0` (gate closed) | 7215ed994b24f842dc9b7bc0b1a6247e |
| L4 ragged fp8 verify pad + P-1 route counters | `vllm/v1/attention/backends/flash_attn.py` | `VLLM_V53_L4` / `VLLM_V125_P1` | `1` (on) / `0` (off) | 4500df595690ff51a4edf7c214efbe70 |
| L10 KV-splits tier logs | `vllm/v1/attention/backends/turboquant_attn.py` | `VLLM_V53_TQTIER` | `0` (off) | 192f701929d8614e898e0b750e65d11a |
| L7 deep warmup buckets | host: `/root/build/dt_warmup_v53.py` (not in image) | `V53_DEEP_WARMUP` | `1` (on) | (host-side; calibrated) |

Bake chain: `bake_v126t1_Dockerfile` (v1.2.5 → t1, proposer 7740c68c…), then
`bake_v126t2_Dockerfile` (t1 → t2, proposer only: ub-source cascade + UNSTABLE
gate). Base md5s gated at t1 bake: proposer `0371f785…`, flash `da085d4e…`,
tq `eb4c86b1…` (verified live in v1.2.5 before build). Build dir:
`/root/build/v126t1/`. Both bakes all-gates-green.

### L2 — deep-ctx spec-loss fork (REPORT §2.1/§4, D6) — POST-MORTEM

Problem (unchanged, from REPORT): mtp4 deep decode loses to nospec (11.7 vs
17.7 tok/s @128k; 242k 6.1 tok/s). Sequential drafter pays (k−1) eager draft
forwards/step while acceptance collapses at depth.

v53 t1 implementation: batch-min ctx adaptive width in the base proposer
(MTP loop cut + dflash emission truncation, list contract, floor k=1).

**Validation trail (Lane A, mtp4+tq4nc, 2026-09-07):**

1. t1 with `VLLM_SPEC_CTX_K=65536:1`: battery SHAs bit-exact (all six +
   ACCEPT 0.612 vs 0.613, p6 = documented #18 knife-edge) — but deep decode
   11.7/6.1 IDENTICAL to refs and 0 fire lines: **t1 never fired**. Root
   cause: the `_seq_lens_cpu` field is a DEPRECATED lazy cache that nothing
   in the spec path populates (backend.py:412 `= None`; only the deprecated
   `seq_lens_cpu` property fills it) — the read was always None → inert.
2. t2 cascade fix (`seq_lens_cpu_upper_bound` → shadow → one-shot sync):
   hot-swapped into the live container, rebooted, warmup fired
   `v53 L2 ctx-adaptive k fired (#1, src=ub): min_ctx=65536 >= 65536 -> k=1`
   on both workers — decision + source working (ub = resident CPU tensor,
   zero sync, error ≤ spec width vs a 65536 threshold).
3. **17 s later: WEDGE.** The request was mid-chunked-prefill (computed
   65536→88k, mamba v52f fixups tracking `draft=1` correctly, TQ MQ decode
   kernels JIT-compiling at the transition) when both TP workers went
   silent for exactly 300 s → `TimeoutError: RPC call to sample_tokens
   timed out` → EngineDeadError, engine death (crash log:
   `bench126/b_l2a_crash.log`).

**Root cause (from AsyncScheduler source, async_scheduler.py:21-45):** the
v42c contract — "The placeholder list width IS the scheduled spec width
under async scheduling; everything downstream (spec slot count,
num_output_placeholders accounting, engine-side −1 padding in
update_draft_token_ids_in_output) derives from the scheduled width".
`_spec_token_placeholders = [-1] * _emit_k` is computed ONCE at boot (from
`num_spec_tokens` / static `DFLASH2_EMIT_K`). The v42 EMIT_K cap is sound
because BOTH sides (scheduler + dflash2_proposer) read the same static env
at boot. A proposer-side DYNAMIC narrowing desyncs the scheduler's
placeholder/step accounting under async scheduling + whole-step FULL_DECODE_
ONLY graphs → worker wedge. mamba fixups tolerated the width (draft=1
tracked) — the failure is in the spec-slot accounting path, not mamba.

**Signature note (distinguished 2026-09-07 15:31, see prod-restore event
below):** the L2 wedge is a SILENT 300 s spin with ZERO device errors
(0 `DEVICE_LOST`/`RuntimeError` lines in `b_l2a_crash.log`). The
prod-restore crash the same day was the OPPOSITE signature — instant
`UR_RESULT_ERROR_DEVICE_LOST` on Worker_TP0 → same downstream
`sample_tokens` RPC timeout/EngineDeadError tail but a different,
hardware-level mechanism. Do not conflate the two.

**Disposition:** t2 keeps the machinery but gates narrowing behind
`VLLM_V53_L2_UNSTABLE=1` (default CLOSED; the armed knob then only logs
would-fire decisions). Evidence trail preserved for L2b.

### L2b — the correct dynamic design (follow-up, NOT implemented)

1. Decide the width in the SCHEDULER per step: it owns exact CPU-side
   context (`request.num_tokens`) — no GPU sync, no upper-bound heuristic,
   and it can apply per-request widths where the placeholders are built.
2. Plumb the decision: `SchedulerOutput` gains the step's spec width →
   gpu_model_runner → spec worker → proposer uses it (replacing any
   proposer-local decision).
3. Graphs: whole-step FULL_DECODE_ONLY capture must include BOTH widths
   when L2b is armed (the v42 "async+width-aware whole-step graphs"
   machinery is the precedent — it captures the EMIT_K-capped shape).
4. Verification: battery bit-exact below threshold; deep-decode gate
   (136k: target ≥ nospec-class ~17 from 11.7; 242k: from 6.1); sustained
   mixed conc (ragged verify → exercises L4 interplay); crash-sequence
   matrix from v123 cert.

### L4 — ragged fp8 multi-row verify misses SPLITS (REPORT §4, D2-class)

Problem: v51 fp8 uniform gate requires `num_actual_tokens == q*B`.
Ragged verify batches (unequal accepted tails; conc/mixed traffic) fall
to v33 MQ3D, which measured SLOWER than the FA2 varlen fallback at
several depths.

Fix: front-pad ragged rows into a pseudo-uniform `[B*q]` layout and run
the v51 SPLITS Triton kernel, then unpad. Causal math stays exact: real
row j′ of a q_i-wide request sits at slot `q−q_i+j′`; the kernel's
per-slot limit `seq_len−q+1+j` becomes `seq_len−q_i+j′+1` — exactly the
row's own causal limit. Pad rows produce empty partials, handled by the
kernel's stage-2 blind combine by design (verified against
`triton_fp8_mq.py:343,135-138` + stage-2).

Gate: `1 < q <= 8`, causal, no softcap/sliding-window/q_descale,
head_size ≤ 256, block_table + seqused_k present, waste ≤ 50 %
(`num_actual_tokens*2 >= q*B`), fp8 KV + paged + `_V51_FP8_MQ` (v51
gate). try/except → v33 fallback (counter `v53_l4_fallback`).
`VLLM_V53_L4=0` restores v1.2.5 routing exactly.

### P-1 — fp8 route-mix observability (REPORT §3.1/§6)

`VLLM_V125_P1=1` arms `_v125_p1_route` counters on the four verify/
drafter routes (`v51_uniform`, `v53_l4_ragged_pad`, `v53_l4_fallback`,
`v33_mq3d`): [calls, tokens] per route, logged every 2000 calls.
Answers "how often does the ragged case actually hit" (GMU-gate
hit-rate unknown, REPORT §3.1) with data instead of guesses. Default
off = zero cost.

### L10 — TQ KV-splits tier visibility (REPORT D10 suspect)

`VLLM_V53_TQTIER=1` logs `_effective_kv_splits()` decisions every 512th
call: `max_seq_len base tier_splits eff graph_kv`. Static finding baked
into the comment: the tier ladder (≤49k 2×, ≤131k 4×, else 8×) is capped
at `_tq_max_kv_splits`=1024 with base=256 ⇒ 4× and 8× tiers BOTH land at
eff=1024 — the 131k tier boundary is not a splits cliff in eff value,
weakening the D10 tier suspect. **Outcome (Lane C)**: the runtime log
never fired — the hook sits after the `not _tq_adaptive_splits` early
return (line 594) and `_tq_adaptive_splits` is False under
FULL_DECODE_ONLY graphs (line 523) — but this itself CLOSES D10
structurally: the ladder never executes at runtime on these lanes
(graph-fixed grid 256 + device-side re-partition), so D10's decode
asymmetry cannot be a tier effect. See the Lane C post-mortem above.

### L7 — deep warmup buckets (REPORT §4; extends dt_warmup_v51 p1–p9)

p10 (~131k) + p11 (~227k) single-shot prefills w/ 24 tok decode,
`ignore_eos`, `temp=0`. Filler is a seeded VARIED record generator
(~8.5 tok/record, 24-word pool) — KNOWN_ISSUES #13: repetitive filler
triggers instant-EOS ≥32k. Warms: chunked-prefill continuation JIT at
the 131k/227k buckets (the `_tq_full_dequant_kv` continuation JIT class
that p6–p8 cover only to 70k), deep decode kernels, AND (Lane A) the
k=1 L2 path. Opt-out `V53_DEEP_WARMUP=0` (p1–p9 unchanged).

## Re-scoped: L6 — TQ continuation prefill O(n²)

Original idea: reuse the dequantized-prefix workspace across chunks.
Convicted memory-infeasible: the continuation path dequants the whole
cached prefix per LAYER per chunk into a shared scratch buffer —
content differs per layer, so cross-chunk reuse needs the workspace
pinned per layer (~65k tok × 2 (K+V) × hs × layers ≈ tens of GB at
262k). Real fix = L1-class in-kernel dequant prefill (Triton kernel
dequantizing KV tiles inside the prefill attention loop) — same tier as
L1/L8 (High effort, kernel work), documented here, NOT attempted in t1.
L7 (bucket warmup) hides the JIT part of this cost at boot.

## Validation results (bench126, 2026-09-07 — actuals)

Runner: `/root/build/run_lane126.sh <tag> <kv> <spec> [gmu] [maxlen] [env] [img]`
— same battery as v125 audit (boot → warmup v53 → dt_probe2 battery →
ctxscan 2k–65k → deep 115k/205k words → conc8@2k / conc4-mixed /
conc2@65k → health tail incl. v53 counters). Lane A phases 2+ ran via
`l2a_phases.sh` (live-server resume, no reboot). Host copies of all
`.sh` CR-stripped (`sed -i 's/\r$//'` — MSYS CRLF breaks bash).

### Lane A `l2a_mtp4_tq4` (mtp4 + tq4nc, 0.9/262144, t1 then t2 hot-swap)

- Boot t1 13:09, L2 armed both workers (`VLLM_SPEC_CTX_K=65536:1`).
  Battery after calibrated warmup: 6/6 SHAs bit-exact vs v125 refs
  (3cecc747 / c87e27c4 / 25058c3d / 744e88f5 / 252b4dc1 / ebcc8258),
  ACCEPT 0.612 vs 0.613, p6 compl 131 vs 118 = documented #18 knife-edge.
- ctxscan 2k/16k/32k/65k = 61.8 / 38.5 / 28.6 / 21.0 (refs 61.8-class
  / 38-class / 28-class / 21.0 — unchanged). Deep 136k/242k decode
  11.7 / 6.1 — IDENTICAL to refs + 0 fire lines → t1 L2 never fired
  (dead `_seq_lens_cpu` read, see post-mortem above).
- t2 hot-swap + reboot: warmup fired `k=1 (src=ub)` on both workers →
  17 s later TP wedge → `sample_tokens` RPC timeout (exactly 300 s) →
  EngineDeadError (crash log `bench126/b_l2a_crash.log`). Conviction
  recorded above; t2 gate-closed rerun = battery bit-exact, inert.
- Conc: c8 122.5 / cmix 61.0 / c65k 4.1 (v125 mtp4_tq4 class: c8 ~118 /
  cmix ~60 / c65k ~4). 0 STALLS / 0 ERRS.

### Lane B `l4b_mtp4_fp8` (mtp4 + fp8_e4m3, 0.9/262144, t2, `VLLM_V125_P1=1`)

- Battery 6/6 SHAs bit-exact vs v125 fp8 lane (83dd8aaa / c87e27c4 /
  47dfb71b / 6765850d / f61457ef / 724d7401), ACCEPT 0.572.
- **P-1 verdict: `v51_uniform=2000 (181245 tok)` on both workers after
  the FULL lane (single-stream + conc8 + cmix + c65k) — 100 % uniform,
  0 ragged, 0 fallback.** Whole-step scheduling makes verify uniform by
  construction on static-width spec lanes → L4 pad is a no-op there
  (kept as safety net for variable-width traffic).
- ctxscan 2k/16k/32k/65k = 31.8 / 15.9 / 14.4 / 12.8 (v125 fp8 refs
  ~31.8 / ~16 / ~14 / **9.6** → 65k **+33 %**, matches post125 warm
  12.5-class). Deep 136k/242k decode 6.6 / 4.4 (ref 6.7/4.4 — flat).
- Conc: c8 66.5 (ref 61.4, +8 %, N=1 noise) / cmix 29.9 / c65k 3.0;
  all ok, 0 err. Health tail: 0 STALLS, 0 ERRS, F8=1 (single firing,
  engine healthy, all requests full-length — v123-cert-class benign),
  JIT=13 (during warmup p10/p11 as designed), L2ARM=0, P1=2.
- Note: Lane B's in-container server log was not preserved past lane
  teardown (Lane C reused the container); verdicts above were captured
  live from the running lane.

### Lane C `tierc_df7_tq4` (dflash k7 + tq4nc, 0.9/262144, t2, `VLLM_V53_TQTIER=1`)

Boot 15:03:21 → BOOT_OK 15:06, WARMUP_OK 15:13 (p10 79.6 s, p11 150.2 s),
LANE_DONE 15:22:11.

- Battery: 5/6 SHAs bit-exact vs v125 df7_tq4 refs (3cecc747 / c87e27c4 /
  744e88f5 / 194e1de8 / 1f9c461b; ACCEPT 0.476). **p3_para is a run-to-run
  knife-edge cell on this lane**: ref 000f64ab (322 tok), lane run 39aa3944
  (313), immediate rerun 1b5558c3 (307) — three distinct draws on the same
  image = #18-class (v42.1 documented the same on dflash lanes); the v125
  ref was itself a single draw. Not a v53 regression.
- ctxscan 2k/16k/32k/65k = 46.8 / 30.2 / 18.4 / 16.6 (refs 46.9 / 32.8 /
  18.3 / 14.3 — 65k +16 %, N=1).
- Deep 136k/242k decode **4.2 / 7.3 — identical to v125 refs**: L7's
  122k/198k buckets did NOT close the D7 first-deep-cold gap (see L7
  verdict). The 4.2→7.3 cold→warm asymmetry reproduces exactly with the
  buckets pre-warmed → the asymmetry is not a warmup-fixable JIT.
- Conc: c8 93.7 (ref 93.4) / cmix 34.2 (ref 35.5) / c65k 4.1 (ref 4.1);
  all ok, 0 err. Health tail: STALLS=18 (v125: 17 — same class), 0
  STRIKES, 0 ERRS, TIER=0 lines.
- **L10 post-mortem**: `V53_TQTIER` never logged. Root cause:
  `_effective_kv_splits` (turboquant_attn.py:594 in the overlay) early-
  returns `self._tq_graph_kv_splits` when `not self._tq_adaptive_splits`
  — and line 523 sets `_tq_adaptive_splits = adaptive_env == "1" and not
  full_capture` → **False under FULL_DECODE_ONLY graphs** (the deployed
  mode; boot logs "graph-fixed KV splits: grid=256, device-side split
  masking"). The L10 counter sits AFTER the early return → unreachable.
  Fixing the placement (logging before the return) would only ever print
  the constant `graph_kv=256` — zero information. **D10 disposition:
  closed structurally** — the tier ladder is bypassed entirely at runtime
  on these lanes (graph-fixed grid + device-side re-partition), so the
  df7 128k-vs-227k decode asymmetry cannot be a KV-splits-tier effect;
  combined with the static collapse (4×/8× tiers → same eff=1024 when
  the ladder IS active), the D10 suspect is retired. The remaining D10
  candidates live elsewhere (dflash draft window vs 2048-token mamba
  blocks; graph-capture bucket gaps at deep ctx).

### L7 (deep warmup buckets) — verdict from Lanes A/B/C

Calibrated buckets ran at boot on all three lanes (tq4: p10 80.6 s,
p11 158.1 s; fp8: 117.0/224.7 s; df7: 79.6/150.2 s). Benefit is **mixed**:

- tq4 65k ctxscan: 21.0 → 21.0 (no change — p7/p8 already covered the
  ≤70k buckets; p10/p11 only add 122k/198k).
- fp8 65k ctxscan: 9.6 → 12.8 (+33 %, matches the post125 "warm 12.5"
  class — but N=1, and see below).
- fp8 deep decode (different text at depth): 6.6 vs 6.7 ref — NOT
  warmed. The post125 "warm 12.5" fp8 deep number was same-prompt
  prefix-cache-specific, not transferable to different-text deep
  prompts; the fp8-MQ deep-decode specializations keyed to prompt
  content/bucket do not transfer via varied-filler warmup.
- df7_tq4 cold-128k decode (4.2 vs 7.3 warm, D7): **Lane C answers —
  NOT warmed either.** With p10/p11 pre-warmed the asymmetry reproduces
  exactly (4.2 then 7.3), so D7's first-deep-cold is not a
  warmup-fixable JIT; its root cause lives elsewhere (suspects: KV-pool
  state after a first deep request, graph bucket behavior at deep ctx,
  dflash/mamba interaction). D7 stays open, deprioritized.

References (v125 audit, image v1.2.5, `/root/build/bench125/`):
mtp4_tq4 battery SHAs above; deep 136k/242k 11.7/6.1 (cold==warm);
fp8A warm deep 12.5/5.1; df7_tq4 deep 4.2/7.3 (cold==warm — the
post125 warm re-run collapsed TTFT via prefix cache but decode was
identical 4.2/7.3); conc8 ~283 tps (v123 cert, mtp4 lane).

## Prod-restore event (2026-09-07 15:26–15:31) — stock-v1.2.5 GuC-reset class

The first prod-restore boot (v1.2.5, mtp4+tq4nc, 0.9/262144, NO v53
knobs — env verified) died during warmup p8's deep chunked prefill:
Worker_TP0 hit instant `RuntimeError: level_zero backend failed with
error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` at
`num_accepted_tokens_event.synchronize()` (gpu_model_runner.py:4254) at
15:31:42, ~1 min into p8 (request at computed=88,063, v52f mamba fixups
healthy, draft=1 tracked); NO jit_monitor/DFLASH_STALL line preceded it.
EngineCore then hit the 300 s `sample_tokens` RPC timeout →
EngineDeadError. Evidence: `bench126/prod_restore_gucreset.log` (623
lines). Class = the documented intermittent xe GuC-preempt-watchdog /
engine-reset lineage (crash-2 v43b, crash-4 v52), NOT the L2 wedge
(zero device errors there vs instant DEVICE_LOST here — see the
signature note in §L2). First occurrence on this exact config+script
combo; ~7 same-day boots (Lanes A×2/B/C + v125 master/post legs)
passed p8 unharmed. Retry boot (15:41) succeeded: HEALTH OK after
~160 s, warmup p1–p9 clean through the same p8 window, WARMUP_DONE
15:49:25, battery 6/6 SHAs bit-exact vs certified refs
(3cecc747/c87e27c4/25058c3d/744e88f5/252b4dc1/ebcc8258, ACCEPT 0.613,
p6=118) — **prod restored: v1.2.5, mtp4+tq4nc, 0.9/262144, warm,
standing on a bit-exact battery.** If the reset recurs, candidates:
GuC watchdog tuning, prefill chunk-window pacing at deep ctx, or
driver level (out of v53 scope; log class matches the v52 crash-4
dossier).

## Rollback

- Env-only: `VLLM_SPEC_CTX_K` unset → L2 inert; `VLLM_V53_L4=0` →
  exact v1.2.5 fp8 routing; P-1/TQTIER default off.
- Image: prod re-tags `llm-scaler-exp:v1.2.5` (untouched).

## Post-validation disposition (2026-09-07)

- **No promotion.** v1.2.6t2 = v1.2.5 behaviorally (all levers measured
  inert or default-gated; battery SHAs bit-exact on both spec lanes).
  The Lane A promotion precondition (deep ≥ nospec-class ~17) is
  UNREACHABLE via proposer-side narrowing — L2 convicted (wedge);
  L2b (scheduler-side width) is the only sound route and is follow-up
  work.
- Prod stays on **v1.2.5** (restored after Lane C); t1/t2 remain test
  tags. `VLLM_SPEC_CTX_K`, `VLLM_V53_L2_UNSTABLE` unset in prod; P-1 /
  TQTIER observability knobs available on t2 only.
- L4 stays default-on (harmless; safety net for variable-width traffic
  — dflash2 confidence truncation, future L2b). P-1 answered §3.1:
  ragged does not occur on uniform-width spec lanes.
- **D10 CLOSED** (KV-splits tier is not the df7 deep-decode asymmetry
  cause — ladder inactive at runtime under FULL_DECODE_ONLY graphs;
  see §L10/Lane C). **D7 stays open, deprioritized** (first-deep-cold
  reproduces exactly with p10/p11 pre-warmed — not warmup-fixable).
- L7: keep `V53_DEEP_WARMUP=1` default (cost ~4-8 min boot; mixed
  benefit — fp8 65k +33 %, deep-decode JIT for same-bucket shapes).
- L1/L8 (in-kernel dequant prefill; MQ3D→SPLITS parity work) and L2b
  remain parked items — see REPORT §4 + §L2b above.
