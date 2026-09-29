# v60 — WEDGEFIX: async scheduling default OFF + spec draft-barrier default ON + sync zombie guard + no-draft degenerate-row sanitize

`patch_wedge_v60.py` — four-needle patcher applied at boot (and baked into
images ≥ `llm-scaler-exp:v1.2.15`). Root-cause response to the 2026-09-21
Claude-Code-intensive crash loop on `llm-scaler-exp:v1.2.14`:

```
0 tok/s → v55.3 fast-clean SIGKILL → crash-loop   (+ xe ccs/bcs engine
resets on both tiles; NOT OOM — zero oom-kill records, host 6G/376G used)
```

## What it changes

- **WEDGEFIX-A — `vllm/platforms/xpu.py`**: `async_scheduling` is forced
  `False` on XPU unless research env `VLLM_XPU_ALLOW_ASYNC=1` (default `0`).
  The fork's AsyncScheduler default was silently ON in every prior boot
  (including the "async OFF"-labelled §24 K set, whose label came from a
  serve-flags grep, not a runtime check). The async event pipeline is the
  amplifier that turns a dead spec-drafter stream into a full-engine
  0 tok/s stall.
- **WEDGEFIX-B — `vllm/v1/worker/gpu_model_runner.py`**:
  `VLLM_XPU_SPEC_DRAFT_BARRIER` default `0`→`1` and
  `VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX` default `0`→`0` (**every
  step**). Restores the `torch.xpu.synchronize()` drain before drafter
  collectives (the v2x-era oneCCL wedge mitigation that v37 turned off
  under a short-context posture). First cut used `MIN_CTX=8192`, but
  drill 3 (2026-09-21) wedged on a request at `computed=3994` — below
  the gate, so the barrier never fired for it; the default is now
  unconditional. Explicit `VLLM_XPU_SPEC_DRAFT_BARRIER=0` restores the
  v37 posture for diagnosis (cost of the per-step drain: the v33-era
  ~10-14% short-ctx decode tax — accepted for crash-free priority).
- **WEDGEFIX-C — `vllm/v1/core/sched/scheduler.py`** (added after the
  stage4 soak crash): the async flip in WEDGEFIX-A silently disabled the
  AsyncScheduler zombie reapers (v52m zero-emission strike-out + F8v2
  force-finish live ONLY in `async_scheduler.py` — the sync
  `sched/scheduler.py` had zero llm-scaler marks). ~2 min into the
  2026-09-21 14:21-stage4 soak (3× 12k-30k tool-call workers), a
  crash-4 NaN-zombie request (onset ~11k ctx) stayed scheduled with
  empty emissions until its poisoned rows hit an unclamped worker
  gather: `IndexKernelUtils.h:63 "vectorized gather kernel index out of
  bounds"` SYCL assert storm → VllmWorker-1 native abort →
  EngineDeadError (0 engine resets; v55.3 never fired; evidence
  `serve_v60_soak_oob.log` + `v60_soak_crash1_oob.log`). WEDGEFIX-C
  ports the v52m semantics to the sync scheduler: after
  `VLLM_V60_ZOMBIE_STRIKES` consecutive poisoned/empty sampled rows —
  **default 1 = immediate** (crash-3, 14:39 soak, proved the fatal
  gather fires the step AFTER the first sentinel row, so strike-2/3
  never accrue; the strike-out rides the same `update_from_output`
  that first saw the sentinel) — or a runaway
  `num_output_tokens > max_tokens + spec_width + 8`
  (`VLLM_V60_RUNAWAY_GUARD=0` disables) — the request
  is FINISHED_ABORTED with the terminal output injected into the same
  step's client buckets (engine stays up). Structured-output requests
  are exempt (grammar-masked bonus rows can legitimately resolve
  empty); prefill chunks and finished requests are skipped; the wrapper
  never raises. Poisoned-row detector: any sampled id ≥ vocab_size
  (healthy rows never contain them; −1 rejection tails are normal
  padded transport) — the MTP-lane zombie signature (v52n) keeps RAW
  rows non-empty, so the async-era empty-row detector alone was inert.

## What it does NOT change

- **Speculative decoding (MTP ×4) stays fully supported** — v60 keeps the
  drafter; it removes the async amplifier and drains the device before
  drafter collectives. (Standing directive: spec support on all images,
  always.)
- **XGrammar-2 untouched** — guided decoding stays on xgrammar 0.2.7
  (XGrammar-2); the patcher's verify gate pins it.
- **WEDGEFIX-E — `vllm/v1/worker/gpu_model_runner.py`** (v60g; added
  after the stage4-T2 structured-output regression): no-draft
  plain-decode steps on the MTP lane (drafter proposes 0 tokens, e.g.
  after rejection-tail steps; scheduler then runs a bare 1-token step
  with `spec_decode_metadata=None`) reach sampling with UNMATERIALIZED
  deferred state. Mechanism (instrumented 2026-09-21, serve_full.log
  468-492): the deferred `tlogits` gather
  `sample_hidden_states = hidden_states[logits_indices]`
  (gpu_model_runner.py ~4732) reads rows the nospec
  `FULL_DECODE_ONLY` graph never wrote — `sample_hidden_states` NaN,
  deferred logits zeros — and the XPU `xpu_topk_topp_sampler` op
  deterministically returns **token 0** for degenerate rows (offline
  repro3: only all−inf/NaN rows do this; the op itself is correct).
  Strict-JSON grammars reject token 0 (`backend_xgrammar.py:158
  Failed to advance FSM` → scheduler `grammar rejected tokens [0]` →
  FINISHED_ERROR) → **deterministic HTTP 500 on every
  structured-output request whose lifetime contains a no-draft step**
  (XGrammar-2 regression surfaced by the async flip: async always runs
  the spec/verify pipeline, so the async lane never hit it; plain
  completions on the same step class silently sampled token 0).
  First fix cut (v60e) recomputed logits from
  `sample_hidden_states` — INSUFFICIENT: those hidden states are NaN
  too. v60g supersedes it: AFTER `apply_grammar_bitmask`, NaN entries
  are replaced with 0.0 so the sampler picks uniformly over
  GRAMMAR-ALLOWED tokens (the −inf mask survives) — schema-legal
  token, no FSM reject, no 500, XGrammar-2 crash-free. Graph wiring
  (fork nospec FULL_DECODE_ONLY output materialization) is NOT fixable
  from a Python patcher — tracked as a known limitation: requests in a
  heavy no-draft stretch may emit whitespace runs and hit the length
  cap (schema-valid, non-fatal). Decode-only steps (all scheduled
  tokens == 1) are touched; prefills stay stock.
  `VLLM_V60_NOSPEC_LOGITS_FIX=0` disables. First 10 firings log
  `llm-scaler v60g NO-DRAFT DEGENERATE-ROW SANITIZED`.
- All v1.2.12 crash-free-certified improvements are carried (f15b,
  arstage, v55.3, v58_p1, STALFIX, v52f) — see the stage gates below.

## Verification gates (all must hold on any v60 boot)

1. `grep -c "F8 guard" /root/serve_full.log` == 0 (AsyncScheduler never
   instantiated — the runtime async proof). NOTE: exactly ONE
   `Asynchronous scheduling is enabled` line may remain, printed by the
   APIServer at config-construction time *before* the platform check
   flips the flag — cosmetic, not runtime. Workers/EngineCore must NOT
   print it.
2. `python -c 'from vllm.v1.worker import gpu_model_runner as g;
   print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_MIN_CTX)'` →
   `True 0` (barrier every step).
3. Pedigree marks ≥1 each: `llm-scaler f15b` + `llm-scaler v55.3`
   (gpu_model_runner.py), `llm-scaler arstage` (xpu_communicator.py),
   `llm-scaler v60` (xpu.py + gpu_model_runner.py), `llm-scaler v60c`
   (sched/scheduler.py, ×5), STALFIX (responses/serving.py),
   `_stalfix_name_emitted` (qwen3xml_tool_parser.py).
4. xgrammar version == 0.2.7.
5. `/health` == 200.
6. T2 structured-output regression (WEDGEFIX-E): `t2_xgrammar_probe.py`
   + `t2_thinking_probe.py` (variant A `enable_thinking: false` +
   strict json_schema = the deterministic-500 reproducer) — every
   request HTTP 200, zero `grammar rejected tokens` /
   `Failed to advance FSM` lines, JSON parses on stop-finishes;
   `llm-scaler v60g NO-DRAFT DEGENERATE-ROW SANITIZED` lines present
   on both TP workers (the safety net firing).

## Evidence (2026-09-21, ainode01)

- Control drill (v1.2.14, async ON, barrier OFF):
  `/root/build/lce1/v60_control_drill.log` — wedge in ~6 min of sustain
  round 2, v55.3 KILL 11:30:55 (`async_output_copy event`, 30 s bound),
  f15b ring tail = #11 fingerprint (ARs 25600 → silence,
  `propose_gpu_done` PENDING), generation 38 → 11.5 → 0.0 tok/s,
  +4 xe Engine resets (37→41), health 000.
- Drill 3 (async OFF + barrier @8192) — **wedge persists, gate inert**:
  `/root/build/lce1/v60_drill_at8192.log` +
  `v60_drill3_forensics.log` + `v60_analyze.log` + `serve_v60_drill3_dead.log`
  + `v60_drill3_dumps/` (f15b ring/pyspy/xpusmi). Rounds 1-2 clean;
  wedge at 12:16:46 on request `chatcmpl-97e1b457…` at `computed=3994`
  (< 8192 → barrier never fired). Death: workers 604 s stuck inside
  `sample_tokens` (step watchdog kills), EngineCore `shm_broadcast` RPC
  timeout → clean EngineDeadError shutdown (async OFF changed the death
  signature from v55.3 SIGKILL to fast engine death — ~3× longer
  survival than control). f15b ring: drafter ARs all DONE then silence;
  dmesg ccs+bcs reset pairs on both tiles. Boot-time stall at 11:54 with
  the same AR-DONE signature (recovered) — trigger is probabilistic per
  step, NOT context-size-gated.
- Full forensics: `/root/build/lce1/killan_evidence{1..5}.log`,
  `serve_kill_1056.log` (preserved dead-engine serve log), WD_crash_*
  devcoredumps.
- **Stage4 soak crash → WEDGEFIX-C** (2026-09-21 13:45:33, pre-C lane
  = v1.2.14 + A/B only): `serve_v60_soak_oob.log` (worker stderr:
  IndexKernelUtils.h:63 vectorized-gather OOB assert storm),
  `v60_soak_crash1_oob.log` (soak harness view). Death chain:
  NaN-zombie at `chatcmpl-9bdd1903…` (computed=11264, out=50, KV 6%)
  scheduled with empty emissions for ~14 s → SYCL assert → VllmWorker-1
  abort (multiproc_executor.py:283) → EngineDeadError. Sync
  `sched/scheduler.py` had NO zombie guard (async-only v52m/F8v2),
  though `sched/async_scheduler.py` (the v52m carrier) and core.py's
  v52 flush (AsyncScheduler.flush_pending_finishes) were present but
  dormant — engine-side, not scheduler-side. Fix = WEDGEFIX-C;
  post-fix soak re-run: `/root/build/lce1/v60_soak.log` (T5 census:
  `v60c STRIKE-OUT` / `v60c ZERO-EMISSION` / sycl-assert counters).
- v60 boot verify: `/root/build/lce1/v60_gatecheck.log`,
  `v60_pedigree.log`, `v60_drill_b0.log`, `v60_soak.log`.
- **Stage4-T2 structured-output regression → WEDGEFIX-E arc**
  (2026-09-21 17:48-18:30): root-cause chain — pre-fix deterministic
  500s (`/root/build/../tmp` evidence: `/tmp/t2fix.txt`,
  `/tmp/repro2_out.txt`, `/tmp/repro3_out.txt` = offline op sweep
  proving only degenerate rows return 0; `/tmp/think_out.txt` =
  discriminator probe; `/tmp/v60fearly.txt` + `/tmp/around.txt` =
  serve_full.log lines 468-492 instrumented failure: healthy spec
  steps #6/#7 (amax 8656/220) → first no-draft step `nan_any=[True]`
  amax=1 → op out=[0] → FSM reject → Terminating). First cut v60e
  (recompute) applied + disproven; v60f TEMP sampler instrumentation
  (`dbg_sampler_v60f.py`, backup `.v60fbak`) reverted before bake.
  Post-v60g verify: `t2_xgrammar_probe` 200×4 (run1 cold client
  timeout, no server error; 3× 200 stop-finish VALID_JSON),
  `t2_thinking_probe` A/B/C/D ×2 all 200, A = VALID_JSON both runs
  (23.2 s / 2.9 s), 20 `SANITIZED` log lines (10 × 2 workers),
  zero FSM rejects, sync guard armed 18:21:22, 4× `200 OK` POSTs on
  the new boot. Local extracts: `xi.txt` (FULL_DECODE_ONLY /
  logits_indices / f15b site map), `xj.txt` (execute_model postprocess
  + `_prepare_inputs` non-spec path + `return None` deferred store).
- **Barrier A/B tok/s (2026-09-21, user-requested)**: identical probe
  (`v60_ab_probe.py`, 3 reps × 3 shapes, temp 0.6, thinking off, MTP ×4
  active both phases) — `/root/build/lce1/v60_ab_on.log` /
  `v60_ab_off.log`, driven by `v60_ab_barrier.sh`. Medians:

  | shape | barrier ON | barrier OFF | OFF vs ON |
  |---|---|---|---|
  | short decode (25-tok prompt, 512 out) | 50.6 | 66.2 | **+31%** |
  | ~2k ctx (2850-tok prompt, 512 out) | 39.2 | 42.6 | +9% |
  | conc-4 aggregate (4×2k) | 87.9 | 97.6 (means flat: 86.4 vs 86.2) | +11% |

  Spec mean acceptance unchanged (2.70–2.96 both phases; draft
  acceptance ~43–49%) — the drain costs step time, not draft quality.
  Matches the v33-era ~10–14% mid-ctx tax, worst on very short decode
  (full drain every step). Accepted: crash-free beats the tax; barrier
  OFF remains diagnosis-only (`VLLM_XPU_SPEC_DRAFT_BARRIER=0`).

## Boot integration

Inserted into the `repro_bootV1212.sh` patcher chain after `stall_v59`:

```
docker cp /root/build/patch_wedge_v60.py lsv-test:/root/patch_wedge_v60.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v60.py
```

Modes: `--apply` (default) / `--check` / `--revert`; backups
`<file>.v60bak`; anchor-count==1 asserted; `py_compile` with auto-restore
on failure.

# WEDGEFIX-F (v61) — replicated drafter: DEAD END (bisect 2026-09-22)

`patch_wedge_v61.py` (archived here) tried to remove the wedge hazard at
the source: TP1-view drafter (`sys.modules` swap during Qwen3_5MTP
init/load), keep-own full-vocab embed/lm_head, `get_top_tokens`
lm_head.tp_size, draft KV block 1024→512 for page unification, barrier
default OFF. Mechanically it worked (acceptance 3.12 healthy, zero eager
draft collectives) — and the 2×2 hardware bisect killed it:

| boot | replication | barrier | result |
|---|---|---|---|
| 5 | ON | OFF | **C4 crash #3**: `UR_RESULT_ERROR_DEVICE_LOST` in target `gdn_attention` eager prefill |
| 6 | ON | OFF | same crash at 3rd C4 (clean host first — deterministic, not #05 host-state) |
| 7 | ON | ON (drain) | same crash — barrier irrelevant |
| 8 | OFF | ON (drain) | 3×C4 clean, stock perf (S1 warms 23→49→60) |

Both the crash AND the −40% S1 perf regression (27–33 tok/s stuck;
full-vocab bf16 lm_head ×2/rank + full-width MoE per draft step) AND
+~4.7 GB/rank (gmu 0.8→0.87) come from the replication complex itself.
Crash logs preserved: `/root/build/w61_boot{5,6,7}_crash.log` (md5s
934a7e50…, be3a4dc6…, b10723bf…); perf `w61_b{6,7,8}_perf.txt`.
**Conclusion: stock sharded drafter stays; the barrier is the fix.**
*(2026-09-27 correction: "the barrier is the fix" was an over-attribution
— the v61-boot-5/6/7 DEVICE_LOST crashes were the v88 int32 pool-offset
class [KNOWN_ISSUES #27], which the barrier demonstrably does not
prevent. See the v62 addendum below and
`wedgefix-v75/PATCH_STACK_ANALYSIS.md`.)*

# WEDGEFIX-G (v62) — host-side draft barrier: default since v1.2.16 ("ROOT FIX" attribution superseded by v88 — see addendum below)

## Mechanism

The v24 matrix already proved a device-side tiny-gather "barrier"
wedges too (5/8) — it is itself a non-preemptible oneCCL spin. The v60
drain works because it parks both HOSTS at a defined point with empty
queues, but it forfeits run-ahead (−9…−31%). v62 keeps the host
rendezvous and drops the device sync: at the same pre-drafter site,
both TP worker processes meet in a **/dev/shm flock barrier**
(`/dev/shm/llm_scaler_spec_hb_<EngineCore-pid>.{lock,state}`, 2-party
sense-reversing, count+epoch as two int64; 60 s timeout degrades to
unsynchronized rather than hang). Collective submission skew is bounded
at the host; device queues keep their ≤1-step run-ahead; the guc
640 ms preempt window is never approached.

Env map (`VLLM_XPU_SPEC_DRAFT_BARRIER`): `0` off (v37 posture,
unprotected) · `1` v60 full drain · `2|host` **v62 flock barrier
(default in v1.2.16)**. Knobs: `VLLM_XPU_SPEC_HOST_BARRIER_PARTIES`
(default 2), `VLLM_XPU_SPEC_HOST_BARRIER_TIMEOUT_S` (default 60).

Files: `patch_wedge_v62.py` (test variant, boot 9, applied on top of
v61-state gmr) · `patch_wedge_v62b_bake.py` (bake variant for fresh
v1.2.15-state files: adds G2 = def-block default flip `"1"`→`"2"`;
G0b machinery; G1 dispatch).

## Boot-9 evidence (2026-09-22, REPLICATED_DRAFT=0 + BARRIER=2)

- Perf vs boot-8 drain reference (same probe, 3 reps): S1 57.9→65.9 vs
  49.2→60.0; L2 41.3/45.0 vs 39.3/39.6; C4 agg 83.7/102.7/86.0 (avg
  92.3) vs 75.7/87.4/63.2 (avg 75.4, **+22%**). Every shape faster.
- Wedge drill (3×14-phase sustain, ~33 min): `SUSTAIN_COMPLETE_NO_WEDGE`,
  fence 0, engine resets 0→0, v55.3 kills 0, health 200 throughout,
  p14 cycles 5/5 PASS ×3.
- Stage4 soak+battery: done=105 tool_ok=105 err=0 (908 s); T1 litellm
  tool_use OK; T2 XGrammar-2 VALID_JSON stop; T3 47.3 tok/s MTP active;
  T4 clean; T5 strikeouts=0 zeroem=0 sycl_asserts=0.
- XGrammar ×5 + thinking A–D ×2 all HTTP 200; zero barrier-timeout
  warnings; `/dev/shm` barrier pair live and advancing.

## v1.2.16 bake (`stage5_bake_v1216.sh`)

Baked from a FRESH v1.2.15 container (never from the v61-test lane) +
`patch_wedge_v62b_bake.py` only: image `llm-scaler-exp:v1.2.16`
(6f883b07c526, 22.7 GB). 20/20 content gates OK (v62 marks ×3; v61
absent; v60 pedigree ×10 incl. WEDGEFIX-E + STALFIX + v60c;
`barrier_default=False True 0`; xgrammar 0.2.7; v1215 pedigree marker;
baked serve gmu0.8 bs64 e4m3 pc-ON spec-ON async-OFF). Fresh V1216
boot: HEALTH_OK ~140 s, SANITY `'READY-v1216'` stop, engine-side async
disabled, live env BARRIER=2, barrier pair `hb_459` active, XGrammar-2
clean, C4×1 S1 64.9 / L2 33.7 / C4 agg 105.2 (best C4 yet). Watchdog
lineage `repro_bootV1212.sh` repointed (backup `.pre_v1216`); spec ×4
+ XGrammar-2 supported and crash-free, per standing directive.

## v62 addendum (2026-09-27) — "ROOT FIX" superseded by v88; is the barrier still needed?

The v88 investigation (`wedgefix-v75/RCA_TINYSTEP_WEDGE.md` P24n–P24v,
KNOWN_ISSUES #27) proved the era's dominant wedge was the **int32
pool-offset overflow in the GDN conv kernels** (live since the v26-era
wheel `d20260830` — i.e., throughout the v60/v62 record above), fixed by
the v88 int64 wheel shipped in v1.2.21. Against that class **the draft
barrier is demonstrably inert**: the v61 bisect crashed identically with
the drain ON (boot 7); v1.2.20 with `BARRIER=2` died deterministically at
serialized cycle 17; the v86 fault capture (layer-16 prefill, ssi=4173)
was taken on a barrier-2 lane. The boot-9 clean drill was pool-id-
lifecycle fortune, not causality — and the barrier's original v2x
justification (#11) had already been fixed BY CONFIG at v31.1
(`TORCH_COMPILE_DISABLE` for spec+TP2, still active in v1.2.21).

Post-v88 posture (full analysis: `wedgefix-v75/PATCH_STACK_ANALYSIS.md`):
keep `VLLM_XPU_SPEC_DRAFT_BARRIER=2` standing — µs-class host-side
insurance, zero observed firings or faults, and v1.2.21 is certified WITH
it — never `1` (standing user constraint; strictly dominated by mode 2),
and retire `2` only via a controlled barrier-off validation leg
(PATCH_STACK_ANALYSIS.md §3.6) if and when live lane changes are
permitted and it shows a measurable improvement.


# TTFTFIX (v63) — adaptive chunked-prefill budget under decode
contention: v1.2.17

## Problem (Phase-0 RCA, 2026-09-22)

CC intensive usage: ~136k-token prompts, zero prefix reuse (head drift
— reuse itself proven healthy end-to-end at 94-98.8% hit rate, warm
TTFT 1.7-2.7 s), serialized 7168-token chunked prefill with GDN
attention stretching chunk steps 4 -> 9 s. Every co-running decode
request advances once per scheduler step: all other sessions freeze at
0.1-0.5 tok/s for the whole prefill (reproduced: 0.255 tok/s, gaps up
to 8.25 s; cold 106k TTFT 101.8 s).

## Fix

`patch_sched_v63_bake.py` (two needles in `v1/core/sched/scheduler.py`,
identical to test variant `patch_sched_v63.py` except baked default):
when >=1 RUNNING request is in decode phase and the step is shared,
`token_budget` is capped to `VLLM_V63_CONTENDED_BUDGET` (default 2048;
floor-clamped to one mamba block 1024 — smaller budgets yield
zero-token block-aligned chunks and would stall the prefill; 0 =
stock). Solo prefills keep the full budget; pure-decode steps are
unaffected (min() clamps). One-time `V63_TTFTFIX_ACTIVE
contended_budget=N running=M` INFO line in serve_full.log.

## A/B evidence (probe_fair_v63, 106k cold prefill + concurrent decode)

| budget | starved decode | big TTFT | gaps during |
|--------|---------------|----------|-------------|
| stock 8192 | 0.255 tok/s | 101.8 s | 6.7-8.25 s |
| 4096       | 0.433 tok/s (+70%) | 106.2 s (+4%) | ~3.5 s |
| 2048       | 0.998 tok/s (3.9x) | 118.0 s (+16%) | ~1.4 s |

2048 baked (starvation is the named issue; every halving ~ doubles
concurrent-decode fairness at ~10% prefill cost). Battery on 2048:
S1 52.8/51.9/67.9, L2 44.3/46.8/44.9 (parity), C4 agg 87.5 (band
75-105), XGrammar-2 x3 + alts + thinking A-D all 200 VALID_JSON,
preemptions 0, 3x14-phase wedge drill SUSTAIN_COMPLETE_NO_WEDGE.

## JIT warm (same bake)

`probe_jitwarm_v63.py` runs ONE ~130k cold prefill + sampled decode
inside the bake container before commit (lane must be down; bake serves
on :8000) and stamps `/root/.v63_jit_warmed`. Kills the 3-5 s
first-big-turn kernel_unified_attention / reduce_segments / topk_topp
compile spikes on every fresh boot; torch JIT cache persists in the
image layer. Boot script stamp-gates a re-warm as safety.

## Files

`patch_sched_v63.py` (test), `patch_sched_v63_bake.py` (bake, default
2048), `probe_fair_v63.py` (fairness A/B), `probe_jitwarm_v63.py`
(JIT warm), `stage5_bake_v1217.sh` (gates + commit + boot script).
Spec MTP x4 + BARRIER=2 + XGrammar-2 0.2.7 untouched and crash-free,
per standing directive.


# TTFTFIX-2 (v64) — decode-step interleave + budget-1024 default: v1.2.18

Task: from certified v1.2.17, further minimize co-session starvation during
big chunked prefills (user directive: MAXIMUM starved-decode tok/s).

Two scheduler needles on top of the baked v63:

1. **Budget default 2048 -> 1024** (N3 in `patch_sched_v64_bake.py`): at one
   mamba block, the aligned chunk usually floors to 0 after decode tokens
   are deducted — prefill naturally yields most steps; decode runs
   near-continuously (0.06 s gaps). Env `VLLM_V63_CONTENDED_BUDGET`
   unchanged (0=off, <1024 floors to 1024; 2048 restores v1.2.17 posture).
2. **Decode-step interleave** (`VLLM_V64_DECODE_INTERLEAVE`, default 2):
   every K-th contended step caps the budget to
   `VLLM_V64_DECODE_BUDGET` (default 512, clamped [64,1023] — below one
   mamba block) so the chunked prefill takes the certified
   `num_new_tokens <= 0 -> continue` skip and co-running decode gets a
   dedicated ~0.06-0.2 s graph step. Neutral at budget 1024 (already
   skip-dominated); at 2048 it lifts fairness 1.0 -> 1.8 tok/s. Solo
   prefills and pure-decode steps unaffected.

## A/B evidence (probe_fair_v63, 106k cold prefill + concurrent decode)

| config | starved decode | big TTFT | gaps during |
|--------|---------------|----------|-------------|
| stock 8192 | 0.255 tok/s | 101.8 s | 6.7-8.25 s |
| v63 2048 (v1.2.17) | 1.007 tok/s | 117.1 s | ~1.4 s flat |
| v64 K=3 @2048 | 1.411 tok/s | 121.7 s | 1.4/1.4/0.06 |
| v64 K=2 @2048 | 1.794 tok/s | 125.2 s | 0.06/1.4 |
| v63 1024 no-IL | 6.395 tok/s | 160.4 s | ~0.06 flat |
| **v64 K=2 @1024 (baked)** | **6.361/5.818 tok/s** | 162.6/152.3 s | ~0.06 flat |

~6.4 tok/s = 25x stock and the mechanism ceiling (~40-50% of the decoder's
solo 13-17 tok/s; remaining chunk steps bound it). Probe acceptance is
optimistic (counting task ~5.0); real CC acceptance ~2.4 halves absolute
rates, ratios hold: ~3-3.5 real tok/s vs ~1 at the 2048 posture — for
multi-session CC use that is the difference between alive-streaming and
looks-hung (see KNOWN_ISSUES #23). Cost: contended big-turn TTFT +50-60%
vs stock; solo/uncontended prefill untouched.

## mnbt 16384 REJECTED (run D)

Solo cold 106k = 112.7 s at mnbt 16384 vs ~102 s at 8192. Chunk splitting
is FLOP-invariant; 15360-token chunks only move boundary overhead and at
this width the GDN path gets slower (tiling/activation effects). mnbt
stays 8192. C4/S1/L2 parity confirmed at 16384 (no memory issue); the
rejection is purely perf.

## Files

`patch_sched_v64.py` (test), `patch_sched_v64_bake.py` (bake; N3 flips the
v63 default to 1024), `probe_solo_cold.py` (solo cold TTFT), `battery_v64.sh`,
`validate_v64.sh` (solo + battery + 3x drill chain), `sanity_v1218.sh`,
`stage5_bake_v1218.sh` (gates + commit + boot script). Spec MTP x4 +
BARRIER=2 + XGrammar-2 0.2.7 untouched and crash-free, per standing
directive.


# v65 Phase-0 — big-TTFT cost model + lever census (post v1.2.18)

Goal (user directive): optimize big TTFT (cold/warm), maximum performance.

## Instrumentation

`patch_v65_probe.py` — TEST-ONLY module-level wrap of `Scheduler.schedule`
(env `VLLM_V65_STEP_LOG=1`, default off, never baked). Gap between successive
schedule() entries = full engine step wall (schedule + execute + process).
This fork logs no per-step batch lines; the probe provides them.

## Measured cost model (solo cold ~118k-token prefill, stock v1.2.18)

Chunk step wall grows linearly in prefix: 4.7 -> 9.3 s across 14 chunks
(slope 3.97e-5 s per prefix-token per 8192-chunk). Decomposition:

- ~66 s constant per-chunk: 48 linear/GDN layers + MoE + chunk-local
  self-attention + fixed step overhead (state COW, launches).
- ~34 s quadratic total: prefix KV-read of the 16 full-attention layers
  (`full_attention_interval: 4`, GQA 24q/4kv, head_dim 256, FA2, fp8 KV).
- First chunk carries a consistent ~0.6 s premium; decode steps 0.11-0.12 s.

## Lever census (all measured, 2026-09-22)

| lever | result |
|---|---|
| mnbt 4096 | 103.08 s — self-attn halves, steps double: cancels |
| mnbt 6144 | 102.06 s |
| mnbt 8192 (kept) | 102.3 s — flat plateau 4096-8192 |
| mnbt 15360 | 112.7 s (rejection reconfirmed) |
| pass_config fuse_norm_quant | XPU platform rejects: `xpu.py:544` "not yet supported on XPU and will be disabled" — closed |
| FlashInfer autotune | not exposed on this XPU backend — closed |
| chunked-prefill budget | v63/v64 own this dimension; contended posture is the user-directed max-fairness point |

Conclusion: solo cold is at its engine-side plateau (~102 s). Remaining levers
are deep kernel work (GDN/full-attn prefill efficiency) or client-side warm
hit-rate (CC head stability). Warm path is healthy: 1.7-2.7 s at 94-98.8%
prefix hit on stable heads; drift recompute is bounded by the 4096-token
mamba-align reuse granularity (ESIMD kernel surgery required to tighten).

## Incidents

- **Crash 14:54:28 (run H2):** `UR_RESULT_ERROR_DEVICE_LOST` on Worker during
  solo prefill mid-run (15.9 s wedged chunk), respawn died OUT_OF_DEVICE_MEMORY,
  shutdown drain wedged. Sporadic hardware-class event (same class as the v61
  2x2 bisect crashes); NOT reproducible on stock — seed 206 clean (102.1 s)
  after host reboot on stock config. Fusion flag was platform-disabled before
  the crash, so it never executed. Host reboot = recovery (standing directive).
- **`expand_kernel` Triton JIT spike:** 2.1-2.2 s on the FIRST big-turn decode
  after every fresh boot, INCLUDING baked-image boots (warm cache does not
  cover this shape). Fix queued for next bake: extend in-bake JIT warm to
  exercise the post-prefill decode shape.

## Files

`patch_v65_probe.py` (test-only probe), `v65_p0.sh`/`v65_p1b.sh`/`v65_p1c.sh`
(instrumentation + curve extraction), `v65_runFG.sh` (mnbt grid),
`v65_runH2.sh` (fusion A/B), `v65_seed206.sh` (crash closure).


### v65 Phase-1 — deep-kernel census (Option b): layer attribution + alternative-backend A/Bs (2026-09-22)

Standing order: attempt deep kernel work (GDN/full-attn prefill efficiency —
the only >5% solo-cold lever per Phase-0); on failure, stop at the plateau.
Phase-1 ran the bounded-surgery ladder and closed every rung with data.

**Layer-type attribution (new, measured).** Test-only layer probe
(`patch_v65_lt.py`, env `VLLM_V65_LAYER_LOG`, NEVER baked; appended to
flash_attn.py, wraps `FlashAttentionImpl.forward` for prefill steps and
shadows `torch.ops.vllm.gdn_attention_core_xpu` — both gated >1024 tokens so
decode/XPU-graph capture never syncs). Solo cold seed 210 = 102.63 s
(instrumentation overhead ~0.3 s vs 102.3 stock):

| pool | time | share | verdict |
|---|---|---|---|
| FA2 full-attn prefill (16 layers × 17 steps) | 40.1 s | 39% | kernel-bound; linear growth 0.0422 ms/prefix-token per step (≈345 ms per 8192-token chunk), intercept ≈ −0.3 s → essentially pure q×prefix product work ≈ 41 TFLOP/s/rank |
| GDN core, SYCL `_xpu_C.gdn_attention` (48 layers × 17 steps) | 4.95 s | 5% | **closed — not a target**; ~6.9 ms/layer/8192-chunk, flat per step. Phase-0's guess that GDN sat inside the constant term was wrong in weight: the SYCL kernel is fast |
| residual (MoE + projections + norms + TP allreduce + step overhead) | ~57.6 s | 56% | GEMM-bound, ~3.4 s/step constant; moe_backend='auto' (config/kernel.py:141) with no surfaced XPU alternative |

(17 chunk steps for the ~119k probe: 14×8192 + 3072 + 1458.)

**Alternative-backend A/Bs.**
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` env flip: **dead** — this fork's v1
  selector (v1/attention/selector.py) does not read that env at all; boot
  stayed on FLASH_ATTN (inconclusive null, run P8).
- Forced TRITON_ATTN via test patch (`patch_xpu_force_triton.py`, env
  `VLLM_V65_FORCE_TRITON`, default off, NEVER baked; env-gated branch in
  xpu.py get_attn_backend): force confirmed on boot, then solo cold seed 212
  **timed out at 420 s** (vs 102.6 s) — Triton prefill attention is >4×
  slower at q=8192 / head_dim 256 / GQA 6:1 / fp8 KV. Zero engine errors —
  slow, not broken. **Lever closed.**
- Cascade attention: structurally inapplicable — `use_cascade_attention`
  gates on num_reqs ≥ 8 and a *common* prefix across a batch; a solo big
  prefill (1 request) never qualifies (flash_attn.py:1737).
- FlashInfer GDN prefill backend: CUDA sm90 only (gdn_linear_attn.py
  ChunkGatedDeltaRule) — XPU uses the in-tree Triton/FLA kernels for the
  generic path, but live XPU prefill runs the SYCL op anyway (5% pool).
- `fuse_allreduce_rms` etc.: same XPU pass_config family Phase-0 already
  rejected (platform hard-disable at xpu.py:544); upside <5% even if live.

**Phase-1 conclusion.** Every bounded lever on both kernel-bound pools is
closed with measurement. The remaining theoretical upside — a new
DPC++/XeTLA/ESIMD prefill-attention kernel above FA2's ~41 TFLOP/s/rank, or
deep MoE GEMM work — is weeks-scale kernel surgery in the v61 crash-family
territory, out of bounded-change scope. **Option b failed at the bounded
level; Option a (documented plateau, v1.2.18 standing) is the accepted
posture.** The `expand_kernel` post-prefill-decode JIT warm gap (2.2 s per
fresh boot) stays queued for the next bake that carries a real change.

Files: patch_v65_lt.py (layer probe, test-only), v65_lt_run.sh (attribution
run), v65_lt_tab.py (per-step analysis), v65_p8.sh (dead-env flip + revert),
patch_xpu_force_triton.py + v65_p10.sh (forced A/B + revert), v65_p11.sh
(lane cleanup + stock verification).

### v66 FAIRFIX + v1.2.20 — waiting-admission starvation bypass + Claude Code stack repair (2026-09-22/23)

**Symptom (v1.2.18/v1.2.19).** Multiple clients connect; new requests sit in
Waiting indefinitely while decode sessions run (`Running: 2 reqs, Waiting: 4
reqs, Avg prompt throughput: 0.0 tokens/s`) despite `--max-num-seqs 16` —
clients drop. Reproduced on stock v1.2.18 (probe_admission_v66.py: 3 decode
sessions + 3 CC-shaped ~12k-token clients): `ADMIT_MAX_TTFT 91.20 s,
verdict=STARVED` (7 consecutive Running:3/Waiting:1 stat lines, 12×
`prompt throughput: 0.0`).

**Root cause.** The v63 contended budget (1024 baked) and v64 interleave
(K=2, budget 512) cap the per-step token budget whenever decode-phase
co-runners exist. After decode-token deduction the mamba-block-aligned chunk
floors to 0, so the head of the waiting queue is never admitted while any
long decode runs. The user-directed fairness posture made this an admission
deadlock for new arrivals.

**Fix (v66 FAIRFIX, patch_sched_v66_bake.py).** When the waiting queue has
been continuously non-empty for > `VLLM_V66_PREFILL_STARVE_S` (default 2.0 s
baked, 0=off), bypass BOTH caps for exactly one scheduler step: the head
gets a full-budget (8192) chunk, then the caps resume. The check is
waiting-queue-age based and evaluated once per step (`_v66_bypass_now`),
needled into both the v63 `elif` and the separate v64 `if` — solo prefills,
empty-queue big-prefill crawl, and pure-decode are untouched.

**A/B (same probe, seed 661, warm engine).** stock 91.20 s STARVED → v66
ttfts `[14.37, 5.99, 9.36]`, `ADMIT_MAX_TTFT 14.37, verdict=PASS` (6.3×).
Regression battery with v66: xgrammar ×3+alts 200/PARSED_JSON, thinking A-D
200s, preemptions 0.0, fairness 5.45 tok/s during 106k prefill (gaps 0.06 s
flat — interleave intact), solo cold 100.38 s (v66 inert when queue empty).

**Companion repairs (Claude Code multi-client compat).**
- **litellm was hard-down since the 2026-09-20 21:36 rebuild**: the new
  `main-latest` image requires a DB in `user_api_key_auth`; every route on
  every model 400'd `No connected db.` → all CC traffic failed. Rebuilt
  `litellm-proxy` with `LITELLM_MASTER_KEY=sk-dummy` (db-less master-key
  mode; matches CC's existing `ANTHROPIC_AUTH_TOKEN`, zero client changes).
  glm z.ai / local OpenAI route / anthropic `/v1/messages` all 200 after.
- **This fork streams thinking as `delta.reasoning`** (not
  `reasoning_content`). With the baked serve defaults
  `preserve_thinking:true, reasoning_effort:xhigh`, requests with modest
  max_tokens finished entirely inside unterminated thinking (`finish=length`,
  200 OK, zero visible deltas) → empty completions for clients. v1.2.20
  drops `--default-chat-template-kwargs` from serve_user.sh (reasoning
  parser qwen3 stays; litellm entries still set per-request
  chat_template_kwargs per tier).
- CC thinking round-trip (assistant history WITH thinking blocks, anthropic
  format): all turns 200, correct answers, no echo of old thinking.

**CC deep validation (local CLI 2.1.278 → litellm :4000 → lane, local
models only).** Three concurrent instances, one exercising the Task tool
subagent flow: `SUBAGENT_OK <subagent sentence>` / `PONG` / `391` — all
completed, zero drops or admission stalls.

**v1.2.19 note.** Image 4640ec2449e3 was baked but never certified (sanity
JIT-zero=7 — compile frontier unbounded by enumeration; drill aborted after
GPU engine resets + a real wedge in round 1). Superseded by v1.2.20, baked
from the certified v1.2.18 base: v66 patch + thinking-defaults rollback +
v1219 warm set + NEW sampler-shape round (default/top_p/top_k/combined
2000-tok generations; zero-JIT gate kept on the proven 8-kernel set,
sampler kernels evidence-only). Ship gates include the v66 admission
acceptance (`ADMISSION_DONE verdict=PASS`) on the fresh boot.

**v1.2.20 SHIPPED (2026-09-23 05:09 UTC).** Image `e1d92193a106` (22.7GB).
Ship arc: first bake attempt aborted on a gate miscount (`v66_bypass_v63`
expected 1, got 2 — both v63 and v64 sites insert the identical bypass line;
gate corrected to `v66_bypass_sites 2`, patch itself was never wrong);
second abort: sanity ran `/root/probe_admission_v66.py` without docker-cp'ing
it into the lane (fixed: sanity now copies the probe; resume_v1220.sh resumes
the chain post-boot without re-baking). Final chain all green: bake gates
30/30, fresh boot HEALTH_OK + READY-v1220 + v66 `marks=4 -> APPLIED`,
**admission PASS on fresh boot (TTFTs 14.81/8.38/9.54 s vs 91.2 s starved)**,
solo cold seed114 100.6 s (v66 inert when queue empty), battery clean
(xgrammar valid, preemptions 0), 3×14-phase drill SUSTAIN_COMPLETE_NO_WEDGE
fence-hits=0, fairness regression 6.77 tok/s during 106k prefill with 0.06 s
gaps (v64 interleave intact). Watchdog lineage repointed to V1220
(.pre_v1220 backup), watchdog un-paused, lane healthy, litellm 200. Known
residual, documented not gated: fresh-boot JIT count 7→8 on the sanity/
validate traffic shapes (first-touch compiles outside the warm enumeration —
eagle_prepare_inputs_padded/batch_memcpy plus sampler-class kernels;
steady-state 0.06 s gaps prove no compile stalls in service; same evidence
class as the v65 Phase-0 expand_kernel residual).

Files: patch_sched_v66.py (test), patch_sched_v66_bake.py (bake),
probe_admission_v66.py (repro/acceptance), probe_raw_stream.py (SSE
diagnosis), probe_think_cc.py (CC round-trip), stage5_bake_v1220.sh,
sanity_v1220.sh, validate_v1220.sh, ship_v1220.sh, resume_v1220.sh.

# CC-thinking fix + V1221 durable lineage (2026-09-23, no image change)

Symptom: Claude Code stopped displaying thinking text ("✢ Skedaddling…").
Two stacked defects, both engine-external — image v1.2.20 unchanged:

1. **litellm 1.103.0 Responses-API bridge.** `/v1/messages` for
   `openai/*` deployments (ours) unconditionally routes through
   `LiteLLMMessagesToResponsesAPIHandler` (messages/handler.py:73,
   `_RESPONSES_API_PROVIDERS = {"openai"}`), which drops vLLM reasoning.
   Every downstream hop was proven healthy first (engine `delta.reasoning` →
   gpt_transformation.py:841 rename to `reasoning_content` → adapter
   transformation.py:1582/1642 emits thinking blocks when called directly) —
   only the routing was broken. Fix:
   `litellm_settings.use_chat_completions_url_for_anthropic_messages: true`
   (opt-out read at messages/handler.py:83). Verified required AND sufficient
   on the same-day `main-latest` (3def0387871a): plain config drops thinking,
   fixed config streams 72× thinking_delta (side-container test on :4004,
   removed after).
2. **Template rejects reasoning_effort "high".** CC's thinking param maps to
   `reasoning_effort:"high"`; the stock template only accepts
   xhigh/medium/low → 400. Fix: user's `chat_template_qwen38_high.jinja`
   served via `--chat-template`.

Validation through :4000: stream+nonstream thinking counts, E2/E2B (54
deltas), CC-shape (78 deltas), forced tool, thinking+tool, history replay
with thinking/tool_use blocks, nonthinking tier — all green; real CC runs
show thinking text and working tool flow. Cosmetic residual: usage
`thinking_tokens:0` annotation.

**Durability:** repro_bootV1221.sh = V1212 lineage + template host mount +
boot-time `--chat-template` injection into baked serve_user.sh + fail-loud
gates (exit 11/12); lane_watchdog.sh repointed V1212→V1221 (service
stop/edit/start, backup .pre_v1221); live container's serve_user.sh patched.
Ship-time defect caught in review: the sed-inserted mount line lost its
trailing `\` — docker run would have truncated before the image name on
every watchdog relaunch (invisible to `bash -n`); fixed via awk append +
`diff` review against V1212 (exactly 5 added lines).

Artifacts: ../ccthink-fix/ (template, litellm_config.yaml, repro_bootV1221.sh,
lane_watchdog.sh, probes). Full RCA in KNOWN_ISSUES #26.

# v88 int64 pool-offset fix — tiny-step WEDGE ROOT FIX, v1.2.21 (2026-09-25)

The "wedge crash after prompt changes" family (KNOWN_ISSUES #11 residual,
reopened by multi-client CC traffic) is root-caused and fixed at the kernels
level. Full evidence chain: ../wedgefix-v75/RCA_TINYSTEP_WEDGE.md P24n–P24v;
summary in KNOWN_ISSUES #27.

Root cause: the xe2 prefill conv kernel computes the per-request mamba state
pointer as `conv_states + states_id * conv_states_stride_0` in signed int32
(chunk_causal_conv1d_xe2.hpp:186/547; same at causal_conv1d.hpp:232/452/736/
833). The unified mamba pool pads rows to 1 MiB — stride(0)=524,288 fp16
elements (measured live) — so ids ≥ 4097 wrap: mapped targets get silently
corrupted, unmapped ones UR_RESULT_ERROR_DEVICE_LOST (the wedge; captured
fault: layer-16 prefill, ssi=4173). Deterministic standalone repro:
repro_v88_int32.py replays the captured fault call against a strided
serve-geometry pool — DEVICE_LOST at 4173 pre-fix, exact-row write post-fix,
bit-identical in-range outputs (numerics-neutral).

Fix: `static_cast<int64_t>` at all six sites (the pattern the delta-rule
kernels already used); wheel rebuilt KERNELS_MAX_JOBS=52 (standing for all
kernels builds). Present since the v26-era wheel d20260830 — v26 hardened
the delta-rule kernels but not the conv kernels.

**v1.2.21 SHIPPED (2026-09-25 19:27).** Image `09e114a46903` (24.7GB) =
v1.2.20 + fixed wheel, baked from a fresh lsv-bake with a bake-time
acceptance gate (standalone repro exit 0 inside the bake container before
warm). 50 content gates OK (lineage marks, v84–v87 probe absence,
no-GDN_CAPTURE, serve config). Fresh-boot validation: READY-v1221,
admission PASS, solo cold 101.04 s (parity), serialized 24/24 SURVIVED
(historically dead at 17), bursts 72/72, 3× drill no-wedge, fairness
6.997 tok/s @ 0.06 s gaps, JIT recheck 0, CC round-trip through litellm
green (incl. explicit thinking-param shapes, nonstream + stream, on the
template-less lane — v1.2.20's per-tier litellm kwargs make the
ccthink --chat-template boot additions unnecessary; that lineage is
preserved at repro_bootV1221_ccthink.sh / repro_bootV1212.sh.pre_v1221 —
name collision resolved by renaming ours in place).

Ship-time tooling defects caught and fixed (recorded for the next bake):
pip rejects non-canonical wheel filenames (docker-cp with the original
basename + version-gate the install); generation seds needed BOTH cases
(v1220 AND V1220 markers); patch marks must be `//` comments or the
kernel build dies; and `docker start lsv-test` is required after host
reboots in install legs.

Artifacts: ../wedgefix-v75/ (patch_v88_int64fix.py, repro_v88_int32.py,
v88_int64_fix.diff, v88a–v88d leg scripts, build_vxk_wheel_v88.sh,
stage5_bake_v1221.sh, ship_v1221.sh).

**v1.2.22 SHIPPED (2026-09-27).** Performance release from the v89
decode-speed plan (../perf-v89/) — NO engine-code or wheel change
(v88 wheel `d20260925` inherited and version-gated). Two serve-config
changes over v1.2.21: tool parser **qwen3_coder** (validated all day
under the live CC fleet; qwen3_xml lineage retained in-tree) and
**cudagraph_capture_sizes extended to 85 sizes** (dense 1..16 ∪ 5k,
k=1..64 ∪ legacy extras) = exact-fit spec-verify graphs — solo decode
+3-4 % (71.2-71.4 → 73.6/74.1 tok/s), contention neutral, zero boot
penalty (64 graphs, 31 s, KV unchanged). Scheduler A/B legs REJECTED by
measurement and NOT shipped: V63=512 (77.0/76.8 vs 77.1 baseline) and
V64=3 (77.6 ≈ 77.1) — baked defaults stay 1024/K=2 with the new
`if 0 < V63 < 1024: = 1024` floor guarding the env knob; barrier-off
leg identical (73.3/73.0 vs 73.6/74.1) → **BARRIER=2 kept standing**
(PATCH_STACK_ANALYSIS §3.6 closed with live measurement). Spec MTP×4
and XGrammar-2 0.2.7 intact (standing user requirement).

Two images: `v1.2.22-raw` = `e6735a4c72a2` (the stage5 bake commit,
24.7 GB) and production **`v1.2.22` = `89b17e0b0f8d`** (24.9 GB) — a
docker commit of the fully-validated warm lane carrying the complete
74-entry warm triton cache. Documented exception to the never-bake-from-
the-lane rule: the full 60+ gate battery was re-run against the
committed image itself (gates_v1222_lane.sh, ALL PASS), and the raw bake
is preserved untouched for re-derivation.

Triton first-use semantics established during ship validation (code-level
evidence, triton 3.7): `jit_post_compile_hook` fires on disk-cache LOADS
as well as compiles (compiler.py:274-289 hit-return + jit.py:878/886), so
every fresh boot logs ~11 once-per-boot "JIT compilation during
inference" lines that are cache loads, not compiles. Gate on cache-dir
delta (== 0 on v1.2.22 through the full decisive traffic chain), never
on monitor-line count. Full record: ../perf-v89/PHASES.md P6b-P7,
KNOWN_ISSUES #28.

Validation on ship: fresh-boot sanity+ADMISSION PASS, solo cold rc=0,
genspeed 72.6/73.9/74.0, 4 sampler shapes 200×4, warm_ext replay OK,
resets 0, CC battery green through litellm :4000. lane-watchdog repointed
(repro_bootV1222 content at the V1212 lineage name; `.pre_v1222`
backup). Artifacts: ../perf-v89/ (stage5_bake_v1222.sh, gates_v1222_
lane.sh, decisive_v1222_wb.sh, probe_jit_mech.sh, t_jit_probe.py,
patch_cc_sizes.py).

**v1.2.23 SHIPPED (2026-09-28).** T3 round release (../perf-v123/,
record ../perf-v89/T3_ROUND_REPORT.md + PHASES.md P8-P14) — engine code
unchanged (v88 wheel `d20260925` version-gated again). Three posture
changes over v1.2.22, all under the same v88 wedge-fix lineage:

1. **`--async-scheduling` baked ON** (serve_user.sh, one flag). The
   fork's AsyncGPUModelRunnerOutput path moves the post-verify host
   handoff (parse → IPC → scheduler → launch) off the step critical
   path: 4×1024 aggregate decode **122-158 tok/s vs 72.6-74.0 sync
   (+85-110 %)**, per-stream p50 ~25 tok/s vs 10-15 under fleet
   contention, py-spy Worker_TP0 parse_output 74.6 % → 30.9 %,
   EngineCore ~96 % idle. Solo 1×1024 = 73-74 tok/s (sync parity — the
   win is pipelining, not single-stream). The v89 "~82 tok/s aggregate
   ceiling / bandwidth-bound" verdict was substantially a
   sync-serialization artifact (DECODE_SPEED_ANALYSIS §9).
2. **`VLLM_RPC_TIMEOUT=60000`** (env, bake+boot). NEW ROOT CAUSE found
   by attempt-1's drill death: under async, EngineCore samples via
   `collective_rpc("sample_tokens")` — an RPC that waits behind
   everything queued in the worker; default 10000 ms < heavy-chunk
   queued latency → `TimeoutError` → EngineDeadError (dmesg clean, NOT
   a GPU wedge). 60000 ms passed the identical killer drill 3/3 plus
   the full wedge battery (KNOWN_ISSUES #29). Rule: the value must
   exceed worst QUEUED-work latency, not just the call's own work.
3. **`VLLM_XPU_SPEC_DRAFT_BARRIER` default 2 → 0** (single-site code
   flip, patch_barrier_v123_bake.py, v123 marker; env `=0` also passed
   at boot — belt+suspenders). User directive 2026-09-28; T2b had
   measured barrier-off identical (73.3/73.0 vs 73.6/74.1); the v62
   mode-2 machinery is retained for emergency re-enable. Full wedge
   suite re-certified on the posture: serialized 24 ×3 SURVIVED 24/24
   (72/72), bursts ×3 108/108 resets 0, drills ×3 fence-hits 0.

Spec MTP×4 and XGrammar-2 0.2.7 intact (standing user requirement),
parser qwen3_coder + 85 capture sizes inherited from v1.2.22.

Two images: `v1.2.23-raw` = `480840080515` (stage5 bake from
v1.2.22-raw, 67 bake gates 0-fail, 24.7 GB) and production
**`v1.2.23` = `23a07b1e5ff6`** (24.7 GB) — clean-warm lane-commit per
the v1222 exception, full gate battery re-run against the committed
image (gates_v1223_lane.sh ALL PASS incl. dynamic triton floor
shipped 72 ≥ raw 72), fresh-boot prod sanity + admission + posture
(async-engaged, barrier `False False 0`, RPC_TIMEOUT env) PASS, CC
battery through litellm :4000 green (after fixing two defects in the
battery script itself: missing `Authorization: Bearer` and a JSON
template tail missing the object close brace — cc_battery_fix_
v1223.sh; the image/lane were healthy throughout). lane-watchdog
repointed to repro_bootV1223_prod.sh (`.pre_v1223` backup), armed.

P13 validation (fresh host after reboot): solo cold 100.99 s parity,
battery clean, drills ×3 SUSTAIN_COMPLETE_NO_WEDGE, serial24 ×3 all
SURVIVED, bursts ×3 all SURVIVED, fairness V66 PASS (0.06 s gaps),
parser T1/T2/T3 PASS, tracebacks 0, resets 0, async 4×1024 135.45
tok/s. The inherited zero-JIT-lines gate aborted post-chain (8 lines)
— adjudicated per the #28 cache-delta standard: 11 lines = first-use
loads + 1 bounded compile (cache 72→73, rejection_greedy drill shape)
= the accepted v1222-raw precedent; gate recalibrated to cache-delta
≤ 2 inside validate_v1223_run.sh (p13_jit_adjudicate_v1223.sh, all 21
acceptances re-verified). Boot-script TRAP fixed for the async era:
the V1212-lineage baked-config gate FORBADE `--async-scheduling`
(exit 10); repro_bootV1223.sh inverts it (async REQUIRED) — older
watchdog/boot scripts must NOT be reused as-is on this posture.

Artifacts: ../perf-v123/ (patch_barrier_v123_bake.py, stage5_bake_
v1223.sh, gates_v1223_lane.sh, validate_v1223_run.sh, ship_v1223.sh,
p13_jit_adjudicate_v1223.sh, cc_battery_fix_v1223.sh, preflight_v123
under ../../.tmp-v123/).

# v1.2.24 + fp8-KV certified posture (2026-09-28)

`llm-scaler-exp:v1.2.24` = `b13f56543057` (raw `0423c13f8c21`, 24.7 GB).
Certified posture from the perf-v124 chain: **fp16 GDN pool + spec MTP×4 +
`--kv-cache-dtype fp8_e4m3`** — fp8 KV cache is quality-clean (0/104 wrong
answers + 60/60 tools through the heaviest later probe, P23F control arm)
and is the realized fp8 capacity lever; in production since this image.
gmu0.8 bs64 pc-ON spec-ON mnbt8192 async-ON no-template; async/barrier-0/
RPC-60000 posture inherited from v1.2.23. Ship chain: stage5_bake_v1224 →
validate → ship_v1224_cont (gates ALL PASS, CC battery green via
cc_battery_fix_v1224.sh). NOTE (recorded for honesty): v1.2.24's prod-boot
sed (`s/v1\.2\.23-raw/v1\.2\.24/`) was a NO-OP — the script already said
v1.2.24-raw, so the "prod" fresh-boot verification re-ran -raw; fixed
properly in v1225 (`sed 's/llm-scaler-exp:v1\.2\.25-raw/llm-scaler-exp:v1.2.25/g'`
— tag actually swaps; verified by grep). KNOWN_ISSUES #30(1).

# v1.2.25 — opsall ROOT FIX + C7 fp8-state refusal; fp8 round closed by measurement (2026-09-29)

Full record: ../perf-v125/COMPLETE_ROUND_WRITEUP.md (PHASES.md Runs 1-7y
live). Task: "both fp8(e4m3/e5m2) implementations has to be superior of
baseline on any possible cases but degradation is not allowed and has to
work end-to-end" — verdict by measurement:

1. **fp8 GDN-state (=1 ESIMD, =2 SYCL; both formats) NOT certifiable —
   documented known-broken, refused in-code.** Speed below fp16 in every
   family (solo −2.4…−12.5 %, agg −2.6…−13.9 %, Run 7n uniform matrix) AND
   quality broken under concurrency: P23D =2 20 % post-prefix tool salad
   (both formats, fp16 control 0/30, healing poisoned-state windows);
   P23F =1 8.75-13.75 % fresh-boot wrong answers exploding to 25-41.7 %
   post-prefix, 18/18 concurrent-wrong→serial-right flips (pure
   per-request state-slot crosstalk when prefills coalesce — the v88
   int32 root family; SILENT: tracebacks 0, resets 0). Ship gate for any
   future claim: P23F 0/80 + 0 flips + P23D 0-salad + full battery +
   speed parity, BOTH formats.
2. **fp8 KV (e4m3) IS the superior fp8 surface and ships as default**
   (--kv-cache-dtype fp8_e4m3): quality-clean through the P23F control
   arm, +1.16-3.50 % KV capacity. e4m3/e5m2 dtype resolution stays
   gate-verified on every boot (end-to-end support standing); spec MTP×4
   + XGrammar-2 0.2.7 intact everywhere.
3. **v125 opsall ROOT FIX (the round's non-fp8 win):** stale
   `custom_ops='none'` from pass-1 resolution survived the TP>1
   TORCH_COMPILE_DISABLE gate (mode flips to NONE, custom_ops doesn't) →
   native fallback NaN'd every single-token GDN decode forward; spec-4
   survived only because verify steps are multi-token. patch_v125_
   opsall_fix.py restores 'none'→'all' at the gate site (guarded), boot
   line `v125 root fix`. This unblocked the spec-off surface entirely and
   moved cert boots off the accidental-'none' posture.
4. **C7 refusal (in-code verdict carrier):** import-time RuntimeError in
   `_xpu_ops.py` when VLLM_XPU_GDN_FP8_NATIVE ∈ {"1","2"} — message cites
   P23D/P23F + kernel-round scope; refuses if the env name is already
   referenced (lineage guard). Discovery en route: the P22B env switches
   were NEVER on the raw base (zero references in `_xpu_ops.py` on
   v1.2.24-raw — live-only lane-venv handling), so =1 was silently inert
   on raw-lineage boots; C7 upgrades inert→REFUSED.
5. **Images:** v1.2.25-raw = `4e83528bfd66` (bake 93/0, 08:48:00) →
   **v1.2.25 = `3f3c91637692`** clean-warm lane-commit, ship gates
   **97 OK / 0 FAIL 10:29:56** (incl. opsall marker + backup, C7 marker +
   functional refusal trio, wheel_no_fp8_strings==0, no C5/C6, full
   v1218→v1225 pedigree + v64→v1225 jit-warm stamps, triton floor
   shipped 79 ≥ raw 72). Fresh-boot validation ALL GATES PASS 10:14:08
   (P23F probe A/B clean on the shipped default, drills 3/3, serial24 ×3,
   bursts 36/36, resets 0, solo 78.6/79.7/79.5, async 4×1024 agg
   149.80 tok/s). CC battery ALL GREEN via python-built bodies; watchdog
   armed on repro_bootV1225_prod.sh (code=200 10:54:34). Standing trap
   unchanged: V1212-lineage boot scripts FORBID async — V1225 lineage
   only.

## KNOWN_ISSUES #30 — v1.2.24/v1.2.25 round entries

1. **ship_v1224 prod-boot sed no-op** (see v1.2.24 section): a sed that
   pattern-matches nothing replaces nothing — prod-boot "verification"
   must grep the swapped tag in the generated script before trusting it.
2. **Shell-assembled curl JSON is a recurring corruption source** (3rd
   occurrence: v1223 missing brace, v1225 stray trailing quote
   `)}]}\""` → litellm 400 "unexpected content after document"). Rule:
   bodies built ENTIRELY in python (json.dumps) and passed `-d @file`,
   pre-validated. ship_v1223.sh still carries the buggy line in-repo —
   do not copy it; cc_battery_v1225.sh is the canonical pattern.
3. **litellm probe semantics:** `/health` requires the master key (401
   bare) AND round-trips the backend (times out mid-boot);
   `/health/liveness` (200, unauthenticated) is the correct up-probe.
   Requests need `Authorization: Bearer sk-dummy` through :4000.
4. **Gate-script stdout is `tail -80`** — early GATE-FAILs fall outside
   the window; always grep the full log. And `\"` inside single quotes
   is a literal backslash pair — gate literals with double quotes must
   use plain `'"… "…"'` quoting.
5. **lane-watchdog status cadence is 10 cycles × 60 s** (~10 min to the
   first armed-line after restart) — armed-verify windows must match,
   and `systemctl restart` (not reload) is required to re-parse the
   repointed script.
6. **Single-exposure probes cannot certify state-format quality** (n2val
   + agg2 passed; P23F later measured 8-42 % wrong-answer rates on the
   same postures). Certification = rate study: ≥80 fresh + post-prefix +
   serial control + flip detection.

Artifacts: ../perf-v125/ (COMPLETE_ROUND_WRITEUP.md, PHASES.md Runs 1-7y,
all p22*/p23*/patch_v125* scripts, stage5_bake_v1225.sh, gates_v1225_
lane.sh, ship_v1225*.sh, cc_battery_v1225.sh, repro_bootV1225*.sh,
evidence/); ../perf-v124/ (v1.2.24 chain + evidence).
