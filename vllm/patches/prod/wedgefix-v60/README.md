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

# WEDGEFIX-G (v62) — host-side draft barrier: ROOT FIX, v1.2.16

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
