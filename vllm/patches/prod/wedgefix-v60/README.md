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
