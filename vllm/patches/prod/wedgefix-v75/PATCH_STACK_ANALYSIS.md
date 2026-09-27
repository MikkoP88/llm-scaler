# PATCH_STACK_ANALYSIS — the v1.2.21 protective stack: what every barrier,
# guard, and patch still protects after the v88 int64 root fix

**Date:** 2026-09-27 · **Mode:** analysis and research only — no vLLM
instance, container, serve process, or host was modified (user directive
2026-09-27: "do not affect any of vllm instances", "only analyzing and
research"). All live evidence below is read-only (logs, git, patch sources).

**Question answered:** is `VLLM_XPU_SPEC_DRAFT_BARRIER` — or any other
barrier/guard in the standing stack — still needed now that the v88 int64
kernels fix (v1.2.21, KNOWN_ISSUES #27) removed the dominant wedge class?

---

## 1. Executive verdict

| Item | Guards against | Post-v88 verdict |
|---|---|---|
| `VLLM_XPU_SPEC_DRAFT_BARRIER=2` (v62 host flock) | TP collective submission skew (theoretical) | **Not provably load-bearing** — both original justifications are void (§3). Keep while live validation is not permitted; retirement protocol in §3.6. Never use `1`. |
| v31.1 inductor gate (`TORCH_COMPILE_DISABLE` for spec+TP2) | #11: inductor piecewise × spec livelock ≥32k | **NEEDED — keep.** The #11 root cause (inductor codegen) is unfixed upstream; the gate is the fix. Verified firing on every v1.2.21 boot. |
| `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` (v27) | ≤32k eager oneCCL `all_reduce` wedge class | **KEEP.** Independent wedge fix, still active, zero cost. |
| `VLLM_V55_EVENT_TIMEOUT_S=600` / `VLLM_V55_DECODE_TIMEOUT_S=30` | silent stalls → loud bounded kills | **KEEP.** Watchdog (defense-in-depth); fired correctly in the v60 control drill; zero steady-state cost. |
| `VLLM_F15B=1` / `VLLM_F15B_STALL_S=45` | stall observability (f15b ring) | **KEEP.** Produced the decisive forensic fingerprints; diagnostics only. |
| WEDGEFIX-A: async scheduling OFF | async event-pipeline amplifier | **KEEP.** Independent amplifier removal; standing directive (no async scheduler). |
| WEDGEFIX-C: v60c sync zombie guard | NaN-zombie → SYCL gather-OOB abort class | **KEEP.** Independent defect class (2026-09-21 soak crash; real evidence). |
| WEDGEFIX-E: v60g degenerate-row sanitize | no-draft step → token-0 → XGrammar FSM-reject 500s | **KEEP.** Independent defect class (deterministic 500s; real evidence). |
| v63/v64/v66 scheduler caps + fairfix | prefill/decode fairness, waiting-head starvation | **KEEP.** Unrelated to wedges; standing fairness behavior. |
| v58_p1, arstage, STALFIX v59 | async-copy region, AR staging, tool-call stream stall | **KEEP.** Independent fix classes. |
| v88 wheel (int64 conv-kernel offsets) | the int32 pool-offset wedge ROOT CAUSE | **THE root fix** — this is what made the other wedge arguments moot. |
| `VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0` | (gate for the draft barrier) | Moot under mode 2; keep as-is (its 8192→0 justification was int32-confounded, §3.2). |

Bottom line: **the draft barrier is the only element whose necessity is
genuinely in question, and the honest answer is "not provably needed, but
also never proven harmful or costly — retire only via the validation leg in
§3.6, never ad-hoc."** Everything else in the stack guards a defect class
that is independent of, and still live despite, the v88 fix.

---

## 2. What the standing stack actually is (v1.2.21)

Env block (repro_bootV1221.sh:20–51) — protection-relevant lines:

```
-e VLLM_V55_EVENT_TIMEOUT_S=600 -e VLLM_V55_DECODE_TIMEOUT_S=30   # v55.3 watchdog
-e VLLM_XPU_SPEC_DRAFT_BARRIER=2 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0  # v62 host flock
-e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45                             # f15b stall ring
-e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1                              # v27 allreduce fix
-e VLLM_GUIDED_DECODING_BACKEND=xgrammar                           # XGrammar-2
```

Plus, inside the image/boot chain: the v31.1 inductor gate (fork platform
code), WEDGEFIX-A (async OFF), WEDGEFIX-C (zombie guard), WEDGEFIX-E
(degenerate-row sanitize), v63/v64/v66 scheduler behavior, and the v88
kernels wheel. Serve config: TP2, gmu 0.8, e4m3 KV, MTP ×4, sync
scheduler, prefix caching ON, reasoning/tool parsers qwen3/qwen3_xml.

Each class above traces to a specific, evidence-backed defect. The wedge
taxonomy (§3.3) shows why v88 does not obsolete them.

---

## 3. Deep dive: `VLLM_XPU_SPEC_DRAFT_BARRIER`

### 3.1 What it is (code anatomy)

Introduced in the v2x era as a **`torch.xpu.synchronize()` drain before the
drafter collectives** in `gpu_model_runner.py` (oneCCL wedge mitigation,
KNOWN_ISSUES #11 era; first appears in git at `38dfce8`). Today three modes
(`patch_wedge_v62b_bake.py`, baked since v1.2.16):

| Mode | Meaning | Cost |
|---|---|---|
| `0` | off (v37 posture) | 0 |
| `1` | v60 full device drain, every step | **−9…−31%** (v60 A/B) — **FORBIDDEN by standing user constraint** |
| `2`/`host` | **v62 host-side flock rendezvous** (default) | two /dev/shm flock round-trips per spec step (µs-class), no device sync |

Mode 2 parks both TP worker **hosts** at the same pre-drafter site in a
2-party sense-reversing barrier over
`/dev/shm/llm_scaler_spec_hb_<EngineCore-ppid>.{lock,state}` (60 s timeout
degrades to unsynchronized rather than hang). Device queues keep their
≤1-step run-ahead; the v60 drain's device sync is gone. Insertion site:
`gpu_model_runner.py`, gated only by
`common_attn_metadata.max_seq_len >= _SPEC_DRAFT_BARRIER_MIN_CTX` (0 =
every step).

### 3.2 Chronology — every justification the barrier ever had

| When | Event | Status of the barrier | Evidence |
|---|---|---|---|
| v2x (Aug 2026) | #11 wedge era; drain introduced as oneCCL mitigation, default ON | mitigation for #11 | KNOWN_ISSUES #11; `38dfce8` |
| v26 (`d0d620830` wheel) | GDN ragged-batch OOB fixed in wheel; **int32 conv-offset defect ships here** (v26 hardened delta-rule kernels, not conv kernels) | — | `d0d6201`; KNOWN_ISSUES #27 |
| v27 | ≤32k wedge convicted as eager oneCCL `all_reduce` → `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` | #11 story starts fragmenting | `192705c` |
| v31/v31.1 (2026-09-01) | **#11 ROOT CAUSE convicted: inductor-compiled piecewise path × spec** — fixed BY CONFIG (`TORCH_COMPILE_DISABLE=1` for spec+TP2), graphs kept, faster than the wedging config | **original justification VOID** — #11 never needed the barrier | KNOWN_ISSUES #11 v31/v31.1 update |
| v33 | barrier removed as keeper: **strictly better OFF** — fp8+spec 2k **+13.7%**, 4bit+spec 2k **+28.5%**, 65k +3–4%, hashes unchanged, no wedge in sustained testing on the post-v31.1 tree | OFF = keeper | `23be3b8` |
| v37 | default flipped OFF (`v37_barrier_default.py`) | OFF = default posture | `50a08b3` |
| v60 (2026-09-21) | v1.2.14 CC crash loop (0 tok/s, v55.3 kills, engine resets) → re-enabled: default `1`, every step, + async OFF. Control drill (async ON, barrier OFF) wedged with "#11 fingerprint"; drill 3 (async OFF, barrier gated @8192) **also wedged** | ON again — but see confound below | wedgefix-v60 README |
| v61 (2026-09-22) | replicated-drafter bisect: boots 5/6 `UR_RESULT_ERROR_DEVICE_LOST` in `gdn_attention` **eager prefill** with barrier OFF; **boot 7 same crash with barrier ON (drain)** → "barrier irrelevant" | barrier proven unable to stop the DEVICE_LOST class | `43b954f`; w61_boot{5,6,7}_crash.log |
| v62 (2026-09-22) | host flock barrier (mode 2), default in v1.2.16; boot-9 drill clean → labeled "ROOT FIX" | mode 2 standing | `43b954f` |
| v75–v88 (2026-09-24/25) | **the tiny-step wedge reproduced deterministically ON v1.2.20 with BARRIER=2**: serialized traffic died at cycle 17 every leg; v83m burst deaths at ALL spec depths; v86 captured the fault (layer-16 prefill, ssi=4173); v87 measured stride(0)=524,288 → **int32 overflow at ids ≥4097 = the actual root cause**; v88 fixed it | **"ROOT FIX" claim superseded** — the barrier never protected against the class that dominated the era | RCA_TINYSTEP_WEDGE.md P24n–P24v; KNOWN_ISSUES #27 |
| v1.2.21 (2026-09-25) | post-v88: serialized 24/24, bursts 108/108, 3× drill no-wedge, fairness intact, resets 0 | standing WITH mode 2 | ship_v1221 evidence |

### 3.3 The confound, stated plainly

Every wedge that motivated re-enabling the barrier (v60 control drill,
drill 3, the v61 crashes) occurred **while the v88 int32 defect was live**
(it shipped in the v26-era wheel `d20260830` and ran through every image
up to v1.2.20). With post-v88 knowledge:

- The v61 `UR_RESULT_ERROR_DEVICE_LOST` crashes in `gdn_attention` eager
  prefill are textbook int32-class signatures (prefill GDN region, wrapped
  pointer, unmapped page) — and they occurred **with the barrier ON**.
- The v60 "#11 fingerprint" spins (drafter ARs complete → silence,
  `propose_gpu_done` PENDING, engine resets) are exactly what a device
  that just died mid-step looks like from the host — the v75-era
  devcoredump analysis (semaphore-wait, LR cleanup) was the **aftermath**
  of the overflow fault, later superseded by the v86/v87 capture chain.
- The decisive counter-proof: **v1.2.20 with BARRIER=2 wedged
  deterministically at serialized cycle 17, every leg** — if the draft
  barrier protected against the real wedge, that lane could not have died.

Conclusion: the v60 re-enablement and the v62 "ROOT FIX" attribution were
**timing fortune on a lane whose wedge trigger was actually the pool-id
lifecycle**, not collective skew. The barrier's drill passes and the
serialized-24 deaths differed only in traffic pattern (drills recycle ids
differently than 24 back-to-back fresh requests crossing id 4097).

### 3.4 What the barrier can still be said to do

- **Theory (unvalidated post-v31.1):** bounds host-side collective
  submission skew between the two TP workers each spec step. The v62
  mechanism text argues this keeps queue divergence inside the ≤1-step
  run-ahead the engine already enforces. No wedge class has been
  attributed to this mechanism on any post-v31.1 tree with the barrier
  off (v33/v37 sustained testing was clean).
- **Observed in production: nothing.** Zero barrier-timeout warnings in
  any archived log (`grep 'peer did not arrive'` → 0 hits across
  `/root/build/lce1/*`). The barrier has never demonstrably fired in
  anger; it has also never caused a fault. It is silent insurance.

### 3.5 Cost accounting

| Comparison | Delta | Source |
|---|---|---|
| mode 1 (drain) vs OFF — short decode | OFF **+31%** (50.6→66.2 tok/s) | v60 A/B (`v60_ab_barrier.sh`, 3 reps × 3 shapes) |
| mode 1 vs OFF — ~2k ctx | OFF +9% | same |
| mode 1 vs OFF — conc-4 agg | OFF +11% (means flat) | same |
| mode 2 vs mode 1 | mode 2 **+22%** on C4 agg (92.3 vs 75.4) | v62 boot-9 vs boot-8 drain ref |
| **mode 2 vs OFF** | **never measured** | — no direct A/B exists |

Mode 2's residual cost is two host-side flock round-trips per spec step
(µs-class against ~60 ms step gaps) plus the loss of host submission
run-ahead between the two workers — in TP lockstep execution the workers
produce the same scheduler output, so skew is already structurally small.
Expected residual: near-zero; **unproven**. The v37-era OFF gains
(+13.7…+28.5%) were measured against the *drain*, which mode 2 already
eliminated.

### 3.6 Verdict and retirement protocol

**Verdict.** `VLLM_XPU_SPEC_DRAFT_BARRIER=2` is **not provably needed**
after v88 — both of its historical justifications are void (#11 was fixed
by the v31.1 config gate; the v60-era wedge was the v88 int32 defect,
which the barrier never prevented). It is also **not proven costly or
harmful**: µs-class, zero observed firings, zero timeouts, zero crashes
attributed to it. Under the standing "crash-free beats the tax; only
improvements allowed" constraint, the correct posture is:

1. **Keep `BARRIER=2` as the standing default.** It is cheap insurance on
   a lane that is now certified crash-free *with* it; v1.2.21's
   crash-free certification does NOT extend to a barrier-off boot.
2. **Never `BARRIER=1`** — forbidden by explicit user constraint and
   strictly dominated by mode 2.
3. **Retire only via a controlled validation leg** (below), when live
   lane changes are permitted again — and only if it demonstrates a
   measurable improvement (per "only improvements are allowed").

**Retirement leg spec (deferred until live testing is allowed):**
fresh-host boot (standing reboot directive) of v1.2.21 with
`VLLM_XPU_SPEC_DRAFT_BARRIER=0` (no other change) → solo-cold seed-114
parity gate (≈101 s) → **serialized 24 ×3** (the historically-dead
pattern; 24/24 each) → burst_harsh ×3 (108/108) → 3×14-phase drill
(SUSTAIN_COMPLETE_NO_WEDGE, fence 0) → fairness + v66 admission gates →
perf A/B vs the standing numbers on the same probe family
(`v60_ab_probe.py`-class shapes + fairness 6.997 tok/s @ 0.05–0.06 s gaps
reference) → CC battery through litellm. Pass = all crash gates green AND
a measurable throughput gain; any wedge/fault → immediate restore of
`BARRIER=2` and the leg is closed as "still needed". Abort criteria,
restore path, and both-cases documentation mirror the v1221 ship chain.

---

## 4. Why the other guards stay (independent classes, still live)

- **v31.1 inductor gate** — #11's actual fix. Verified live in v1.2.21:
  every boot logs `xpu.py:344 WARNING … inductor compilation disabled for
  speculative decoding with TP=2: compiled pieces livelock both devices
  at >=32k context (KNOWN_ISSUES #11, v31.1)`. The upstream inductor
  codegen defect is unfixed; removing the gate resurrects the ≥32k
  livelock. **Load-bearing; keep.**
- **`VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1`** (v27) — the ≤32k eager
  oneCCL `all_reduce` wedge conviction. Independent of everything above;
  zero measured cost; keep.
- **v55.3 timeouts** — convert silent stalls into bounded, logged kills
  (they fired correctly in the v60 control drill, `async_output_copy
  event`). Watchdogs of last resort; keep.
- **f15b stall ring** — pure observability; it captured the fingerprints
  (AR tail → silence, `propose_gpu_done` PENDING) that drove three RCAs;
  keep.
- **WEDGEFIX-C (v60c)** — the NaN-zombie → unclamped gather → SYCL
  `IndexKernelUtils.h:63` abort chain is a real, evidenced class
  (2026-09-21 soak crash) unrelated to int32 offsets; keep.
- **WEDGEFIX-E (v60g)** — no-draft degenerate rows → sampler token 0 →
  XGrammar FSM reject → deterministic 500s: independent graph-wiring
  defect (nospec FULL_DECODE_ONLY materialization), still unfixed
  upstream; keep.
- **Async OFF, v63/v64/v66, v58_p1, arstage, STALFIX** — amplifier
  removal, fairness, and independent stream/stall fixes; keep per
  standing directives.
- **MIN_CTX=0** — its "unconditional" justification (drill 3 wedged at
  `computed=3994` under an 8192 gate) was an int32-class event; under
  cheap mode 2 the gate is moot. Keep as-is; revisit only alongside §3.6.

---

## 5. Sources

- `vllm/patches/prod/wedgefix-v60/README.md` — v60/v61/v62 arc, A/B
  tables, drill evidence (lines 131–204, 219–294).
- `vllm/patches/prod/wedgefix-v60/patch_wedge_v62b_bake.py` — barrier
  code anatomy (G2/G0b/G1), mode map, flock implementation.
- `vllm/patches/prod/image-bake-keepers-v37/v37_barrier_default.py` —
  v33/v37 keeper rationale and perf numbers.
- `vllm/patches/prod/wedgefix-v75/RCA_TINYSTEP_WEDGE.md` — the v88
  evidence chain (P24n–P24v), test matrix incl. `barrier3 → wedge`,
  invalidated-results ledger.
- `vllm/patches/prod/wedgefix-v75/repro_bootV1221.sh:20-51` — standing
  env block.
- `vllm/KNOWN_ISSUES.md` — #11 (lines ~1280–1520, v31/v31.1 conviction),
  #27 (int32 root cause), #21/#25 (related residuals).
- Git: `38dfce8` (v2x #11 FINAL), `d0d6201` (v26 wheel), `192705c`
  (v27), `23be3b8` (v33), `50a08b3` (v37), `43b954f` (v62), `b78d23e`
  (v88 ship).
- Live read-only checks (2026-09-27): v1221 boot gates (mode-dispatch
  mark present), `peer did not arrive` → 0 hits, v1.2.21 serve log
  carries the active v31.1 inductor-gate warning, dmesg Engine resets 0.
