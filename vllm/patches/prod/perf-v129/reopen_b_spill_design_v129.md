# REOPEN-B — e4m3 frozen-state spillover: design document (P54, document-not-do)

perf-v129 Phase-2 item 3. Verdict up front: **STAYS_CLOSED** — this
file records the design, the perturbation analysis against the
recurrence law, and the reopen triggers. No code, no measurement.

## 1. Context and the question

The v127 recurrence-compounding law (M1 rejection) says:

- e4m3 error injected EVERY decode step compounds through the GDN
  recurrence and collapses quality at N~6 steps (measured; reproduced
  analytically in P53a: e4m3 SS error p50=27%, max=715%).
- fp16 is the minimum viable RECURRING state format.
- fp8_e4m3 KV cache SURVIVES because it is write-once: a token's KV
  is written at prefill and read many times — never re-quantized.

The one profile the law leaves open is between those poles: state
that is written once, read once, and does NOT recurse in between —
a FROZEN (preempted/swapped-out) conversation's GDN pool rows.
REOPEN-B asks: can frozen states spill to e4m3 to extend effective
pool capacity?

## 2. Proposed mechanism (if ever implemented)

- **Freeze** (on preemption): copy the request's fp16 GDN pool rows
  to a spill area encoded with the certified v126 bridge semantics —
  `stored = e4m3(h * scale)`, per-(hv,k) feature scales from the M0
  rule (0.98 * 448 / runmax). Encode/decode at the storage layer is
  already proven (v127 M2 gates B/D: boundary idempotence + storage
  superiority on stratified states).
- **Thaw** (on resume): decode `h = fp32(stored) / scale` back into
  fp16 pool rows; the state then recurses in fp16 as usual.
- Only the freeze/thaw ORCHESTRATION is new code; the numeric core
  ships today.

## 3. Perturbation analysis vs the recurrence law

A one-shot e4m3 roundtrip injects eps ~= 6.25% ONCE per freeze/thaw
cycle. Unlike the per-step injection that killed M1:

- **No OU amplification.** The P53a amplification factor
  sqrt(1/(1-gamma^2)) applies to CONTINUOUS injection. A one-time
  perturbation delta at thaw propagates as delta * gamma^n — it
  decays geometrically and is further diluted by new inputs each
  step. This is exactly why write-once fp8 KV survives the same eps.
- **Thrash is the only compounding path.** If a request is
  preempted and resumed repeatedly, each thaw re-injects 6.25%. The
  injections are separated by however many decode steps the request
  got to run in between; with gamma p50 = 0.973 the previous
  injection has decayed substantially (>2x reduction) after ~25
  steps. Worst case (instant re-preemption, zero steps between
  thaws) the errors add linearly in T, the thrash count — but a
  request that never gets to decode is already a scheduling failure
  regardless of state format.
- **Quality gate by construction:** single-shot 6.25% < the 18%
  rejection bar; the bar is only approachable under pathological
  thrash, which admission control (WS-C) exists to prevent.

## 4. Cost / benefit / risk

- **Benefit:** frozen-state bytes halve (fp16 2B -> e4m3 1B per
  element) -> ~2x concurrent frozen-state capacity; relieves the
  eviction pressure P48 quantified (90.1% of the 18.8% recompute
  excess is eviction-class) without touching the hot path.
- **Cost:** spill alloc/free paths, request-state bookkeeping, thaw
  on resume; a second state area to size and police.
- **Risks:**
  1. The preemption path is the crash-prone regime — P51b measured
     DEVICE_LOST under sustained oversubscription (24-way, ~1.1x
     working set, T+67 min; captures/crash_WD_150545_excerpt.txt).
     Adding spill machinery there BEFORE the crashfix workstream
     closes the class compounds blast radius.
  2. Thrash quality tail (above) — bounded but nonzero.
  3. Opportunity cost: if REOPEN-A's int8 hybrid (P53) lands, the
     in-pool state itself halves in bytes and spill-to-e4m3 becomes
     the inferior density option (1B vs 1B, but int8 has 16x better
     eps if it ever does recur).

## 5. Dependency ordering and reopen triggers

REOPEN-B is gated on ALL of:

1. The DEVICE_LOST crash class is closed by the crashfix workstream
   (root cause + fix certified) — preemption churn is its trigger
   regime.
2. WS-C concludes CAPACITY is the required lever (if admission
   control suffices, spill is moot).
3. REOPEN-A is resolved CLOSED (int8 hybrid rejected) — otherwise
   REOPEN-A supersedes on density and eps.

## 6. Measurement plan (only if reopened; M3-harness shape)

- One-shot decay curve: inject 6.25% at step 0 on harness shapes,
  measure per-step relative error vs fp16 reference — expect
  geometric decay (fit ~gamma^n), NOT growth; validates the
  no-amplification claim empirically.
- Thrash matrix: T in {1, 2, 4, 8} x re-preempt intervals
  {0, 8, 64, 256 steps}; gate = 18% bar never crossed at any
  measured point after thaw.
- Storage-layer gates (idempotence, superiority) re-run unchanged
  from v127 M2 — already certified, re-run only as regression.

Verdict: **STAYS_CLOSED.** Documented per directive; reopen only on
the three triggers above.
