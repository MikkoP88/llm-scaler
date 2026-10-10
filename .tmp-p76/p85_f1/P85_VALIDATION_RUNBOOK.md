# P85 — F1 Validation & v1.2.32 Bake Runbook

**Round:** P85 Phase 3/4 · **Date:** 2026-10-07 evening · **Host:** 10.20.3.65
**Candidate:** `llm-scaler-exp:v1.2.32rc1` (image id adcf2071…) — v1.2.31
filesystem + exactly one changed file:
`vllm/v1/worker/gpu_model_runner.py` md5 `6269b213` → `72607f46`
(patcher: `p85_stage/patch_v85_f1.py`, 7 anchored sites, py_compile clean).

## F1 content (root fix, python-only, zero GPU ops / zero barriers)

1. **Rid-keyed accepted remap** — `_update_states` wait side keys the pinned
   `num_accepted_tokens` by request identity (record-step `_v85_rec_index`
   stash in both align/non-align branches) instead of positional
   `prev_positions`; unknown rows → 1; envelope clamp [1, num_spec+1].
2. **Rid-truthful prev_positions repair** — rows whose claimed record row ≠
   the rid-truthful row are repaired before any positional consumer runs:
   the num_computed GPU correction (M2 tear), the accepted-count where()
   overwrite (v52e rollback column), `_prepare_input_ids` token injection.
3. **vsc freshness gate** — `valid_sampled_token_count_gpu` is only
   assigned on the drafter fallback path; after fits-path steps the GPU
   correction gathered a stale-step row order into
   `corrected = prev_computed + valid_counts`. Gate: epoch advanced since
   the previous record-step mirror; stale steps take the CPU fallback.
4. **Forensic ledger** (env `VLLM_V85_LEDGER=<path>`, OFF by default) —
   per-step accepted vector + repair count.

**Evidence this fixes:** crash-6 death tail shows PRE-COPY advancing state
21→22 while DEF-PP (stale `acc=1`, i.e. the neutralized/misaligned value)
copies it back 22→21 at the same boundary, same second. With rid-truthful
acc, `newc` lands past the boundary → `dest == src` → copy self-cancels.
**Live proof during smoke:** `prev_positions repaired (first occurrence:
1/1 rows)` fired on the first request — the desync occurs in normal
traffic and F1 catches it.

## Gates (all must pass before bake)

| # | Gate | Criterion |
|---|------|-----------|
| G0 | Boot | Certified chain `boot_v1227_restore.sh P85VAL`, image override only; HEALTH_OK; posture ckpt verified; md5 `72607f46` in-container |
| G1 | Standard storm | 72/72 completions, zero resets, wall within parity band (v1.2.28 band 1089/1029/1042 s; v1.2.31-era ±) |
| G2 | Depth storms ×3 | 108/108 × 3 legs (the 3/3-death shape), zero resets, zero devcoredumps |
| G3 | Perf parity | bench_run7n: agg/solo within ±2% of v1.2.31 era ledger |
| G4 | Agentic soak | ≥75 min continuous deep-spec (3 workers, 45→80k histories, resends), ZERO resets — ≥3× the 25-min band; target 120 min |
| G5 | Ledger scan | All accepted values ∈ [1,5]; repair counts reported; no envelope violation |
| G6 | MTP intact | spec counters advancing, acceptance ~3.1 mean, decode tok/s ≥ v1.2.31 era |

## Execution log

- 20:05 host rebooted (post crash-6, pre-test-round law). Flag persisted,
  litellm self-recovered.
- 20:10–20:12 `V1227_IMAGE=llm-scaler-exp:v1.2.32rc1 bash
  /root/build/v127_stage/boot_v1227_restore.sh P85VAL` — HEALTH_OK 170 s,
  KV 370,254 tokens, posture verified. G0 PASS.
- 20:13 smoke completion OK; F1 markers: ACTIVE ×2 (both TP ranks),
  repair first-occurrence ×2.
- 20:13:17 **user's live client (10.100.8.6) reconnected** — rode the whole
  validation window as uncontrolled co-load (5 concurrent deep histories,
  ~3 completions/min).
- 20:15–20:41 cold standard storm: 72/72, 0 resets, 0 errors, wall 1562 s
  (cold engine + co-load; recorded as warm-up leg only).
- 20:46–21:09 warm standard storm: 72/72, 0 resets, 0 errors, wall 1358 s
  (co-loaded — NOT a valid parity number; v1.2.28-era band 1089/1029/1042
  was quiet + sync-spec era, not comparable).
- 21:20 bench_run7n under co-load: solo 27.6/27.1, agg4 110/116 — fair-share
  math with 5 deep live streams (era quiet absolutes: solo 74.4, agg4
  239-258, agg8 435-437 from p84_stage/ERA_BENCH_V126_V131_20261007.md).
  One-sided gate unusable under this co-load; parity must be measured quiet.
- Mechanism audit (deployed file): P3-stale else-branch = one non-blocking
  H2D copy of ≤64 ints (cheaper than the GPU branch it replaces); P2 loop
  ≤64 dict ops/step; deep decode is fallback-path → fresh → GPU branch
  unchanged. **No decode-cost mechanism exists in F1.**
- Telemetry compare (p85_telem_compare.py, boundary 20:12): rc1 p90 bands
  26.1/24.6/27.2/20.8 vs era 36.1/34.7/30.9/24.9 at running=4/5/6/8 — all
  within co-load explanation (era bands were quiet-bench-heavy windows).
- **User directive (21:15): async + spec + xgrammar-2 must stay enabled for
  maximum performance.** Verified on lane: ALLOW_ASYNC=1, ZOMBIE_STRIKES=3,
  num_speculative_tokens=4, xgrammar 0.2.7. Ship keeps all three (image
  delta is one scheduler file).
- 21:23 live soak evidence: 65+ min of the exact killer shape on rc1, ZERO
  resets (v1.2.31 died at 13-41 min exposure, 6-for-6). G4 material.

## Holding pattern

Heavy legs (depth ×3, quiet parity bench, ledger soak) HELD until the
user's live window closes (ESTAB from 10.100.8.6 drained ≥10 min). The
live traffic is the strongest soak; controlled benches need exclusivity.

## Ship sequence

1. ~~`docker tag llm-scaler-exp:v1.2.32rc1 llm-scaler-exp:v1.2.32`~~ —
   **DONE 22:05 (user-ordered)**: v1.2.32 = adcf2071b646 = rc1 digest
   (validated bits, no rebuild); venv runner md5 72607f46 verified in-image.
2. ~~Update default image in `boot_v1227_restore.sh`~~ — **DONE**: both
   copies (/root/build/v127_stage/ + /root/build/) now default
   `llm-scaler-exp:v1.2.32`; bash -n OK; **new chain md5
   `9fcaf1d55f3d92d6fa7106d0d3b6b067`** (was 1e0d53a7).
3. PENDING (at round close, after user's live window ends):
   `rm /root/build/lane_watchdog.paused` → watchdog arms on v1.2.32;
   running container already IS v1.2.32 content (same digest).
4. Copy-paste posture delivered to user (docker run + env + serve args).
