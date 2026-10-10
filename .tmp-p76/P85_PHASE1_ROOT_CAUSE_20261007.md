# P85 — Phase 1 Root-Cause Analysis: guc-ccs crash class under live deep-spec traffic

**Round:** P85 · **Date:** 2026-10-07 (evening) · **Lane:** 10.20.3.65 `lsv-test`
**Posture analyzed:** v1.2.31 (certified) + spec MTP×4 + async (`ALLOW_ASYNC=1`,
`ZOMBIE_STRIKES=3`) + `--chat-template qwen38_high`
**Directive:** deep-analyze crash + testing; fix root issue with zero performance
degradation; MTP must stay supported; identical validation testing before ship.

---

## 1. Executive verdict

The crash class is an **exposure-bound async carry-index desync** that strikes
during DEEP spec forward (20–80k-token histories), with two downstream
presentations:

1. **NaN zombie** (survivable, symptom): a stale `num_accepted_tokens` value
   makes the mamba carry read a state slot the GDN kernel never wrote →
   all-NaN target logits. v52e diagnosed this; v52k clamped the copy-read
   bias; v60c strikes-3 absorbs transient rows. Confirmed **symptom, not
   cause**: zero STRIKE-OUTs in all four death logs.
2. **Engine death** (the crash): the MAIN compute stream stops retiring inside
   a deep forward; the spec+async-only side-copy context (`valid_sampled_
   token_count_copy_stream`, guc_id 22/32 "LR job cleanup") parks on its
   leading `wait_stream` semaphore and becomes the first GuC watchdog victim
   → ccs reset → 30 s bcs cascade → GT reset → DEVICE_LOST → watchdog
   relaunch (~4 min). ACTHD == RING_BBADDR in both banked devcoredumps proves
   the parked context executed exactly one instruction — the cross-stream
   wait — i.e. **the main-stream timeline it waits on never advanced**.

**Phase-1 static kernel audit result (new this round):** every data-dependent
INDEX surface in the live native GDN kernels is already clamped, and the
accepted-count tensor is structurally bounded. This **demotes the
runaway-loop theory** and promotes a wait/fence-desync mechanism (see §4).

---

## 2. Evidence chain (all banked, cross-checked)

### 2.1 Four deaths today, one signature (EXPOSURE-TIME HAZARD LAW)

| # | dump time | card(s) | exposure | last logged op (v52f) | KV at death |
|---|-----------|---------|----------|------------------------|-------------|
| 1 | 17:03:06 | 1+2 | ~25 min | PRE-COPY slot 39→40 @ computed 40,959 | 14.5% |
| 2 | 17:15:49 | 1 | ~13 min | PRE-COPY slot 53→54 @ 55,296 | 31.6% |
| 3 | 17:32:50 | 2 | ~13 min | PRE-COPY slot 19→20 @ 20,480, sched=884 | 27.2% |
| 4 | 18:26:17 | 1+2 same second | ~19 min | strike @ computed 80,023; slot 23→24 @ 24,576 | 99.2% |

- 4/4 last-logged-op = mamba boundary-carry event at a 2048-token state-slot
  boundary, at depth 20k–80k. **Depth (slots 19–78) is the invariant; KV
  usage is not** (14.5%–99.2%).
- Crash #4 was POST-REBOOT (host fresh 17:42) → reboot clears only the
  exposure clock, not the class.
- Dual-card same-second dumps (#4) rule out per-card independent timing: the
  trigger hit both engines in one submission wave (TP=2 same step).
- Storms (8k-deep) stay clean: the drafter fits (`input_fits_in_drafter`),
  direct-propose path, **no side-copy context fire**. Live agentic 20–80k
  takes the fallback path where `_copy_valid_sampled_token_count` fires on
  prefill/mixed steps — seqno duty of the hung context (~0.07–0.1 jobs/s)
  matches that fire rate.

### 2.2 Devcoredump register forensics

- Hung context: Class 5 compute, guc_id 22 (card1) / 32 (card2), one inflight
  job (LRC head<tail), Reason "LR job cleanup".
- **ACTHD == RING_BBADDR** (crash-4 card1: `0x0000c001fc0f3b68`/`...b69`):
  the engine fetched and executed the FIRST instruction of the parked batch —
  the leading semaphore wait of the side-copy context — and stopped. The job
  waits on the main-stream timeline; that timeline never advances.
- RING_START `0x0241b000` identical across eras/days (deterministic alloc);
  GuC log body is compressed (strings-dead); only ONE context (the reset
  victim) is listed per dump.
- GPU VAs (0xc001…/0xd556…) never appear in /proc/pid/maps — maps-based
  attribution of hung-batch VAs is structurally impossible (device memory is
  not CPU-mapped). devdump_watch v3 still banks pid-maps for future corroboration.

### 2.3 The side-copy context is spec+async-only

`gpu_model_runner.py` (XPU-aliased): `valid_sampled_token_count_copy_stream`
exists under `if self.num_spec_tokens:` **and** `if self.use_async_scheduling:`
— i.e. only on the crashing posture. `_copy_valid_sampled_token_count` does
`wait_stream(default_stream)` → pinned copy → event record. Nospec lanes have
no such context → structurally zero-risk (consistent with all nospec history).

---

## 3. Native kernel audit (live binary) — indices clamped, bounds structural

**Binary lineage (proven by md5):**
live wheel `_xpu_C.abi3.so` = `ee1c864d32292c41b274c31d4c755ea7` =
`/root/build/vxk/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/_xpu_C.abi3.so`
(built **Sep 25 16:51**). Source = current tree minus the Sep 28 P22B
fp8-state edit (dormant: live SSM pool is fp16; v126 dual-bridge handles any
fp8 window by dequantizing to fp16 before the kernel). **No unshipped wedge
fix exists in the tree.** The live binary contains:

| Fix | Date | Site | Effect |
|-----|------|------|--------|
| #11 init_col clamps | Aug 30 | `gated_delta_rule.hpp:396-402` | `init_col = accepted-1` clamped to `[0, num_spec_tokens-1]` — stale/garbage accepted can never index past the cache row |
| #11 ragged walk | Aug 30 | `gated_delta_rule.hpp:419-434`, `causal_conv1d.hpp:633+` | rectangular `b*num_spec_tokens+t_local` replaced by `[qsl[b], qsl[b+1])` walk — kills the historical OOB `token_indx` read / wild `core_attn_out` scatter |
| V88FIX int64 offsets | Sep 25 | `causal_conv1d.hpp` (4 sites) | `id * stride` overflow at id ≥ 4097 killed |

**Structural bounds (why runaway indexing is demoted):**
- `num_accepted_tokens` = `(output_token_ids != -1).sum(dim=1)` — device-side
  count, always ∈ [0, num_spec+1]. The async desync delivers **stale or
  row-misaligned** values, never out-of-range magnitudes.
- Slot ids come from the allocator block table — always valid pool rows.
- Provenance chain of the desync surface (fully mapped):
  `gpu_model_runner.py:1846` (device compute) → `:1858/:1883` (TWO pinned-copy
  sites + event record) → `:2527-2553` `_update_states` (event wait, np read,
  write-back, `copy_to_gpu`) → `:2871` (metadata) → `gdn_attn.py:160/314`
  (row-select) → kernel `cache_indices[r, accepted-1]` load. The `:2556`
  comment literally names the "2048-boundary carry bias =
  num_accepted_tokens_cpu - 1" (v52e surface); v52k clamped the mamba-copy
  read side of it.

**Residual unclamped surfaces (ranked, see §4):**
- (a) spec-path loop bound `token_count = qsl[b+1] - qsl[b]` — host-built
  cumsum under async; internally consistent with `token_indx` (same builder
  pass) → garbage requires a torn metadata build. Low probability, cheap to
  harden.
- (b) `global_t = token_indx[t]` write remap — stale-but-in-range values
  give wrong-row writes (activation corruption → NaN family), not wild VAs.
- (c) The `v126` dual-bridge gather/scatter — dormant on fp16 SSM.

---

## 4. Causal theory (unified) — ranked mechanisms for the hang

**Root cause (established):** async carry-index desync — under
`--async-scheduling`, the accepted-count side-channel (pinned CPU copy +
event, consumed one step later by `_update_states`) can deliver a stale or
row-misaligned `num_accepted_tokens` while the engine is already building the
next step's GDN metadata. Everything below is downstream of this one race.

**Hang mechanism candidates (to discriminate in Phase 2):**

| # | Mechanism | Support | Confidence |
|---|-----------|---------|------------|
| M1 | Corrupted inter-op synchronization object: a stale-`global_t`/wrong-slot write (bounded VA, activation space) clobbers a semaphore/event/flag tensor the deep forward subsequently waits on; the wait never fires; main stream stops; side-copy context parks (ACTHD==BBADDR). | Explains probabilistic timing (needs the race + the right victim buffer), dual-card same-second (both ranks corrupt + wait in the same step), 4/4 boundary-carry adjacency, and why all kernel indices being clamped doesn't help. | **Leading** |
| M2 | Torn metadata build: `qsl`/`token_indx` inconsistent (garbage `token_count`) → unbounded spec-path loop. | Requires the host builder to see half-updated state; builder runs in one pass — torn only if the desync splits across its reads. Possible but no direct evidence. | Possible |
| M3 | Subgroup-divergence deadlock in a SYCL collective after an early return (shape- not data-dependent). | Would be deterministic per shape, not exposure-probabilistic; shapes repeat every step. | Unlikely |
| M4 | Pure driver/firmware flake. | **Off program** per user directive ("issue is not on driver or firmwares"); fw mismatch (70.72.1 vs wanted 70.49.4) is chronic-but-stable across clean eras. | Rejected |

---

## 5. Phase plan

### Phase 2 — discriminative repro + capture (leg window, not the live lane)
1. **Agentic-soak repro** mirroring the live shape exactly: 2–3 concurrent,
   growing histories 45→80k, full-history resend waves, unbounded duration
   (expected death within 13–25 min of continuous deep-spec exposure).
2. **Step-ledger logger** (new, tiny): per spec step, record
   `num_accepted_tokens` host-side values + `qsl` deltas to a ring file.
   Post-mortem: any value outside [0, num_spec+1] or negative `token_count`
   delta = M2 confirmed (torn build); all-clean = M1 (sync-object corruption).
3. `VLLM_V52K_TIMER=1` (v52k per-event device timing, already built in),
   devdump_watch v3 armed, `dmesg -w` capture, SPECTIMING flush.

### Phase 3 — fix (zero-perf, MTP×4 intact), layered
- **F1 root fix (Python-only):** close the desync at source — make the
  accepted-count side-channel handshake step-consistent (guard the
  `_update_states` event-wait/read against cross-step re-record; audit the
  two copy sites 1858/1883 and the deferred-postprocess wait at 4538).
  No stream barrier introduced (NO_BARRIERS law untouched); no GPU op added.
- **F2 kernel hardening (cheap, defense-in-depth):** clamp
  `token_count ∈ [0, num_spec_tokens]` in both spec kernels; optionally
  bounds-skip `global_t` writes. Two integer ops per work item — noise.
- Rejected: nospec-bypass-for-deep (kills the MTP deep-decode win the user
  demands); driver/firmware work (off program).

### Phase 4 — identical validation before any ship
- Same batteries as the P84 certification (storm 3× 108/108, solo + con-4
  deep decode, bench_run7n parity ±2%) **plus** the agentic soak ≥3× worst
  observed exposure (≥75 min continuous deep-spec) with zero resets.
- ZERO-DEGRADATION gate: only improvements allowed vs the v1.2.31 baseline.

---

## 6. Live-lane status during this analysis

Lane green through the analysis window (WD_182843 restore 18:28:43; 28+ min
exposure survived by ~18:56 — longest clean window today, consistent with the
probabilistic exposure law). devdump_watch v3 armed (PID 27263), crash-4
leftover dumps harvested + dismissed (slots clean for crash #5).

## 7. Artifact index

- Devcoredumps: `/root/build/lce1/WD_crash{2,3,4}_*/`, `LIVE_crash_184858_card{1,2}.devcoredump`
- Death logs: `/root/build/lce1/WD_crash_*/serve_full_Q22.log`
- Kernel tree (live lineage): `/root/build/vxk/csrc/xpu/gdn_attn/`
  (+ `/root/build/v26/` Aug-30 snapshot, `/root/build/kernel_fix/{orig,fixed}/`)
- Register extractions: this doc §2.2; earlier P71 bank (Oct-2 specimen
  `LIVE_crash_144353_card1.devcoredump`, seqno 1272, same signature)
