# P84 — DECODE FIX LADDER EXECUTED (FIX-1 + FIX-2) — 2026-10-06/07

User order: "deep analyze why tok/s decode speed is really poor … and fix
issue" → AskUserQuestion approval: **Execute FIX-1 + FIX-2** (bake ladder +
async A/B, N≥3 storms, ZERO-DEGRADATION gate throughout).
Root-cause analysis (prior segment) = memory decode-toks-root-cause.md:
165ms sync steps / 12.0 tok/s/req live from (1) v52k not baked, (2) async
force-killed by xpu.py:585 env gate, (3) mixed-eager steps. Acceptance
exonerated (2.47 mean).

## FIX-1 adjudicated — v1.2.31 WINNER, v1.2.32 (R1 head) REJECTED

| leg | image | steps | tok/s/req con-4 deep | verdict |
|-----|-------|-------|----------------------|---------|
| D1  | v1.2.31 (v52k baked) | 154.5ms | 10.3–16.1 | WINNER (165→154.5ms) |
| D2  | v1.2.32 (+R1 sharded view) | 208ms | 9.6–11.8 | REJECTED |

- D2 R1 markers green at boot (rank0 [0:124160) / rank1 [124160:248320)
  zero-copy, drafter table freed) but propose win ABSENT (25.4ms vs promised
  10.6) and net step DEGRADED 208 vs 154.5. KV posture penalty: R1 head eats
  ~2.7 GiB → max_model_len ≤163840 (KV 3.6 vs 5.06 GiB needed @262144).
- D1 storm gate GREEN: 108/108 @1212s = 1747.3 total tok/s (+2.3% vs
  v1.2.30 production 1707.7 @1242s).

## ASYNC WIDE-WAVE ABORT LAW (the round's new law)

v1.2.31 + VLLM_XPU_ALLOW_ASYNC=1 alone: async works (54.2ms steps, 2.85×
sync; warm width-4 clean 23.9–30.6 tok/s/req) BUT wide concurrent waves of
deep (8k) requests abort client-visibly (FINISHED_ABORTED, 0 completion
tokens): width 8 → 5/8 dead; width 18 → 14/18 then 18/18 dead; survivors
≈3–4 regardless of width. NOT JIT/compile-once (second 18-wide wave died
worse, spec counters frozen). Mechanism: the v60c SYNC-path poisoned-row
guard (scheduler.py update_from_output wrap) treats a scheduled request
with an empty/unregistered sampled row as a NaN-zombie sentinel; under
async overlap the final-prefill-chunk→decode transition legitimately
produces 1–2 empty-row steps → strike 1 = immediate FINISHED_ABORTED
(VLLM_V60_ZOMBIE_STRIKES default 1).

**Fix: VLLM_V60_ZOMBIE_STRIKES=3** — strikes are CONSECUTIVE (popped on
first healthy row, scheduler.py:1659) so 3 absorbs the transient lag while
the real NaN-zombie defense stays armed. COUPLING LAW: never ship strikes=3
without async (alone it would weaken the sync guard for nothing).

Proof: AS2 cold 18-wide 18/18 clean ×32tok; warm 18-wide 18/18 ×64tok;
38+ transient strikes absorbed across storms, 0 strike-outs.

## FIX-2 storm matrix — CERTIFIED (AS2R: v1.2.31+async+strikes, fresh boot post-reboot)

| storm | ok | wall | total tok/s |
|-------|----|------|-------------|
| 1 (cold) | 108/108 | 1236s | 1692.1 |
| 2 | 108/108 | 1211s | 1727.1 |
| 3 | 108/108 | 1216s | 1720.1 |

vs v1.2.30 production (1707.7 @1242): +0.7% / −2.1% wall. vs v1.2.31 sync
(1747.3 @1212): −1.6%, inside certified spread (1214–1244s). Zero
strike-outs, errors, resets; 3/3 on ONE engine, no reboot between storms.

## Decode payoff (deep 8k prompts, 512-tok gen, wall-inclusive)

| posture | solo | con-4/req | con-4 agg |
|---------|------|-----------|-----------|
| v1.2.30 sync prod (morning) | — | 12.0 | 39–71 |
| v1.2.31 sync (D1) | — | 10.3–16.1 | ~44–59 |
| **v1.2.31 async+strikes** | **28.1** | **27.0–28.0** | **95–101** |

First drive after lane-state change reads ~17.5 (shape re-warm); warm
repeat = 27–28. Post-storm warm repeat 26.9–27.4 @100.9 agg — win holds.

## PRODUCTION RESTORED 2026-10-06 21:46

- Chain rebased: V1227_IMAGE→v1.2.31, new V1227_ASYNC=1 default (adds
  -e VLLM_XPU_ALLOW_ASYNC=1 -e VLLM_V60_ZOMBIE_STRIKES=3, coupled), banner
  v131-async, new boot asserts (env presence + "Asynchronous scheduling is
  enabled" ×≥1 + zero "disabled"). md5 d37e69cb BOTH copies
  (/root/build + v127_stage).
- Restored via `V1227_SPEC=1 bash boot_v1227_restore.sh ckpt`: patchers all
  ALREADY-APPLIED, HEALTH_OK 140s, KV 370,254 tok, posture verified live.
- Drift check rebased: EXPECT_IMAGE→v1.2.31, EXPECT_SPEC default→1 (spec
  MTPx4 now STANDING — user's live spec window folded in), new EXPECT_ASYNC
  gate (both envs) + ASYNC_OK in verdict conjunction; verdict line
  `V1227_DRIFT_NONE posture=certified-swift-fp8-spec-async` green.
  md5 26215909 (/root/build/v127_stage single copy).
- Watchdog unpaused 21:47 (single instance PID 2735), adopted healthy lane.
- Template: baked in ckpt tokenizer (verified chat_template in
  tokenizer_config.json md5 e8436422) — chain ckpt posture (no
  --chat-template flag) is formatting-identical to the bench/live window.

## Artifacts

- Host /root/build/p84_stage/: boot_p84.sh (md5 30dce398, modes
  clean/inst/async/async_inst/asyncs), p84_decode_drive.py (leg jsonl +
  metrics), D1/D2/AS1/AS2/AS2R drive jsonls + metrics, STORM_AS2_[123].out
  + _rows.jsonl, AS2_serve.log, AS2R_serve.log, R1 masters (4 files).
- Storm driver (unchanged, v130_stage): wsc_pressure_v134_leg7_depth.py;
  extractor p83_bench/toks_sum.py. jsonl is APPEND-shared — per-storm rows
  = `tail -n 108`.
- Round record: .tmp-p76/p84/P84_ROUND.md (+ host copy /root/build/p77/
  convention).

## Deferred

- FIX-3 mixed-eager isolation (UPSTREAM_MTP_RESEARCH phase D) — next decode
  lever after async (eager contamination of prefill-shared steps).
- Async spec-width cap (v42c "1 of 4 slots") — K width pipelining vs slots
  trade unexplored on the new base.
