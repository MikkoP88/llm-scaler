# v79 (DFLASH2_EMIT_K=1, env-only) — BROKEN AS DEPLOYED: falsification record

Date: 2026-10-04 · Container: lsv-test · Boot: `boot_v79_x4.sh` 14:30:53 · Host fresh (reboot 13:58)

## Boot gates (all GREEN — the trap)

- `DFlash2 v42d dispatcher: uniform_decode_query_len=2 (config width 5)` — both workers (cudagraph_dispatcher.py:62)
- `DFlash2 v42d: uniform_decode_query_len=2 (config width 5) — graphs capture at effective width` (gpu_model_runner.py:1155)
- 34/34 graphs (even-only lattice, multiples of udql=2), 15 s, 0.57 GiB
- KV 294,001 tokens; health 200

## Smoke (1 conv, temp 0, "count 1 to 15") — THREE simultaneous failures

1. **Output garbage from token 1**: `"We can be it is the number of the world:ie:ie the number of thes Kron"` — v78 posture (same ckpt/template/greedy) is coherent.
2. **Acceptance collapsed**: 25 drafts × 4.0 tokens (drafter healthy, width 4), accepted **5** total → 0.2 tok/step (v78 ≈ 2.35 rows/token posture-grade).
3. **Accepted-per-pos 3/1/1/0 at positions 0–3** — positions 1–3 "accepting" is impossible at true width-2 verify (their drafts never enter the forward) → the verify/commit machinery ran width-5 assumptions against a udql=2 world.

Zero ERROR/Traceback in serve_full.log. Log banked: host `/root/build/lce1/v79_broken_smoke_serve.log`.

## Root cause: the cap armed only HALF the width chain

- **v42c (async_scheduler.py placeholder cap) NEVER ENGAGED** — `DFlash2 v42c async cap` line absent from serve_full.log. `AsyncScheduler` is not constructed on the engine variant (`VLLM_XPU_MTP_TP1=engine`); the plain `Scheduler.schedule()` runs (consistent with P77 MIXED-EAGER LAW: isolated sched-2 rows = non-padding scheduler). The v42c comment's "self-consistent end-to-end" claim holds only for the AsyncScheduler wiring, which this stack does not use.
- Net posture: **scheduler emits full width-4+1=5 rows while GMR/dispatcher/lattice run udql=2** — every decode step fails `_is_uniform_decode` (5 ≠ 2) and routes through the non-uniform path with a stride-2-keyed graph lattice; every udql-derived stride/slot/key on the target side is then in undefined-behavior territory. First committed token garbage ⇒ the target forward itself was wrong (state/row bookkeeping), not merely draft comparison.

Live hypotheses (not individually adjudicated — moot once halves agree): pad-dummy-row bookkeeping at udql stride; verify-side draft indexing at udql stride; GDN/mamba state-slot keying by udql (mamba_utils is bind-mounted, R1-era).

## The memory-law correction

The standing law "port the fits-in-drafter gate (A5000 stock lever) — it kills both crash class and wedge exposure" was ALREADY falsified pre-deploy (stock gate = width-5 zeros = 5 rows/token, WORSE than spec's 2.35). v79's env-only lever is now ALSO falsified as-deployed. **A width-2 posture requires BOTH halves: udql=2 derivation (works, env) AND a scheduler-side emission trim (v79b — one site).**

## v79b fix (one site, daily-precedent semantics)

`update_draft_token_ids` (in-image scheduler.py storage site, ~:2103) stores `request.spec_token_ids` verbatim. `num_tokens_with_spec = len(output_token_ids) + len(spec_token_ids)` (request.py:244) ⇒ **trimming the stored row to k makes the spec-block arithmetic (`num_scheduled_spec_tokens = num_new_tokens + computed − num_tokens − placeholders`) schedule width k+1 rows natively** — uniform at udql=2, graphed at the captured lattice, exactly the v42 design contract. Width-varying rows through verify/commit are daily precedent (near-finish clamp → "isolated sched=2 rows", numerically fine — they were just eager).

Residual risk (accepted): rare width-1 rows at request tail could pad to 2 with a dummy row (graphed padded replay unexercised on v78 — mixed batches went eager). Bounded to the final token(s) of a request; watch in smoke/storm.

Drafter still drafts all 4 positions (~10.7 ms propose unchanged); positions beyond k trimmed scheduler-side (v79a loop-skip optimization stays queued).
