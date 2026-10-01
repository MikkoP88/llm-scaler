# PHASES.md — perf-v128 (Phase-1 closure legs: WS-D XGrammar + D4 split + genconfig fold; Phase-2 prereqs)

Round directive (user, carried): "Commit and push and continue -> Do deep
improvement, fixing and testing suite, and baked into new production image
llm-scaler-exp:v* using best values and improvements. Important! Spec and
XGrammar-2 has to have supported. Do not use subagents."

Phase-1 targets (user, verbatim): "async scheduling (banked +85-110%
aggregate), 85 cudagraph sizes (banked +3-4%), WS-D XGrammar load fix
(quiet rate 4/11 — real headroom), D4 per-request split." — both
remaining legs (WS-D, D4) measured and closed this round; one NEW LAW
(xgrammar conversion trap) discovered, proven, and closed with a
certified recipe v2.

All legs: lane quiet-checked before replacement; watchdog paused via
`systemctl stop lane-watchdog` and resumed after; certified posture
restored and verified on the live cmdline each time
(BOOT_V1227_RESTORED posture=ckpt; drift guard surface untouched — the
structured-outputs flag is not in the forbidden set).

---

## P46 — restore-path hardening (host + repo)

**Goal:** make the certified restore chain carry this round's knobs
safely, with the certified posture as the DEFAULT.

- `fix_restore_v128b.py` — added the `V1227_XGCOMPACT` knob block to
  both host copies of `boot_v1227_restore.sh` (boot line stamps
  `xgc=/perreq=`; doc anchor re-based on the real comma-style knob-doc
  line — 128a's doc branch had silently never matched).
- `fix_restore_v128c.py` — QUOTING ROOT FIX of the knob: the generated
  serve line must carry the JSON SINGLE-QUOTED
  (`--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`).
  128b's unquoted blob brace-expanded into two words -> pydantic
  "Invalid JSON" -> serve died at arg-parse (LEG_XGCOMPACT boot 11:35,
  health 000). Verified: fixed sed/gate lines present, `bash -n` OK,
  both copies; stale boot PIDs killed; re-fired clean as LEG_XGC2.
- Posture default flipped to `:-ckpt` (a bare invoke restores the
  certified posture, not the raw-image one).
- Repo copy synced byte-identical to the hardened host copies
  (md5 e2f0b00d6d4b5da0bf11613f6243e51a, `bash -n` OK).

**Verdict:** CLOSED. Trap recorded: bash brace expansion on unquoted
`{"a":"x"}` — single-quote every JSON blob in generated shell lines.

Evidence: fix_restore_v128b.py, fix_restore_v128c.py,
captures/boot_FOLDV.out (post-hardening restore boot), host
/root/build/lce1/boot_LEG_XGC2.out.

## P47 — WS-D XGrammar degeneration: root cause + flag verdict (part 1)

**Goal:** why do guided-JSON requests degenerate (quiet rate 4/11), and
does `disable_any_whitespace` fix it?

- `xg_sweep_v128.py` — temperature sweep on plain vs pattern-bounded
  schemas (11->N arms): degeneration is temperature-gated (>=0.6),
  temp-insensitive above that (~50%), two digit modes (integer-runaway
  `2812138121381...`, fraction-runaway `18.50001234...`) + minority
  whitespace class.
- `xg_greedy_probe_v128.py` — greedy (0.0) and 0.1: 10/10 clean,
  identical `222.5` -> engine legality machinery SOUND; degeneration is
  a SAMPLING-LEVEL ATTRACTOR (flat post-mask distribution on guessy
  values coin-flips into runaway).
- `xg_flag_probe_v128.py` — `disable_any_whitespace` on the flag lane:
  whitespace class DEAD by construction (ws run max 1) but probability
  mass REDISTRIBUTES onto digits -> digit degeneration WORSE (f10 14/20
  vs 11/20; f07 17/20 vs 10/20). **Global flag REJECTED by measurement;
  opt-in knob only (V1227_XGCOMPACT).**

**Verdict:** CLOSED (part 1). Engine-side default unchanged; client-side
recipe is the real fix (see P50 for the corrected v2).

Evidence: captures/xg_sweep_v128.json, captures/xg_flag_probe_v128.json,
xg_greedy_probe_v128.py (+ /root/build/lce1/xg_greedy_probe_v128.jsonl).

## P48 — D4 per-request split: EVICTION DOMINANT, no kernel

**Goal:** split the 24.7%-of-computed recompute excess (P37) between
GDN-snap recompute and eviction re-prefill, per request.

- `patch_perreq_v128.py` — dormant, marker-gated per-request telemetry
  (prompt/generated/cached/computed/finish per request) applied at boot
  via `V1227_PERREQ=1` (V1227_PERREQ hook from P46).
- `d4_replay_v128.py` — 8 conversations x 16 turns, ~2.5k-token shared
  preamble, full history re-send, temp 0.6: 128/128 turns OK,
  max_prompt 9547, wall 265 s -> `v128_perreq.jsonl` (194 rows).
- `d4_split_v128.py` — finish-class normalization fix (telemetry emits
  FinishReason .name; filter lowercased), then split:
  - full-hit GDN snap = **9.5 tokens/request** (negligible; kills the
    boundary-kernel justification outright)
  - recompute excess 18.8% of computed (controlled replay; P37 aggregate
    was 24.7%) = **90.1% partial-prefix-miss/eviction** (~640
    tokens/affected-request), 9.9% GDN snap
  - mod-GRAN law fits NO granularity on full-hit rows (best 1024 by
    mean error but nothing sits at ~0)

**Verdict:** CLOSED — **EVICTION DOMINANT.** The residual win is
cache/pool policy (WS-C track), NOT a GDN kernel. No kernel program.

Evidence: captures/d4_split_v128.json, captures/v128_perreq.jsonl,
d4_replay_v128.py, patch_perreq_v128.py.

## P49 — checkpoint generation_config fold + certification leg

**Goal:** align the checkpoint's default sampling with the vendor rec
(temp 0.9 -> 0.6; top_p 0.95 / top_k 20 already correct).

- Discovery: `/models/qwen3.8-27b-fp8/generation_config.json` ships
  temperature **0.9** (swift export) — the regime with maximal
  digit-attractor entry; P38's "default temperature" degenerations ran
  at 0.9, not 1.0.
- `fold_genconfig_v128.py` — single-key fold with backup
  (`.v128prefold.bak`), JSON-validated reread, collateral-key assert.
- `genfold_probe_v128.py` — seed-pinned A/B discriminator (5 reps;
  omitted-temp vs explicit 0.9 vs explicit 0.6). Margin-based verdicts
  because live CC traffic wobbles late tokens (measured: 2/3 exact on
  the claimed arm with one late-token divergence):
  - PRE-fold (live 0.9 lane): A==B 5/5 exact, A==C 0/5 -> PRE_BASELINE_OK
  - FOLDV restore boot (engine re-read the file; posture=ckpt, gates OK,
    health 200)
  - POST-fold: A==C 5/5 exact, A==B 0/5 -> **POST_FOLD_CERTIFIED**
- `sanity_foldv_v128.py` — t1 thinking intact (reasoning field, stop);
  t2 guided JSON parsed. (t2's 7-digit value triggered P50 — see below.)
- Explicit request params always win (protocol fills only omitted
  fields): CC fleet unaffected.

**Verdict:** CLOSED — folded and certified. NOT a degeneration fix
(0.6 still ~50% digit-class on unbounded schemas); default-quality
alignment only.

Evidence: captures/genfold_probe_v128_pre.json,
captures/genfold_probe_v128_post.json, captures/boot_FOLDV.out,
sanity_foldv_v128.py.

## P50 — NEW LAW: xgrammar optional-group conversion trap + recipe v2

**Trigger:** P49's sanity t2 emitted a 7-int-digit value
(`2024052`, finish=stop) under `^-?[0-9]{1,4}(\.[0-9]{1,4})?$`.

Chain of proof (each step a separate probe):
1. `xg_active_check_v128.py` — guidance IS active on the restore lane
   (sentinel `^ZZ[0-9]{2}$` conformant 5/6 + one IN-GRAMMAR ws-runaway);
   the bounded pattern violated 2/6 with clean stops. So neither
   inactive guidance nor freeform luck.
2. `xg_bound_discriminator_v128.py` — `^A[0-9]{3}Z$` 12/12 and
   `^A[0-9]{16}Z$` 4/4 enforced under a MAXIMAL 7-digit pull (grammar
   truncated the wanted constant to `A314Z`). Bounds enforce for
   group-free patterns -> the anomaly is pattern-shape-specific, not a
   spec-decode mask leak.
3. `xg_repro_violation_v128.py` — 120-request matrix under pull:
   `(\.[0-9]{1,4})?` shapes violate 27/30 and 24/30 (model's 7-digit
   constant emitted verbatim); the no-optional-group shape is 30/30
   clean. Trigger isolated to the OPTIONAL GROUP.
4. `xg_membership_test_v128.py` (in-container, model removed):
   `GrammarCompiler.compile_json_schema` + `GrammarMatcher.accept_string`
   — the corrupt shape's compiled grammar **ACCEPTS `{"t": "3141592"}`**
   in BOTH any_whitespace modes; group-free shapes REJECT it. The
   violation is grammar-LICENSED: a deterministic xgrammar 0.2.7
   regex->grammar conversion bug.
5. `xg_fix_shapes_v128.py` — shape matrix (7 shapes x 8 strings x 2 aw
   modes): SOUND = alternation-of-full-branches, alternation-inside-
   group, and **`{0,1}` instead of `?`**; a lone optional ATOM (`\.?`)
   is corrupt too; `{0,4}`-lead is partially corrupt (leaks 5-7).
   **LAW: `?`-optionality (group or atom) after a bounded repetition
   silently destroys the bound. `{0,1}` and alternation forms are
   immune.**
6. `xg_recipe_v2_confirm_v128.py` — lived confirmation: corrupt control
   0/40 conform (pull sails through at 0.6 AND 1.0); both sound shapes
   **40/40 conform** -> **RECIPE_V2_CERTIFIED**.

**Retroactive correction:** P47's recipe probe (0/20 "by construction")
ran the CORRUPT pattern — its digit-bound protection never existed;
0/20 was weak-pull luck. Recipe v2 is the first version with a
grammar-level digit bound that exists. The ws-class and
redistribution conclusions of P47 are unaffected (measured on
patternless schemas). Composing law: legality alone does not runaway —
the corrupt grammar emitted a clean WANTED 7-digit value (C0 20/20
`3141592`, finish=stop); runaway still requires a flat distribution
(guessy values). Attractor law and conversion trap are independent.

**RECIPE v2 (certified):**
- numeric string fields: `^-?[0-9]{1,4}(\.[0-9]{1,4}){0,1}$` (B4; or the
  B1 alternation form) — NEVER `(...)? ` after a bounded run
- plus `disable_any_whitespace` (opt-in, per P47: kills the ws class;
  do NOT make it a global default)

**Verdict:** CLOSED. KNOWN_ISSUES #31 records the trap. No fleet
surface affected (no production schema uses pattern-with-optional-group;
probes only).

Evidence: captures/xg_active_check_v128.json,
captures/xg_bound_discriminator_v128.json,
captures/xg_repro_violation_v128.json, captures/xg_membership_v128.json,
captures/xg_fix_shapes_v128.json, captures/xg_recipe_v2_confirm_v128.json.

---

## Ship decision — v1.2.28 NOT REQUIRED (no image-delta justified)

Measured, image-relevant candidates this round:
- global `disable_any_whitespace` default — REJECTED by measurement (P47)
- perreq telemetry — dormant marker-gated, boot-knob apply PROVEN (P48);
  baking it changes certified source surface for zero runtime change
- XGCOMPACT knob — boot-script-side, image already supports the flag
  (behavior changed on the flag lane with the same image)
- generation_config fold — checkpoint-side, certified in place (P49)
- recipe v2 — client-side documentation (this file + KNOWN_ISSUES #31)

Nothing survives as an image delta -> no bake, per the round's own law
(no-degradation + measurement-gated surface churn; precedent: the v127
round's v1.2.28-cancelled record). Production posture remains
llm-scaler-exp:v1.2.27 certified-swift-fp8 with the hardened restore
chain (P46) + folded checkpoint defaults (P49).

## Open items (Phase-2, unchanged)

- REOPEN-A: int8 + per-feature scales regime measurement (M2 harness,
  perf-v127/scaledspace/) — offline, gated on tolerance law.
- REOPEN-B: e4m3 spill for frozen/preempted states — document-not-do.
- WS-C eviction track now has a quantified target from P48 (90.1% of
  the 18.8-24.7% excess is partial-prefix-miss; perreq knob available
  for the measurement legs).

### Ship-on-delta policy (when Phase-2 lands an image)

An image is baked only when a Phase-2 track produces runtime content
(the trigger), and the dormant riders fold into that SAME bake:

- **Triggers** (any one justifies the next image, v1.2.28+):
  REOPEN-A success (int8 scaled-state kernel + code), a WS-C policy
  change needing baked engine defaults, or an engine-side xgrammar
  conversion fix.
- **Riders** (fold in only when a trigger bake already runs; riders
  never justify a bake on their own): the dormant perreq telemetry
  patch (P48 instrument — consolidates boot-side source surface into
  the image, still marker-gated dormant).
- **Never image-side** (by classification, not omission): genconfig
  fold (checkpoint mount), recipe v2 (client docs), XGCOMPACT
  (boot-knob; engine already supports the flag).
- If no trigger fires (REOPEN-A fails the regime gate, WS-C resolves
  config-side), no image is baked — by design.

