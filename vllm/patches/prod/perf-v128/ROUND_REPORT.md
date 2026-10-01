# ROUND_REPORT.md — perf-v128 (2026-10-01): Phase-1 closure + the xgrammar conversion-trap law

User directive (carried): "Commit and push and continue -> Do deep
improvement, fixing and testing suite, and baked into new production
image llm-scaler-exp:v* using best values and improvements. Important!
Spec and XGrammar-2 has to have supported. Do not use subagents."

Execution: direct, no subagents. Every leg window: lane quiet-checked,
watchdog paused/resumed via systemctl, certified posture restored and
verified (BOOT_V1227_RESTORED posture=ckpt, health 200) — final state
of the round is the certified lane serving with the folded checkpoint
defaults, watchdog ACTIVE.

---

## 1. What was tested, what passed, what failed

| # | Test | Result | Verdict |
|---|------|--------|---------|
| 1 | Restore-chain knob hardening (128b/128c) + posture default ckpt | boot stamps `xgc=/perreq=`; LEG_XGC2 clean boot; FOLDV restore gates OK | PASS |
| 2 | XGrammar temperature sweep (plain + pattern schemas) | degeneration temp-gated >=0.6, ~50%, flat above; 2 digit modes + minority ws | PASS (measured) |
| 3 | Greedy discriminator (0.0 / 0.1) | 10/10 clean identical `222.5` | PASS — legality machinery SOUND |
| 4 | disable_any_whitespace flag probe (f10/f07) | ws class dead; digits WORSE (14/20, 17/20) | PASS — global flag REJECTED |
| 5 | Recipe probe v1 (bounded schema + flag, temp 1.0) | 0/20 degenerate, max digit run 4 | SUPERSEDED — pattern was corrupt (see §3); luck, not construction |
| 6 | D4 replay + perreq telemetry (8x16 turns) | 128/128 OK, 194 rows harvested | PASS |
| 7 | D4 split analysis | full-hit snap 9.5 tok/req; excess 18.8% of computed; 90.1% eviction | PASS — EVICTION DOMINANT, NO KERNEL |
| 8 | genconfig pre-fold baseline (seed-pinned, 5 reps) | A==B 5/5 exact, A==C 0/5 | PASS — omitted temp resolves to the file's 0.9 |
| 9 | genconfig fold + FOLDV restore + post-fold probe | A==C 5/5 exact, A==B 0/5 | PASS — POST_FOLD_CERTIFIED |
| 10 | Post-fold sanity (thinking, guided JSON, log scan) | t1 reasoning intact; 0 tracebacks; t2 parsed BUT 7-digit value | ANOMALY -> §3 (root-caused, law + fix certified) |
| 11 | Guidance-active check (sentinel ZZ) | 5/6 conform + 1 in-grammar ws-runaway | PASS — guidance ACTIVE on restore lane |
| 12 | Bound discriminator (^A[0-9]{3}Z$ / {16}Z under 7-digit pull) | 12/12 + 4/4 enforced | PASS — group-free bounds exact |
| 13 | Violation reproduction matrix (120 reqs) | optional-group shapes 27/30 & 24/30 violations; group-free 30/30 | PASS — trigger isolated |
| 14 | Grammar membership test (in-container, model removed) | corrupt shape ACCEPTS 7-digit string, both aw modes | PASS — conversion bug PROVEN |
| 15 | Fix-shape matrix (7 shapes x 8 strings x 2 modes) | B1/B3/B4 sound; `\.?` and `{0,4}`-lead corrupt | PASS — law characterized |
| 16 | Recipe v2 lived confirmation (3 arms x 2 temps x 20) | control 0/40; sound shapes 40/40 | PASS — RECIPE_V2_CERTIFIED |

No production regression anywhere: certified posture, spec MTPx4 +
XGrammar-2 crash-free on every boot this round (gates + sanity), fleet
surface unchanged (no production schema uses the corrupt pattern shape).

## 2. Root causes and laws banked this round

1. **Sampling attractor (P47, confirmed):** guided-JSON degeneration is
   a sampling-level attractor on guessy values — flat post-mask
   distributions coin-flip into runaway at temp >= 0.6 (~50%,
   temp-insensitive). Engine legality is sound (greedy clean).
2. **Redistribution law (P47):** killing a legal escape class with a
   global flag does not remove the attractor — it moves probability
   mass into the remaining classes (ws flag -> digits worse). Global
   `disable_any_whitespace` REJECTED; opt-in knob only.
3. **D4 eviction law (P48):** recompute excess is 90.1% eviction /
   9.1% GDN-snap; full-hit snap tax ≈ 9.5 tokens/request. The win is
   cache/pool policy (WS-C), not a boundary kernel. Mod-GRAN fits no
   granularity — no large snap exists to align.
4. **CONVERSION TRAP LAW (P50, NEW):** in xgrammar 0.2.7's
   regex->grammar conversion, **`?`-optionality (group or single atom)
   following a bounded repetition silently destroys the bound.**
   `^-?[0-9]{1,4}(\.[0-9]{1,4})?$` licenses `3141592`. Immune shapes:
   `(...){0,1}`, alternation of full branches, alternation inside the
   group. Proven engine-side (GrammarMatcher membership, model removed)
   in BOTH any_whitespace modes; confirmed lived under maximal pull at
   temp 0.6 and 1.0 (control 0/40 vs sound 40/40).
5. **Composure law:** legality alone does not runaway — under the
   corrupt grammar with a WANTED 7-digit value the model emitted it
   cleanly (finish=stop) 20/20 at both temps; runaway requires a flat
   distribution. The attractor (law 1) and the conversion trap (law 4)
   are independent mechanisms.
6. **Default-sampling fold (P49):** the checkpoint shipped temp 0.9
   (swift export); vendor rec is 0.6. Folded and certified with
   seed-pinned omitted-vs-explicit probes before/after a restore boot
   (5/5 exact both directions). Explicit params always win — CC
   unaffected. Marginal-by-design verdicts: live-traffic batching
   wobbles late tokens (measured 2/3 exact single-pair) — use majority
   margins, never single-pair equality gates.

## 3. The anomaly chain (how the trap was found)

Post-fold sanity t2 returned `"t": "2024052"` (7 int digits,
finish=stop) under a pattern bounding ints to 4. Six-step
falsification chain, each step eliminating one hypothesis:
inactive-guidance -> sentinel proves active; spec-mask-leak ->
group-free bounds enforce 12/12 under maximal pull; flaky -> 27/30
reproducible; engine-illegal -> membership test shows the compiled
grammar LICENSES the string. Root cause: conversion bug. Fix: shape
law + recipe v2. (Full detail in PHASES.md P50.)

## 4. Ship decision

**v1.2.28 NOT REQUIRED.** Every image-level candidate was either
rejected by measurement (global flag), already available via
boot-knobs on the certified image (perreq telemetry, XGCOMPACT),
checkpoint-side (genconfig fold), or client-side documentation (recipe
v2). Baking an image with zero runtime delta would churn the certified
surface against the round's no-degradation law. Production posture
remains **llm-scaler-exp:v1.2.27 certified-swift-fp8** + hardened
restore chain + folded checkpoint defaults. (Precedent: v127 round's
v1.2.28-cancelled record.)

## 5. Traps encountered (new)

- **Bash brace expansion on generated JSON flags:** unquoted
  `{"backend":...}` in a docker-exec sed line split into two words ->
  vllm arg-parse death, health 000 for a whole leg boot. Fix: the
  boot-script generator single-quotes the JSON (fix_restore_v128c.py);
  byte-literals built from raw-string concatenation at quote
  boundaries — no escape arithmetic.
- **Single-pair seed-pinned equality under live traffic:** batch-shape
  nondeterminism flips near-ties at flat temperatures (2/3 exact on
  the TRUE arm). Margin-based verdicts (5 reps, majority + zero on the
  other arm) are the robust form.
- **FinishReason casing:** telemetry emits `.name` (STOP/LENGTH);
  analytics must lowercase before filtering (cost a "no data" run).
- **xgrammar 0.2.7 API in-image:** `Grammar.from_json_schema` /
  `GrammarCompiler(tokenizer_info=...)` + `accept_string` /
  `is_completed` (NOT compile_json_schema-on-Grammar or
  tokenizer_info-on-matcher).
- Write-tool/heredoc: build probe JSON bodies with python (standing
  law re-confirmed: the inline-heredoc attempt died on a scoping bug).

## 6. Artifacts

Scripts (perf-v128/): xg_sweep, xg_greedy_probe, xg_flag_probe,
xg_recipe_probe, xg_active_check, xg_bound_discriminator,
xg_repro_violation, xg_membership_test, xg_fix_shapes,
xg_recipe_v2_confirm, d4_replay, d4_split, patch_perreq,
genfold_probe, fold_genconfig (perf-v128-stage/), sanity_foldv,
fix_restore_v128{,b,c}.

Captures (perf-v128/captures/): xg_sweep_v128.json,
xg_flag_probe_v128.json, xg_recipe_probe_v128.json,
xg_active_check_v128.json, xg_bound_discriminator_v128.json,
xg_repro_violation_v128.json, xg_membership_v128.json,
xg_fix_shapes_v128.json, xg_recipe_v2_confirm_v128.json,
d4_split_v128.json, v128_perreq.jsonl, genfold_probe_v128_{pre,post}.json,
boot_FOLDV.out.

Host (kept): /root/build/v128_stage/ (scripts + v128_perreq.jsonl),
/root/build/lce1/ (probe outputs), hardened
/root/build/v127_stage/boot_v1227_restore.sh +
/root/build/boot_v1227_restore.sh (md5 e2f0b00d, both copies),
/models/qwen3.8-27b-fp8/generation_config.json (folded; .v128prefold.bak).

## 7. Phase-2 outlook (unchanged + one new input)

- REOPEN-A (int8 per-feature scales): offline M2 harness; unchanged.
- REOPEN-B (e4m3 frozen-state spill): document-not-do; unchanged.
- WS-C (eviction/pool policy): now quantified — 90.1% of an 18.8-24.7%
  recompute excess is partial-prefix-miss (~640 tok/affected-request);
  perreq telemetry knob (V1227_PERREQ=1) is the standing instrument.
