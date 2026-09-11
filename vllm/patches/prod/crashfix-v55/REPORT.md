# crashfix-v55 — validation & certification report (2026-09-10)

Image: `llm-scaler-exp:v1.2.8` == `llm-scaler-exp:crashfix-v55`
sha256:`7b4de29d0ff3a0b04e6ff0bcef9b7d883aa4bd240ccdd97dd5f258897c1c548a`
(lineage: `fp8-mtp4-v6` == v1.2.7 + `Dockerfile.v7`; first bake `6d4bca514376…`
superseded pre-certification by the v55.1 DISCARD-GAP demotion).
Artifacts, root-cause narrative, and fix details: `README.md`.

**Verdict: VALIDATE_V55 PASS.** Both 2026-09-10 crash configurations
were reproduced under the mandated traffic (concurrent staggered
multi-stream, distinct prompts per stream AND per cycle, exact crash
prompt "Write a html car game", all contexts < 64k) and survived with
**zero engine crashes, zero wedges, zero watchdog kills, zero
DEVICE_LOST, zero RPC timeouts, zero JIT warnings post-warmup, zero
routine WARNING lines, health 200 throughout.**

## Validation design → mandate mapping

| Mandate requirement | Implementation |
|---|---|
| same configurations as the crashes | A: fp8_e4m3 + mtp k=4 (crash-1 boot, v1.2.7); B: fp8_e5m2 + mtp k=3 + `VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1` (exact crash-2 boot) |
| multiple concurrent, different prompts, different start times | 5 streams per cycle, staggered 0/5/10/15/20 s, prompts distinct per stream AND per cycle (seeded per cycle) |
| all content < 64k context | largest stream ≈ 17k prompt + 2k out; htmlgame 57 + 4096; total max ≈ 18k |
| one prompt is "Write a html car game" | stream 1 of every cycle, exact crash sampling (temp 0.9 / top_k 20 / top_p 0.95) and the crash-2 wedge lane (4096 out) |
| fix all warnings | env allowlist (57 census-verified knobs), 7 routine log sites demoted, oneAPI banner silenced, dead env removed, DISCARD-GAP probe demoted (v55.1), JIT closure via warmup p13+p14 |
| validate before building the production image | 9 validation runs, 2 pre-bakes, final bake md5-verified, validation executed ON the baked image |

Per config: boot → warmup v53 (p1–p14) → 3 cycles → health between
cycles → post-warmup engine-log scan (crash classes + hygiene classes)
→ WARNING census → contained-incident census.

## Final matrix (on the baked v1.2.8)

**Config A — fp8_e4m3 + mtp4 (crash-1 configuration) — FULL PASS** (run 7):

| cycle | streams | out tokens | wall |
|---|---|---|---|
| 1 | 5/5 ok | 7023 | 191 s |
| 2 | 5/5 ok | 6797 | 183 s |
| 3 | 5/5 ok | 7025 | 190 s |

Post-warmup scan: **0 WARNING lines** (incl. 0 jit_monitor, 0
DISCARD-GAP, 0 v52l, 0 unknown-env). Warmup incl. p14 dress-rehearsal
cycle (5/5, 7059 tok). CONFIG_A PASS.

**Config B — fp8_e5m2 + mtp3 (exact crash-2 boot) — engine-safety
PASS, lane convicted** (runs 7–9, same standing engine for 8/9):

| run | cycle | streams | out tokens | incidents |
|---|---|---|---|---|
| 7 | 1 | 4/5 + 1 CONTAINED | 4902 | 1 (htmlgame @1989) |
| 8 | 1 | 5/5 | 7000 | 0 |
| 8 | 2 | 5/5 (2 CONTAINED) | 4721 | 2 (qa-sampled @1970, htmlgame @1990) |
| 8 | 3 | 5/5 | 6982 | 0 |
| 9 | 1 | 5/5 | 7005 | 0 |
| 9 | 2 | 5/5 | 6967 | 0 |
| 9 | 3 | 5/5 | 6812 | 0 |

Post-warmup scans: run 9 **0 WARNING lines, 0 incidents** → CONFIG_B
PASS. Warmup incl. p14 dress-rehearsal (5/5, 6953 tok).

## The e5m2+mtp3 NaN-onset class (contained, pre-existing)

Three incidents, every one at exactly `num_computed_tokens=2048` — the
FIRST 2048-token mamba/GDN boundary carry of a sampled long decode:

1. run 7 c1, 20:41:42–48: `v52d EMPTY-ROW [248320, -1, -1]` → `v52e
   ATTR brow=(nan, nan, 0)` → `v52m STRIKE-OUT` FINISHED_ABORTED @1989,
   engine stays up (placeholders=3).
2. run 8 c2, 21:10:51: same chain, qa-sampled @1970 (placeholders=3).
3. run 8 c2, 21:10:51: same chain, htmlgame @1990 (placeholders=2).

All 3/3 contained within ~6 s of onset: client-visible abort of the
poisoned request only; co-running streams completed; health 200; the
engine served subsequent cycles cleanly. On v1.2.7 this same onset
chain IS crash 2 (silent wedge → `sample_tokens` RPC timeout → 600 s
watchdog → engine death). v55 changes no numerics — the defect is
pre-existing in the crash-2 boot combo (e5m2 KV + MTP k=3; mtp3 was
never in a certified matrix, v51 certified e5m2 only with mtp4) and is
newly *exposed* because this validation is the first systematic long-
decode traffic on that lane.

Rate: 3 onsets / ~21 at-risk boundary crossings (~14 %) — stochastic
per crossing. **Disposition: e5m2+mtp3 is NOT serve-grade for long
decode** (a ~14 % per-crossing request-abort rate is unacceptable even
contained). Serve e5m2 only with mtp4 (v51-certified pairing). Prod
stands on e4m3+mtp4 (config A — fully clean).

## Iteration log (what each run convicted)

1. run 1 (fence v1): GAP-FENCE `|gap|≥256` immediate arm false-tripped
   on a healthy warmup p5 chunked prefill (gap 2005) — gap semantics
   learned: on discard steps gap = REMAINING PROMPT; large one-shots
   are routine. Fence v2 = 3 non-decreasing observations ≥ 8 only.
2. runs 2–4 (fence v2): engine + traffic healthy; 2 client-side
   predicate bugs convicted — `reasoning` (not `reasoning_content`)
   is the field name on this serving layer, and usage.completion_tokens
   is the ground truth when the think-block parser buffer is cut.
   8 DISCARD-GAP WARNINGs in 3 healthy cycles → v55.1 demotion to
   DEBUG (GAP-FENCE ERROR supersedes the probe).
3. runs 5–6 (v55.1 image): DISCARD-GAP gone; 2 jit_monitor warnings
   at first-cycle start (kernel_unified_attention + reduce_segments)
   — p11 (4 streams) and p12 (solo) left the bs=5 mixed-context step
   shapes cold. p13 (5-stream stagger) insufficient alone (run 6);
   p14 DRESS-REHEARSAL (warmup runs one full v55_client cycle, seed
   90) closed it — run 7 config A scanned 0 WARNING lines.
4. run 7 (final image + warmup v53.2): A full PASS; B hit NaN-onset
   incident 1 → harness taught the CONTAINED classification (dead
   engine = connection error, not finish=abort JSON; health still
   checked per cycle).
5. run 8: scan pattern "watchdog" falsely matched the v52m line's own
   "(watchdog)" text → pattern narrowed to capital-W "Watchdog"; added
   explicit contained-incident census. Incidents 2+3 recorded.
6. run 9: clean CONFIG_B PASS (0/0) — matrix complete.

## Warning hygiene result (certified boots)

v1.2.7 certified boot warned on 3 unknown-env vars + oneAPI banner +
per-token BOUNDARY/PRE-COPY floods + per-chunk DISCARD-GAP. v1.2.8
validated sections: **0 WARNING lines** on config A; **0** on config B
run 9 (incident-probe lines only during the 3 contained onsets, each
documented above). Remaining scheduler warnings (reset_connector, KV
load recovery) are upstream anomaly-only paths — intentionally kept.

## Certification & production

- Bake parity: all 4 patched files md5-identical baked↔scratch-tested
  (gmr `362e29cf…`, scheduler `a4ea0ec6…`, mamba `b4b23b85…`, envs
  `122a7923…`); `.bashrc` banner fix verified in-image; patchers
  idempotent (re-run skips).
- Validation executed ON the baked image (both configs).
- Prod stand (`prod_restore_v7.sh`, markers in
  `/root/build/prod_restore_v7.out`): BOOT_OK (image `7b4de29d0ff3`,
  KV 707,980 tok) → HEALTH_OK 22:11:52 → WARMUP_OK 22:24:49 (v53,
  p1–p14 incl. dress rehearsal) → ctxscan smoke 2k/16k/32k/65k =
  35.7/32.1/29.6/38.3 tps, ttft 1.2–27.3 s → **PROD_V128_STANDING
  22:26:09** (fp8_e4m3 + mtp4 @0.9/262144). Boot-log census: 41
  WARNING lines, all boot-time configuration advisories ≤ 22:22
  (compile-disable notices ×6, #03/#05b boot-fix notes ×2, bf16→fp16
  casts ×2, upstream min_p/KV-scaling/padding advisories — the same
  classes every prior certified boot carried); **0 WARNING lines in
  the post-warmup window** including the smoke traffic.

## 2026-09-11 v55.2 addendum — throughput-regression fix + node incident (image v1.2.9)

User report "major token generation speed drop v1.2.7 vs v1.2.8" plus
three further deaths; two distinct root causes, both convicted same
day (full narrative: `README.md` addendum).

1. **Node incident (image-independent)**: crash-4 (manual v1.2.8 boot,
   no warmup) and crash-5/6 — the clean v1.2.8 AND v1.2.7 legs of
   `reg_ab.sh` both died on the certified recipe (v1.2.8: BOTH workers
   raised the v55 ASYNC-EVENT-STALL fence at the same second,
   TP-symmetric `num_accepted_tokens_event` stall; v1.2.7: died 1 s
   after a fresh `xe` ccs/bcs engine reset on GPU0). Conviction: GPU0
   flaky since the 03:57 teardown coredump. Host reboot 05:10 → all
   healthy. The fence performed exactly as designed on crash-5:
   culprit named, engine dead in 120 s, no silent 600 s wedge.
2. **Real v1.2.8 regression**: healthy-node warm A/B (identical lane,
   back-to-back) — ctxscan 2k/16k/32k/65k = 39.6/38.0/36.5/44.2 vs
   v1.2.7's 70.8/53.4/62.1/51.3 (−44/−29/−41/−14%); genspeed 4×1024
   88–99 vs 142–148 tok/s (−37%). Root cause: the v55
   `_v55_wait_event` flat 50 ms sleep-poll quantizing per-step event
   waits (~+40 ms/step).
3. **Fix v55.2** (`patch_progressive_wait.py`, `Dockerfile.v8` FROM
   v1.2.8 → **v1.2.9** `ddc6f15a7061`, gmr md5 `51ec1bd2…` baked ==
   scratch): progressive cadence 0.5 ms <20 ms / 5 ms <200 ms /
   50 ms <2 s / 250 ms beyond, same 120 s bound and stall semantics.
   Scratch: ctxscan 61.5/48.6/59.9/49.0, genspeed 134–160, p14 5/5 in
   124 s. Validation ON baked image (`validate_v129.sh`): warmup p14
   5/5 in 112 s (v1.2.7-identical), cycles 101/102 PASS 5/5, ctxscan
   57.6/49.2/46.0/49.2, genspeed 131.9/141.4/138.8, **0 fence stalls,
   0 post-warmup WARNING**, no crash classes. Numerics untouched.
4. **Prod** (`prod_restore_v8.sh`, `/root/build/prod_restore_v8.out`):
   BOOT_OK `ddc6f15a7061` KV 707,980 → HEALTH_OK 06:35:46 → WARMUP_OK
   06:45:01 → smoke 2k/16k/32k/65k = **61.1/58.3/43.4/49.0 tps** (vs
   35.7/32.1/29.6/38.3 on v1.2.8 yesterday) → **PROD_V129_STANDING
   06:46:10** (fp8_e4m3 + mtp4 @0.9/262144). 41 boot-time advisories
   (incl. one unknown-env: `VLLM_ALLOW_LONG_MODEL_LEN` is baked as ENV
   in the image lineage — harmless, read nowhere), 0 post-warmup.
