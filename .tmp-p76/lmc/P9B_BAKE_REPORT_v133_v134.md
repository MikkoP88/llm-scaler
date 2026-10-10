# P9b bake report — v1.2.33 + v1.2.34 (SAVE_DECODE_CACHE standing + double-unpin ledger + upstream logging)

Date: 2026-10-10 · Host: 10.20.3.65 · Container: lsv-test · Engine: lsv65 TP2

## Image lineage (python-only law, .so 82ce75c7 unchanged throughout)

| Image | ID | Contents |
|---|---|---|
| v1.2.32 | adcf2071b646 | P9 baseline (bake gate MET round) |
| v1.2.33 | 88e0d6838719 | adapter_v140 (per-step double-unpin dedupe + 687-line TEMP debug strip, deployed BOTH vllm + lmcache adapter paths); scheduler LWSDbgMatch block stripped; .bak_v133/.bak_v138 removed; P8C §6 hygiene |
| v1.2.34 | a9f325d57e33 | adapter_v141 (13 per-request INFO→debug); mamba_utils (8 demotions incl. v52f PRE-COPY/POST-COPY/DEF-PP, LMCNANDETECT→error, 3 ERR-path→warning); cache_engine epoch-ledger double-unpin fix; LMCACHE_LOG_LEVEL=WARNING in serve script |

Key md5s (container, live-verified after bake):
- vllm_v1_adapter.py (both paths): dcb23fbe0e8990a0b35d6367a8542f3f (v141)
- cache_engine.py: bb3c905e68d05f314042d0f563171422
- mamba_utils.py: 1a6ae0ec16a02d1ccc87755477711fc9
- adapter_v140.py source: 4005321e68e144c31c5e1180c546cdaf
- ce_unpin_fix.py: e850d396a128fee0684806aa12d3765b

## 1. SAVE_DECODE_CACHE A/B (user directive, legs sdoff/sdon on v1.2.33)

| Phase (median TTFT) | False (baseline) | True | Delta |
|---|---|---|---|
| warm | 0.55 s | 0.31 s | **−45%** |
| churn (112 docs ~573k tok) | 0.65 s | 0.63 s | −3% |
| rehit | 0.49 s | 0.47 s | −4% |
| solo200 decode | 76.2–77.0 tps | 76.2–77.0 tps | parity |
| store(final) lines | 514 | 256 | −50% (incremental decode stores) |
| Double unpin warnings | 188 | 0 | killed |

Mechanism: decode-step stores complete chunk-boundary coverage. With False the
trailing partial chunk (doc-end + response) is never stored, so turn-N+1 lands
one chunk shallower. Turn-N response cached → turn-N+1 rehit. Zero regressions.

**Standing shape flipped to LMCACHE_SAVE_DECODE_CACHE=True.**

## 2. Double-unpin root cause (sealed) + epoch ledger

With async scheduling a request re-enters lookup across steps under the same
lookup_id. Step N's lookup_unpin pops lookup_pins[id] and batch-unpins, but the
id's LOADING/DONE event persists; the duplicate unpin takes the elif
event-cleanup branch (cleanup_memory_objs) and re-releases →
memory_management.py:818 pin −1 warning. Adapter-side per-step dedupe (v140)
alone left 188 (rate-similar to pre-fix 130 — cross-step, not within-step).
Definitive fix in cache_engine (ce_unpin_fix.py, baked v1.2.34):
`_lookup_released` set — marked on pins-path release, discarded on repin,
elif-guard skips event-cleanup for already-released ids. sdv134 leg: 0/0.

## 3. Upstream logging alignment (user directive, v1.2.34)

~9000 non-upstream INFO lines/session classified. Rules applied: per-step /
per-request instrumentation → logger.debug (13 sites in adapter, 8 in
mamba_utils incl. v52f PRE-COPY/POST-COPY/DEF-PP + COPY-TIMER/LMCBOUND/seed
write); faults → error/warning (LMCNANDETECT→error; LMCBOUND/POOL/DbgFwd
ERR→warning); boot/lifecycle one-shots stay INFO (EAGLE detected, Registering
KV caches, LMCache initialized, Hybrid register, Mamba pool/dtype, chunk_size,
LMCPOOL init); lmcache-native volume via LMCACHE_LOG_LEVEL=WARNING; version
tags (v52f/v52k) dropped from active surfaces. Post-bake verification: serve
log 226 lines, 0 non-upstream markers; visible INFO = uvicorn access lines
incl. litellm 10.100.8.6 probes only.

## 4. v1.2.34 qualification leg (sdv134, bench-only vs running engine, True shape)

Records (p2/p3):
- warm 0.31 / 0.30 s (sdon 0.31 ✓)
- churn 0.64 / 0.64 s (sdon 0.63 ✓)
- rehit 0.49 / 0.48 s (sdon 0.47 ✓ within noise)
- solo200 ×4: 76.08 / 76.68 / 76.17 / 76.46 tps (sdon 76.2–77.0 ✓)
- COUNT[Double unpin]=0 · COUNT[LMCNANDETECT]=0 · COUNT[Traceback]=0
- HEALTH=200 · leg 16:34:41→16:46:55, all P1/P2/P3/R1/R2 EXIT=0

## Standing serve shape (v1.2.34)

LMCache ratio 2, kv fp8_e4m3, mamba float16, MTP×4, async,
LMCACHE_SAVE_DECODE_CACHE=True, LMCACHE_LOG_LEVEL=WARNING, PYTHONHASHSEED=0,
gmu 0.8, 60GiB×2 CPU tier lr_u. Serve script /root/serve_lmc_v133.sh
(synced container-side; banked /root/build/serve_lmc_v134_bank.sh).

## Post-bake standing-shape change (2026-10-10 17:17 UTC, user directive)

`LMCACHE_LOG_LEVEL` flipped WARNING→INFO on the running instance (clean TERM
restart, no -9 needed) and propagated everywhere the posture is named:
serve master `/root/serve_lmc_v133.sh` + `/root/build/serve_lmc_v133.sh` +
`/root/build/serve_lmc_v134_bank.sh` (all md5 005e8895), boot-chain assert
line 273 in BOTH copies (new chain md5 d95201c5, bash -n clean), and this
report. INFO restores lmcache-native volume (lookup/store/evict lines) to the
serve log; the v1.2.34 demotions of OUR instrumentation stay debug.
The WARNING numbers in §3/§4 above remain the bake-time record.

## Post-bake standing-shape change 2 (2026-10-10 17:42 UTC, user directive)

`LMCACHE_MAX_LOCAL_CPU_SIZE` 60→100 (per rank; ×2 ranks = 200 GiB CPU tier).
Sized on live measurement: host 376 GiB total, non-pool lane footprint
~100 GiB (2×~45 GiB workers + API/EngineCore + OS/litellm) → 100/rank leaves
~76 GiB headroom; max safe = 110/rank. Bench corpus (~560 chunks ≈ 31 GiB/rank
at the measured ~0.055 GB/1024-tok rate) already fit in 60 — the raise pays
off only on a larger live accumulating corpus (lr_u cycles later). Propagated:
serve master + container copy + serve_ctl_v133.sh + /root/build banks
(md5 6b455e8d for the master), repo bank, this report; clean TERM restart.

## Artifacts banked

Host /root/build/: adapter_v140.py, adapter_v141.py, ce_unpin_fix.py,
leg_p9sd.sh, cache_engine_v141.py, mamba_utils_v134.py,
serve_lmc_v134_bank.sh, bench_{sdoff,sdon,sdv134}_*, leg_{sdoff,sdon,sdv134}.out,
telem_bench_{sdoff,sdon,sdv134}_serve.log.
Repo .tmp-p76/lmc/: same source set + this report.

## Post-bake round: P10 prefill-stall fix baked v1.2.35 (2026-10-10 ~21:00 UTC)

User directive: fix prefill/decode issues from the live-traffic analysis, full
validation, test `LMCACHE_CHUNK_SIZE=256` + `LMCACHE_MIN_RETRIEVE_TOKENS=256`,
then bake.

**Root cause (proven):** long-context chunked prefill (>=16k = multi-step past
the MNBT 8192 first step) parks the worker 23-33 s per chunk-step at the two
`num_accepted_tokens_event` entry waits in gpu_model_runner.py
(`_update_states` + deferred postprocess). The event completes but its
visibility lands absurdly late while the engine round-trip idles. NOT
concurrency (Running:2 during the 418 s prefill), NOT queueing, NOT LMCache
stores (13 ms). Proof: py-spy freeze-frames (worker in `_v55_wait_event`,
engine in get_response) + store cadence (first 8192-step 3.7 s, then 23 s and
26 s gaps, five 1.05 s chunks, ~33 s at the prefill->decode boundary) + length
curve threshold exactly at MNBT 8192.

**Fix (P10, p10_patch.py):** `_p10_wait_counts` — poll the event 250 ms
(normal path <20 ms unchanged); if late, spawn ONE helper daemon thread doing
a direct blocking D2H copy of the tiny counts slice, proceed when either
lands. Empirically a queue-kick: 24/24 resolutions "via event" 0.05-0.10 s
after the helper spawns (enqueueing the copy flushes the L0/oneAPI runtime,
making the event's completion visible). v55.3 crash bound preserved verbatim
— no new wedge exposure.

**Validation (leg_p10cs p10final, 256 reverted to standing 1024/1024):**
- longctx cold (fresh boot): 16k t1 89.5->27.1 s (3.3x), 32k t1
  418.4->27.4 s (15.3x); t2 rehits 0.28-0.48 s across all axes.
- P9b gates parity: warm 0.32 / churn 0.63 / rehit 0.47 s, solo 74.2-75.1
  tok/s, agg4 239-261 tok/s 4/4 clean.
- agg4 concurrent decode ~60 tok/s/req vs 33 pre-fix (+85%).
- Mechanism counters clean: 0 Traceback / 0 Double unpin / 0 NaN / 0 Pin
  timeout; [P10] resolution logs all sub-0.1 s post-kick.

**256/256 env question: REJECTED.** Every axis 2.5-4x worse (warm 1.15 s vs
0.31; 16k 225.8 s; 32k >600 s timeout) plus mechanism breakage: POISONED 137,
Double unpin 4350, engine-side shm_broadcast dequeue TimeoutError, run7n
streams dead. Cause: fixed per-chunk overhead x4 boundaries. Standing stays
CHUNK_SIZE=1024 / MIN_RETRIEVE_TOKENS=1024 (serve md5 6b455e8d unchanged).

**Bake:** `docker commit lsv-test llm-scaler-exp:v1.2.35` = `1a611c68a2c6`.
Certified md5s: adapter dcb23fbe / cache_engine bb3c905e / mamba_utils
1a6ae0ec (unchanged); NEW gpu_model_runner.py =
`6520f91686ac8db3d296595bb43c13e2` (3 `_p10_wait_counts` markers verified).

**Propagation:** chain boot_v1227_restore.sh md5 `c5900b92` (BOTH host
copies + repo prod bank; V1227_IMAGE default v1.2.35, new asserts
LMCACHE_CHUNK_SIZE=1024 + LMCACHE_MIN_RETRIEVE_TOKENS=1024, label v135-lmc;
bash -n clean). Drift check `55d4d6aa` (EXPECT_IMAGE v1.2.35 + new
EXPECT_GMR_MD5 content gate in the verdict conjunction; verdict
`certified-lmc-v135-spec-async-tmpl`; dry-run certified live 20:57:37 UTC).

**Artifacts:** /root/build/p10_patch.py (6fcbaa7b), leg_p10cs.sh,
bench_longctx.py (axes small/mid8k/mid16k/long32k), leg_p10cs_cs256.out,
leg_p10cs_p10final.out, P10_pyspy/ freeze-frames; repo mirror
.tmp-p76/lmc/.
