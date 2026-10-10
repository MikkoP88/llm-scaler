
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
