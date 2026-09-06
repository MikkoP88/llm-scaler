# v50 validation matrix — RESULTS (running)

All cells: v1.2.1 + v50 overlays (core resync + pool clamp), TP1=0,
dflash k7, memutil 0.92, warmup off, same battery (`v50_cell.sh`):
dt_bench (prefill/decode tok/s @2k/16k/65k), conc_n_probe 2×3 temp-0
determinism, canonical cargame 512 (text + usage), acceptance snapshot,
error scan (Traceback/Assertion/clamp).

| cell | dtype | bs | MAXLEN | prefill 2k/16k/65k | decode 2k/16k/65k | det | cargame | acc (per-pos) | errs |
|---|---|---|---|---|---|---|---|---|---|
| v50_tq4nc_bs512_match | turboquant_4bit_nc | 512 | 262144 | 877/1840/1338 | 46.4/17.7/9.1 | exact ×3 | coherent | 0.82/0.48/0.33/0.28/0.25/0.18/0.17 | 0 |
| v50_k8v4_bs512_match | turboquant_k8v4 | 512 | 262144 | 627/1840/1355 | 42.1/14.7/6.9 | exact ×3 | coherent | 0.86/0.58/0.46/0.42/0.38/0.30/0.26 | 0 |
| v50_tq4nc_bs512_ovr_k8v4 | tq4nc target, k8v4 draft (override; certified prod cell) | 512 | 262144 | 632/1837/1344 | 43.2/16.4/9.2 | exact ×3 | coherent | 0.81/0.43/0.30/0.26/0.25/0.16/0.15 | 0 |
| v50_k3v4nc_bs512_match | turboquant_k3v4_nc | 512 | 262144 | —/—/— | 2k —/18.6/10.8 | exact ×3 | coherent | 0.77/0.40/0.24/0.21/0.21/0.11/0.11 | 0 |
| v50_tq3nc_bs512_match | turboquant_3bit_nc | 512 | 258048 | 597/1807/1316 | 48.4/16.7/10.0 | exact ×3 | coherent | healthy | 0 |
| v50_fp8_bs512_dflash7 | fp8_e4m3 | 512 | 262144 | PATHOLOGY | 0.3–0.9 (hist 34/31/22) | — | stalled | 0.596→? | ENGINE RISK |
| v50_fp8_bs512_dflash7_STOCK | fp8_e4m3 (stock v1.2.1, NO overlays) | 512 | 262144 | — | ENGINE DEATH on 1st req | — | HTTP 500 | — | sample_tokens RPC timeout |
| v50_fp8_bs512_nospec | fp8_e4m3 + nospec | 512 | 262144 | 653/1838/1151 | 34.4/30.7/22.4 | exact ×3 | coherent | n/a (nospec) | 0 |
| v50_fp8_bs512_mtp4 | fp8_e4m3 + MTP k4 | 512 | 262144 | CRAWL | 0.2–0.5 (bench stalled) | — | — | acc len 2.5 @ 0.3 tok/s accepted | convicted |
| v50_r6b2_crashcell_tp1 | tq4nc (TP1=1 — exact R2 crash cell, final core incl. F6a+probe, correct EXTRA_VOL slots) | 512 | 262144 | TP1-degenerate | 65k 5.81 (corrupt-lane speed) | exact ×3 | coherent, usage present | 0.45/0.18/0.03 (degenerate, expected) | 0 |
| v50_r6c_prod_ovr_k8v4 | tq4nc + k8v4 draft (certified prod cell, FINAL core 3581a600, correct slots) | 512 | 262144 | 644/1845/1345 | 44.7/15.9/9.13 | exact ×3 | coherent, usage present | healthy | 0 |
| v50_tq4nc_bs2048 | tq4nc (match draft) | 2048 | 262144 | 625/1839/1343 | 51.2/16.6/9.33 | exact ×3 | coherent, usage present | healthy | 0 |
| v50_tq4nc_bs64 | tq4nc (match draft) | 64 | 262144 | — | 47.8/15.1/9.16 | exact ×3 | coherent, usage present | healthy | 0 |
| v50_tq4nc_bs512_mtp4 | tq4nc + MTP k4 | 512 | 262144 | — | 57.2/25.6/11.9 | **distinct=2/2 ×3** | coherent (temp 0.3) | machinery healthy; **temp-0 CORRUPT** | 0 (no engine errs) |
| v50_emitk5_prod | tq4nc + k8v4 ovr + EMIT_K=5 (dflash7) | 512 | 262144 | 624/1835/1342 | 43.55/22.08/11.25 | exact ×3 | **BROKE: 1 token then 18.2 s stall** | temp-0 CORRECT (' Paris.') | **3** |
| v50_emitk4_prod | tq4nc + k8v4 ovr + EMIT_K=4 (dflash7) | 512 | 262144 | REFUSED AT BOOT | — | — | — | — | v42 guard: "FATAL: explicit DFLASH2_EMIT_K=4 corrupts numerics under graphs (v42 conviction)." |
| v122_prod_k8v4 (Cell F) | v1.2.2 BAKED in-image, tq4nc + k8v4 ovr, **WARMUP=1** | 512 | 262144 | —/1734/1317 | 39.0/13.2/**5.35** | exact ×3 | coherent, USAGE, **TTFT 0.26 s** (F4 works) | 0.596 | 0 |
| v122_crashcell_tp1 (Cell G) | v1.2.2 BAKED, TP1=1 opt-in (F1 escape), WARMUP=0 | 512 | 262144 | TP1-degenerate | 17.1/6.07/6.66 | exact ×3 | coherent, usage present | degenerate (expected) | 0; probe 2048/1024 + resync fired IN-IMAGE, engine SURVIVED |
| v50_mtp1_tq4nc (Cell H) | tq4nc + MTP **k1** (graphs ON, match draft) | 512 | 262144 | — | **44.31/28.11/13.04** | exact ×3 | coherent, usage present | healthy | 0; probes ' Paris.' ×3 CORRECT |
| v50_mtp4_eager (Cell I) | tq4nc + MTP k4, **EAGER** (XPUGRAPH=0), probes only | 512 | 262144 | — | — | — | — | — | **temp-0 GARBAGE ×4, ALL DIFFERENT** (Chinese file-dialog text / Kerala / Kerala-drift / French bibliography); boot healthy |
| v50_f2_warmup0 (Cell F2) | v1.2.2 BAKED in-image, tq4nc + k8v4 ovr, **WARMUP=0** | 512 | 262144 | bench3 516.5/298.5/152.1 | 48.89/17.33/9.44 | exact ×3 | coherent, USAGE 512 | — | 0; DFLASH_STALL 12 (vs F's 498) — **bake exonerated**; rp = **CRASH event #4 (FATAL zombie, same signature)** |
| v50_mtp1_loop (mtp1b) | tq4nc + MTP k1, battery replication + FULL extras | 512 | 262144 | bench3 232.7/232.3/143.9, conc8 271.2 | **45.13/28.39/13.1** (replicates H) | exact ×3; coh×3 ' Paris' **-0.452 ≡ dflash ref** | coherent USAGE 512; loopscan 27,610 chars NO-LOOP **w/ USAGE** | 0.62-0.76 live | 0 battery errs; **dt_loop8 3/4 think-trapped (dflash 0/4); rp round-1 long0 → #11-CLASS ENGINE WEDGE** |
| v50_f8a (guard replay) | v1.2.1 + overlays 1-5 + **EXTRA_VOL6 F8** (match draft), dflash7 | 512 | 262144 | — | 45.84/16.82/9.6 | exact ×3; coh×3 -0.452 (guard inert) | coherent USAGE 512; loopscan 26,372 NO-LOOP w/ USAGE | healthy | **ZOMBIE RECURRED 2× (rp rounds 1+2) → GUARD FIRED BOTH (72>64, FINISHED_ABORTED) → ENGINE SURVIVED, health 200, 0 SYCL asserts, fresh requests served live**; cost = each zombie's client stream hangs (serving-layer half unfixed) |
| v50_v123 (V123 CERT) | **v1.2.3 BAKED in-image** (F8 async_sched a096c5a8), tq4nc + k8v4 ovr, WARMUP=0 | 512 | 262144 | bench3 516.9/298.7/202.4, conc8 283.5 | **55.45/19.11/9.29** | exact ×3; coh×3 ' Paris' **-0.452 bit-identical ≡ ref** | coherent USAGE 512; loopscan 29,353 NO-LOOP **w/ USAGE** | 0.602 | **ZOMBIE RECURRED 5/5 rp rounds → GUARD FIRED 5× (all 72>64, rounds 1+2 telemetry byte-identical) → ENGINE SURVIVED ALL — health 200, 0 SYCL asserts, fresh completion answered post-extras** |

Notes:
- **V123 CERTIFIED — v1.2.3 is the final image; prod restored on it
  (2026-09-06 03:15–04:05)**. Baked image (all 6 overlays in-image, no
  EXTRA_VOL), prod combo (dflash7 + tq4nc + k8v4 draft override), WARMUP=0
  for the cell. Every gate green:
  - **Battery decode 55.45/19.11/9.29 @2k/16k/65k vs certified r6c
    44.7/15.9/9.13 = +24%/+20%/+2%**; det exact ×3; cargame USAGE 512;
    battery error scan 0. Boot log: both probe lines IN-IMAGE (F8 armed
    placeholder limit=64 spec width=7; hasher-resync probe 2048/2048
    silent no-op — F2a inert when hasher/pool agree), bake markers
    `.v50_baked` + `.v503_baked` present.
  - bench3 516.9/298.7/202.4 (65k = the r6c-class 202, NOT Cell F's
    warmup-degraded 129), conc8 283.5 agg, acceptance 0.602 (≡ r6c).
  - **coh×3 ' Paris' top-1 logprob -0.452 bit-identical across rounds AND
    vs the dflash/nospec reference** — the baked image is numerically
    indistinguishable from certified.
  - loopscan 29,353 chars NO-LOOP with usage present.
  - **Crash-sequence survival, 5-for-5**: the crash-3 zombie class
    recurred deterministically in ALL 5 rp rounds (each long0 stream
    stalls ~500–1,500 client-visible toks while the engine accepts
    ~4k; rounds 1+2 telemetry byte-identical: placeholder-overflow
    72 > 64, num_output_tokens=4034, num_computed_tokens=4171). Guard
    fired 5× (03:32:49, 03:41:26, +3), every request reaped
    FINISHED_ABORTED with Running→0 <5 s, **engine survived all five**
    — final probes: health 200, fresh completion answered
    (' Paris.\nThe capital of Germany is'), fired=5, SYCL
    vectorized-gather asserts **0** (unguarded engines died with 1,024).
    Residual unchanged: each zombie's client SSE stream hangs until the
    client times out (serving-layer half = upstream ticket Layer 1).
  - Full log: v123_final_035601.log. **Prod booted on v1.2.3 at 03:58**
    (same combo, WARMUP=1): startup complete 04:01, health 200, F8
    armed + resync no-op in boot log, temp-0 probe ' Paris' correct.

- **Cell F (v1.2.2 in-image) battery CLEAN but SLOW — F2 discriminator
  queued**: functional = det exact ×3, cargame coherent w/ USAGE, errs 0,
  conc8 291.0 (> 283.9), bench3 2k/16k 498.8/293.3 (≈ R6c 515.8/298.2),
  acc 0.596, and F4 warmup works (cargame TTFT 0.26 s — no 50 s JIT
  stall). BUT battery decode 39.0/13.2/5.35 vs R6c 44.7/15.9/9.13 and
  bench3 65k **129.4 vs 202.0 (-36%)**; engine log shows **498
  DFLASH_STALL propose() ~0.25 s warnings pervasive from the warmup
  window onward** (peer-late/IPC-degradation signature), plus JIT
  warnings during battery for 8 spec/tq kernels the warmup phases did
  NOT cover (_tq_mq_decode_stage1, _tq_mq_fwd_stage2, expand_kernel,
  eagle_prepare_inputs_padded_kernel, _topk_topp_kernel,
  rejection_greedy_sample_kernel, batch_memcpy_kernel,
  _tq_full_dequant_kv) — F4 phases 4-5 = partial coverage only.
  Suspects: degraded boot (oneCCL/IPC class), warmup side-effect, or
  bake diff; F2 cell (v1.2.2 + WARMUP=0 + same prod cfg, gated at chain
  end) splits image vs boot/warmup.
- **F2 RESOLVED the F-vs-R6c slowdown — WARMUP SIDECAR, bake exonerated
  (but CRASH event #4 fired)**: v1.2.2 + WARMUP=0 battery decode
  48.89/17.33/9.44 (≥ R6c 44.7/15.9/9.13 at every depth), det exact ×3,
  cargame USAGE 512, errs 0, chat TTFT 51.6 s normal, DFLASH_STALL count
  **12 vs F's 498** — Cell F's 5.35@65k + stall storm = the warmup
  sidecar's IPC-degradation, NOT the bake. bench3 516.5/298.5/152.1 (65k
  ≈ certified 149.1; R6c's 202.0 was the outlier), conc8 274.3 ≥ 198.8.
  THEN rp: tiny OK (finish=length, usage present) → long0 796 toks
  **finish=None usage=None (zombie, sub-fatal ~30.8 s stall leg)** →
  long1 HTTP 500 → health 000 = **CRASH event #4, FATAL, on the baked
  image with WARMUP=0 — exact crash-3 signature, now 2-of-2 on the
  loopscan→rp tail (events #3, #4) and independent of image/overlays/
  warmup**. Server log lost (salvage raced fast container exit,
  salvage_013534.log = 0 bytes; evidence = patched-client zombie
  markers). Response: F8 runaway-request guard implemented (dual-leg
  force-finish in async_scheduler, EXTRA_VOL6) — F8a replay cell queued
  behind mtp1_loop to validate guard-fires-engine-survives on this exact
  sequence.
- **F8a PASS — guard validated live, twice in one cell (2026-09-06)**:
  same crash sequence on the guarded core (v1.2.1 + overlays + EXTRA_VOL6,
  dflash7): battery 45.84/16.82/9.6 (≥ certified), det exact ×3, coh×3
  ' Paris' -0.452 (guard numerically inert on healthy traffic), loopscan
  26,372 chars NO-LOOP w/ usage. THEN the zombie class recurred
  **deterministically in BOTH rp rounds** (long0 stream stalls ~500
  client-visible toks at ~1,990 engine-accepted — outputs stop
  resolving): guard fired at **placeholder-overflow 72 > 64** after ~9
  scheduled steps (02:50:11 and 03:01:34), force-finished
  FINISHED_ABORTED, request reaped (Running: 0 within 5 s), **engine
  survived both: health 200, 0 SYCL vectorized-gather asserts (vs 1,024
  fatal in event #3), loggers/metrics flowing, and a fresh completion
  answered correctly WHILE round-1's zombie stream was hung**. Residual
  (expected, by design): the zombie request's own client-facing SSE
  stream never terminates (the finish marker cannot traverse the same
  output leg that stalled) — the timeout-less rp client hangs until
  killed; that is the serving-layer half (upstream ticket Layer 1) and
  an operational client-timeout requirement. Guard fires per-recurrence
  (2 independent firings observed), engine survives each. Full engine
  log: f8a_final_031420.log.
- **CRASH 3-class event #2 — Cell F rp spot FAILED (v1.2.2 certified
  lane)**: after battery+bench3, rp long0 = ZOMBIE stream-end (1650/8192
  toks, usage=None, last 150 toks crawled at 0.74 tok/s vs 15.1 prior)
  then long1 = HTTP 500 → client stopped. Crash-1 SHAPE on the certified
  lane: F2a killed the fatal assert, but the zombie usage-less end (the
  H2 vehicle) SURVIVES, now manifesting as a request-scoped 500. Engine
  log lost again (watchdog only starts inside the redo gate — fixed for
  F2 v2). Fits the cumulative-load suspect: every fresh-container rp run
  (R4/R5/R6b2) was clean w/ usage chunks; both failures (R6c extras,
  F-after-bench3) came after long cumulative work on one container.
  Countermeasures: rp_longdecode.py patched (prints HTTP 500 BODIES +
  finish_reason zombie marker); F2 v2 + redo run rp with watchdog armed.
- **EMIT_K lever CLOSED (Cells D/E) — default k=7 is the only safe emission**:
  EMIT_K=4: serve script refused at boot by the pre-existing v42 conviction
  guard (verbatim: "FATAL: explicit DFLASH2_EMIT_K=4 corrupts numerics under
  graphs (v42 conviction)."). EMIT_K=5: temp-0 numerics CORRECT (' Paris.'
  ×3 exact, conc distinct=1) and decode 43.55/22.08/11.25 @2k/16k/65k, BUT
  cargame emitted 1 token then stalled 18.2 s (tok_s=0.1, empty head/tail)
  and in-container error scan = 3 — a live engine defect under sustained
  load. Emission truncation is unsafe on this stack at any k<7 tested;
  improvement lever 2 closed, DFLASH2_EMIT_K stays default 7.
- **MTP k4 + tq4nc + XPU graphs = CONVICTED temp-0 CORRUPT (v50 Cell C)**:
  "The capital of France is" (temp 0) → `' is a 2005 film directed by
  David O. Russell...'` ×3 + drift variant — garbage first tokens, mostly
  byte-reproducible, single-request AND concurrent (conc_n distinct=2/2
  with both outputs wrong). Sampling lanes (temp 0.3 cargame) look
  coherent ⟹ corruption is in greedy-path logits under the 5-row verify
  (k4 boundary — the v29c-era k4-corruption class: piece-capture × MTP
  k4, k≤3 bit-stable, eager = reference). MTP throughput 57.2/25.6/11.9
  (beats dflash 44.7/15.9/9.1) is REAL but the lane is numerically
  UNTRUSTWORTHY on this stack. Overlays innocent (no MTP path touched).
  **dt_loop8 mtp arm: 4/4 think-traps still trapped @8192 + 1 tail-loop**
  (vs dflash 0/4 + 0) — trapping plausibly corruption-driven.
  Discriminators queued: mtp1 battery+probes (k≤3 claim), mtp4 eager
  probes (reference). DFlash2 k7 = only correct spec lane on TQ.
  **DISCRIMINATORS DONE (Cells H/I) — conviction EXTENDED + F7 landed**:
  H (mtp1, graphs ON): battery ALL-CLEAN, probes ' Paris.' ×3 CORRECT —
  the v29c "k≤3 stable" claim holds for k=1 on this stack; AND decode
  44.31/28.11/13.04 = +77%/+43% over dflash7 at 16k/65k (2-row verify,
  no cross-model drafter host cost; itl 42 ms vs ~90). I (mtp4 EAGER,
  graphs OFF): temp-0 GARBAGE ×4 with ALL-DIFFERENT outputs — the v29c
  "eager = reference" claim is OVERTURNED on this stack; mtp4 is corrupt
  in BOTH modes ⟹ defect is in the MTP k≥4 multi-row path itself, not
  graph capture. F7 serve-script guard landed (mtp4/mtp7 → FATAL refuse,
  validated exit 1; mtp1 + dflash lanes pass). mtp1 = correct AND
  fastest-at-depth spec lane measured; loop-scan queued before any
  promotion consideration (mtp4 failed dt_loop8 4/4; mtp1 untested).
- **mtp1_loop CLOSED the promotion question — mtp1 = NO-GO on this stack
  (robustness), despite replicating every quality/speed win**: battery
  45.13/28.39/13.1 ≡ Cell H (44.31/28.11/13.04), temp-0 exact ×3, coh×3
  top-1 logprob **-0.452 bit-identical to the dflash/nospec reference**
  (cross-lane numerics agreement), cargame USAGE 512, loopscan 27,610
  chars NO-LOOP **with usage present**, bench3 232.7/232.3/143.9,
  conc8 271.2 (parity w/ dflash 274.3). Two disqualifiers: (1)
  **dt_loop8 3/4 think-trapped @8192** (dflash 0/4; mtp4 was 4/4 —
  entrapment tracks the MTP drafter family, not corruption: no
  periodicity detected, reason runs are coherent 24-34k-char
  deliberations, P6 escapes naturally; still burns full-length
  requests operationally). (2) **rp round-1 long0 → #11-CLASS ENGINE
  WEDGE @~1.1k/8192 toks (02:04:16)**: SpecDecoding metrics FROZEN
  (engine loop stopped stepping — NOT the crash-3 zombie-alone-decode
  pattern, where metrics continue until the assert), Worker_TP0 161% /
  Worker_TP1 100% / EngineCore 87% CPU all spinning, both GPUs ~220 W
  (HBM-spin power profile), **0 SYCL asserts**, health 200 the whole
  time (APIServer alive on a dead engine = silent service loss),
  client stream FROZEN not ended (no usage-less end — the crash-3
  trigger is absent; this is an engine-side stall mid-request under
  plain long decode). Salvage: wedge_mtp1_0228.log. Chain released by
  force (rp client had no timeout; killed + teardown 02:29:54).
  **F8 is inapplicable to this class** — the guard lives in the
  scheduler's post-schedule hook; a wedged loop never schedules.
  Verdict: wedge family = the v29 #11 oneCCL collective-spin class,
  now observed on an MTP lane; dflash7 (v50 matrix: 17+ cells, never
  wedged) stays the only promotion-grade spec lane. mtp1 remains
  F7-allowed as experimental (no corruption; availability risk only).
- **Block-size axis CLOSED (tq4nc dflash7)**: bs2048 51.2/16.6/9.33,
  bs512 46.4/17.7/9.1 (match) / 44.7/15.9/13 (ovr k8v4), bs64
  47.8/15.1/9.16 — all exact ×3, coherent, 0 errors. bs2048 +10% at 2k;
  no regressions anywhere.
- **Looping prevention, dflash arm (R6c)**: loopscan cargame-8192
  server-default sampling = 26,702 chars, **NO-LOOP** (word+char
  periodicity scan), coherent road-render tail; dt_loop8 think-trap 4-pack
  = **0/4 trapped @8192, 0 tail-loops**. CLEAN.
- **CRASH 3 ROOT-CAUSED (redo capture, salvage_010017.log)**: same
  sequence replay (battery ≡ R6c 46.32/16.5/9.01 + coh ×3 clean + loopscan
  14,182 chars NO-LOOP) → engine died 01:00:05 with NO client traffic
  pending: 1,024 SYCL asserts `vectorized gather kernel index out of
  bounds` (torch-xpu-ops IndexKernelUtils.h:63) — first anomaly in the
  entire log (only health GETs precede it) → Worker-0 death → EngineCore
  `cancelled` → 500s → exit. Death dump: ONE zombie request (57-tok
  cargame-class prompt = the loopscan req; battery cargame exonerated —
  it ended normally WITH usage, 512 toks) still RUNNING at num_output_
  tokens=22,688 = 8×2,836 (dflash slot count; 2,836 real toks ≈ 13 tok/s
  × 225 s alone-decode — matches lane speed; 22,688 real is physically
  impossible), scheduled via cached path (`scheduled_cached_reqs`,
  `new_block_ids=[None]`, spec tokens [-1]×7), `finished_req_ids=[]`.
  **Verdict: runaway (zombie) request never reaped → decoded ~4 min alone
  past max_tokens → gather indexed past real token-history length →
  fatal SYCL assert.** Class: crash-1 R2/R3 zombies (TP1 lane), Cell F rp
  long0 (sub-fatal, 500). Overlays innocent; NOT oneCCL/cumulative-load.
  3 events / ~15 cells, always post-battery-scale volume. Fix = F8
  runaway guard (scheduler force-finish), design in PLAN.md.
- **R6 series closure**: R6b2 (crash cell, TP1=1) + R6c (certified prod
  cell) both ALL-CLEAN on the FINAL core (md5 3581a600: F2a resync + F6a
  guard + probe line). R6c ≡ earlier `v50_tq4nc_bs512_ovr_k8v4` within
  noise (decode 44.7/15.9/9.13 vs 43.2/16.4/9.2) ⟹ F6a + probe are
  behaviorally inert on TQ lanes. dt_bench3 on R6c (final core, warm
  engine): ctx2k/16k/65k decode **515.8/298.2/202.0** vs certified
  482.5/280.6/149.1 (meets/exceeds at all depths; 65k +35% — v41 window
  fix paying off at depth), conc8 **283.9** agg vs 198.8, acceptance
  **0.602** vs 0.596. No degradation.
- **B5d convicts spec-on-fp8 as GENERIC (drafter-independent)**: MTP k4
  (5-row verify, single-engine head) crawls at 0.2–0.5 tok/s exactly like
  dflash k7 (8-row); acceptance machinery itself healthy (mean acc length
  2.5, per-position 1.0/0.5). Mechanism = multi-row verify forward on fp8 KV
  ≈5 s/step vs 30 ms single-row on the same pool.
- **Remedy = F6 (v50)**: boot-time refusal of fp8-class KV
  (fp8_e4m3/fp8_e5m2/fp8) + ANY speculative config, enforced at two layers:
  serve script (pre-docker fatal) and EngineCore init (RuntimeError). The
  certified fp8 lane is nospec (B5c, full clean cell).
- **B5c certifies the pure fp8 target path HEALTHY**: decode
  34.39/30.73/22.41 tok/s @2k/16k/65k — matches the historical
  `dt-bench-fp8_e4m3.out` (34.42/30.75/22.4) to 3 digits ⟹ that historical
  record was the nospec path; **fp8_e4m3 + spec drafting was likely never
  healthy**, not a v1.2.1 regression. Pathology is confined to spec-shaped
  kernels on fp8 KV: 8-row verify (tforward d≈4976 ms) + draft propose
  (d≈4027 ms) vs 30 ms single-token steps on the same fp8 pool.
- **fp8_e4m3 + dflash k7 lane is PRE-EXISTING BROKEN on v1.2.1** (B5e stock
  isolation: worker wedged in `sample_tokens` RPC at token 9 of the FIRST
  request, 8-token dflash verify step → shm_broadcast TimeoutError →
  EngineDeadError → container exit). v50 overlays innocent (stock is *worse*:
  hard wedge vs 0.3–0.9 tok/s crawl).
- B5b SPECTIMING (overlays, 20-step medians): `step_wall_ms=18766.7 |
  tforward d=4975.8 | propose d=4026.7 | precompute d=1720.4` vs TQ lanes
  tforward ~30–60 ms class — fp8 KV read path (target verify + draft propose)
  catastrophically slow at depth. Historical v39a-era fp8+dflash decode was
  34.4/30.8/22.4 tok/s → that record is now known to be the NOSPEC path
  (B5c 3-digit match); the v49c "match policy = prime suspect" hypothesis
  was DISPROVEN by B5d (MTP has no v49c draft pool, crawls identically) —
  spec-on-fp8 was likely never healthy.
- k8v4 draft-KV override ≈ match in throughput/acceptance on tq4nc
  (v49c claim re-confirmed on v50).
- Decode speed ordering (65k): tq3nc 10.0 ≈ k3v4 10.8 > tq4nc 9.1 ≈
  ovr 9.2 > k8v4 6.9 — tracks KV width as expected; not regressions.
- All cells: zero Traceback/Assertion/clamp events; every cargame ends
  WITH usage chunk; temp-0 identical outputs across 3 rounds × 2 concurrent.
- Chat TTFT on first request after boot is ~50 s in these cells because
  DFLASH2_WARMUP=0 (warmup sidecar disabled for benchmark determinism);
  the baked image runs dt_warmup (now extended, phases 4-5) which absorbs
  first-use JIT.
