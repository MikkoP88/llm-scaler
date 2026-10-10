# P83 — LAUNCH-HEADROOM GUARD + VIA-POOL (P82 wedge fix) — CLOSED 2026-10-05
# BAKE + 3-IMAGE BENCHMARK appended 2026-10-06 (see bottom).

**Verdict: the P82 deep-storm wedge class is fixed at the root. N=3/3
conviction (ok=108/108 ×3 back-to-back on one engine, no reboot), walls
1235/1231/1214s vs P82's 2207/4057s — 44-70% FASTER than the unguarded
lane's best/worst surviving legs. Zero OOM, zero ccs/guc resets, zero
wedges, f-floor ≥2.2 GB through every storm (P82 wedge floor: 38-65 MB).**

## The two-part fix

### Part 1 — P83 guard (patch_p83_headroom.py v2, split knobs)
- `xpu_worker.py` init_device tail: `torch.xpu.set_per_process_memory_fraction`
  caps per-process reserved at total − `VLLM_XPU_HEADROOM_MB` (leg posture
  1024 → cap 30.89 GiB of 31.89). Cap violations raise CATCHABLE OOMs
  (probe-certified E1-E4); freed cached blocks recycle internally (E3).
- `gpu_worker.py` sizing tail: KV pool shaved by `VLLM_XPU_POOL_SHAVE_MB`
  (fallback = HEADROOM, v1 behaviour). Leg posture 2048 → pool 6.28 GiB.
- md5s: xpu_worker `800b8d67…` markers=2; gpu_worker v2 `663420c3…` markers=2.

### Part 2 — P83b VIA-POOL (patch_p83b_via_pool.py) — the root kill
**SHAPE-RATCHET LAW (measured):** the via-allgather TP all-reduce
(`VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1`, the lane's standing wedge-mitigation
posture) cached its buffers per EXACT tensor shape — `_xpu_allgather_allreduce`
gathered list + v29 stable-output dict — never evicted. Ragged chunked-prefill
shapes differ at every chunk boundary → a deep storm mints dozens of shapes ×
(world_size+1) tensors ≈ +6 GB LIVE growth per storm (argsnap a=23.2→29.9 GB
monotone over 58k GDN calls in leg 2). This was THE dominant driver of P82's
"reserved 19→35.5 GB within one storm" climb, and the death-time allocation
was always the next NEW shape's 142.00 MiB buffer (both legs died on exactly
142.00 MiB, first at the GDN-class chunk, then inside oneCCL allgather).

**Fix:** ONE buffer set per (device, dtype, hidden), sized to row-max,
prefix-row views (contiguous, storage offset 0) for every call. data_ptr
guards: pooled-output callers alias directly (0 copies), legacy callers
stage in/out. `VLLM_XPU_VIA_POOL_MAX_ROWS` (default 8192 = MNBT; 0 = inert).
Footprint measured: 384 MB (hidden 8192) + 240 MB (hidden 5120) = 624 MB
BOUNDED, grown at warmup, zero hot-path allocations after. Pool-grow INFO
lines are the growth telemetry (4 lines total = closed).

- md5: xpu_communicator `3acb75ed…` markers=5.

## Leg matrix (storm vehicle: wsc_pressure_v134_leg7_depth.py, 108 reqs)

| Leg | Posture | Result |
|-----|---------|--------|
| P82 r1/r2 | no guard | WEDGE @ T+2207s / survive 4057s (f-floor 38-65 MB) |
| P83 leg 1 | cap 1024 / shave 1024 | OOM @ 314s ok=19 — cap converts wedge→loud OOM, zero resets |
| P83 leg 2 | cap 1024 / shave 2048 | OOM @ 494s ok≈56 — a=29.9 GB still ratcheting, 142 MiB in oneCCL allgather |
| **P83b leg 3** | **+ via-pool** | **ok=108/108 wall=1235s, r flat 26,636 MB** |
| **P83b leg 4** | same engine | **ok=108/108 wall=1231s, r flat 26,640 MB** |
| **P83b leg 5** | same engine | **ok=108/108 wall=1214s, r flat 26,642 MB** |

Legs 1-2 were the calibration ladder that named the ratchet (each death
banked: P83_wedge_170637, P83_wedge_174108 — both zero-resets, clean
catchable OOM exits = the guard's designed conversion even before the root
fix landed).

## Calibration laws (measured, card = 32,656 MB)
- Deep-storm live set ≈ weights 14.5 + pool + activations; at death legs
  saw a=29.6-29.9 GB WITH pool 6.28-7.28 → activations+misc ≈ 8-9.6 GB.
- Non-torch driver consumers ≈ 0.95-1.0 GB/card → driver-free floor under a
  reserved cap = HEADROOM − non-torch (1024 → ~70-120 MB floor if pinned).
- Healthy guarded posture: reserved plateaus 26.6 GB (a≈24.4 GB), giving
  f-floor 2.2+ GB — 4 GiB of transient margin. The cap is now a BACKSTOP
  that never binds in steady state.
- 892 MB "reserved but unallocated" at leg-2 death = freed-but-not-retired
  (async kernels) + shape-fragmented cache; pooling removes the pressure.

## Kit / artifacts
- Kit: /root/build/p77/ — patch_p83_headroom.py (v2, 6594186d host copy),
  patch_p83b_via_pool.py (a52210cd), boot_p83.sh / boot_p83_2.sh /
  boot_p83b.sh (0b9355c5), p83_monitor.sh.
- Banks: /root/build/lce1/P83_wedge_170637/ (leg-1 OOM freeze), 
  P83_wedge_174108/ (leg-2 OOM freeze + argsnap_full_537.txt growth curve).
- Leg outputs: /root/build/v130_stage/p83_leg7_r{2,3,4,5}.out; jsonl grew
  1596→1979 across legs 2-5.
- Local: .tmp-p76/p83/ (patchers, boots, probes, UPSTREAM_MTP_RESEARCH.md,
  this file).

## Posture / next
- Lane at close: lsv-test serving the certified P80+P82+P83+P83b spec×4
  engine (health 200 after 3 storms); watchdog PAUSED (do not unpause
  without deciding the bake — unpause destroys the patched container).
- **Bake v1.2.30 recommendation: fold P83 v2 + P83b into the image
  (python-only, .so unchanged), knobs defaulted ON (HEADROOM 1024 /
  POOL_SHAVE 2048 / VIA_POOL 8192) — posture-general: protects nospec
  production identically (the ratchet is TP-path, not spec-path). Gated on
  user per standing law.**
- MTP program (research doc phases A-D: dynamic-K, K=0/1 paths, GPU-resident
  verify) remains queued — unaffected by this round; the pool fix already
  removed the TP-collective hazard from that path.

---

# BAKE v1.2.30 + 3-IMAGE MTP BENCHMARK — 2026-10-06 (user-ordered)

## Bake: llm-scaler-exp:v1.2.30 (image id 9901463141cd)
- Base: fresh v1.2.29 container (cd365d112d4c) + patch_p83_headroom.py v2 +
  patch_p83b_via_pool.py ONLY. Python-only; instrument-free (no P80/P82).
- Certified md5s inside the committed image: xpu_worker 800b8d67 / gpu_worker
  663420c3 / xpu_communicator 3acb75ed (byte-identical to leg posture).
- .so UNCHANGED vs pristine v1.2.29: vllm_int4_for_multi_arc.so md5
  82ce75c7 sha256-prefix 2331ae8a (the long-cited "1d9dcf4e" was a stale
  reference — corrected this round).
- serve_user.sh: baked knob defaults with operator-override form —
  VLLM_XPU_HEADROOM_MB:-1024, VLLM_XPU_POOL_SHAVE_MB:-2048,
  VLLM_XPU_VIA_POOL_MAX_ROWS:-8192. Posture-general (protects nospec
  production identically — the ratchet is on the TP path). nospec default
  posture unchanged; spec stays opt-in.
- Stale pyc for the three patched modules removed pre-commit.

## Benchmark matrix (all legs: spec×4 MTP, TP2, gmu 0.8, IDENTICAL serve
line + env set; image = only variable; storm vehicle 108-req depth shape;
fresh container per leg; host reboot before round + after each death leg)

| Image | Result | Wall | Class |
|-------|--------|------|-------|
| **v1.2.30** (99014631) | **ok=108/108, 0 errors, 0 resets, health 200** | **1244s** | CLEAN — parity with certified band 1214-1235s |
| v1.2.29 (cd365d11) | ok=82/108, health 000 (silent wedge) | 1910s | P82 WEDGE — ratchet live: d0 free 123 MB / d1 free 28-132 MB at ceiling; py-spy: TP0 hung in gdn core op, TP1 hung in all_gather INSIDE _xpu_allgather_allreduce (the exact per-shape site P83b pooled); deceleration 42→21→11 req/7min as the ratchet strangled |
| v1.2.26 (92cf94b2) | ok=98/108, 55 ERROR, EngineCore fatal, health 000 | 2102s | DEEP-FORWARD HANG — worker dequeue timeout; death batch num_computed [3676, 30206, 35842, 18432] (30k+ deep = death shape); 500-cascade, no guc reset in dmesg |

Banks: lce1/P83_wedge_V129BENCH/ (py-spy both workers, serve log, xpusmi,
dmesg), lce1/P83_crash_V126BENCH/ (serve log w/ 55 tracebacks + scheduler
death-batch dump, xpusmi, dmesg). Leg outputs p83_bench/leg{1,2,3}_*.out;
host driver bench_leg_host.sh + in-container bench_boot.sh in
/root/build/p83_bench/ (identical serve line across legs).

Verdict: the two-part fix is worth +26 reqs and engine survival per storm
vs v1.2.29, and a loud clean 108 vs v1.2.26's crash. v1.2.30 wall parity
holds (1244s vs 1214-1235s certified). Restore chain moved to v1.2.30.

New traps this round: (1) pkill self-match via ls PATH containing the
plain name in the SAME plink command line (ls .../p83_monitor.log; pkill -f
'[p]83_monitor' killed the shell before docker rm ran — keep pkill in its
own plink); (2) `curl && wc && setsid ... &` backgrounds the WHOLE chain →
holds plink stdout open → hang (launch storm in its own plink; smoke
separately).

---

# TOK/S BENCHMARK — 2026-10-06 12:15 UTC (user-ordered, EXECUTED)

Same vehicle/protocol as the survival matrix (identical spec×4 MTP serve line
via p83_bench/bench_boot.sh, image = only variable, fresh container per leg,
reboot before round + after the death leg; storm = 108 reqs, deterministic
prompt shape Σ 2,114,594 prompt tokens). tok/s extracted from the driver jsonl
per-request usage records (toks_sum.py, host md5 b73e772e). NOTE: the shape is
~99.7% prefill by tokens (finish=stop at mean 55–71 completion tokens/req), so
TOTAL tok/s (prefill+decode) is the comparable headline; out tok/s is decode
only and tiny by construction.

| Image | ok | Wall | Total tok/s | Out tok/s | Fate |
|-------|----|------|-------------|-----------|------|
| **v1.2.30** | **108/108** | **1242s** | **1707.7** | 5.2 | CLEAN, parity 1244s prior |
| v1.2.29 | 108/108 | 2446s | 866.9 | 2.4 | SURVIVED but ratchet-strangled |
| v1.2.26 | 62/108 | 1392s† | 885.6† | 3.2† | CRASH @58% of workload (†until death) |

- **v1.2.30 = 1.97× v1.2.29 throughput on the identical workload.**
- NEW LAW — RATCHET-IS-THROUGHPUT: v1.2.29 did NOT wedge this run (survival
  is intermittent — last round died at ok=82) but spent the entire back half
  pinned at the memory ceiling: d1 free oscillated 22–454 MB from ~T+15m,
  98.6–99.9% mem util (xpu-smi live), wall 2× inflated. Same shape-ratchet
  mechanism; whether it kills or strangles is timing luck. The P83b via-pool
  removes BOTH outcomes (v1.2.30 flat + clean).
- v1.2.26 crashed EARLIER than last round (ok=62 vs 98 @2102s) — intermittent
  tail again. Death batch num_computed [3314, 30019, 29927, 3229, 3234] (two
  ~30k-deep = death shape), EngineCore worker dequeue timeout, 61 tracebacks,
  500-cascade, no GPU reset in dmesg. Banked lce1/P83_crash_V126TOKS/ (serve
  log w/ scheduler death dump, dmesg, xpusmi d0/d1, jsonl, leg outs).
- Artifacts: p83_bench/leg{1,2,3}_toks.jsonl + leg{1,2,3}_toks.out +
  _host.out; v130_stage baks .toks_leg1/.toks_leg2 (+live leg3 jsonl);
  pre-round jsonl archived .pre_toks_round.bak.
- Post-round: host reboot after death leg → BOOT_V130_PROD v1.2.30 nospec
  restored 12:06:52 (health 200) → watchdog unpaused → first cycle 12:14:00
  `alive armed code=200 cycles=10` + `V1227_DRIFT_NONE
  posture=certified-swift-fp8`. Lane certified clean.

## Post-restore close-out (2026-10-06 22:13)
- Watchdog first post-unpause cycle: `alive armed code=200 cycles=10` —
  adopted the running v1.2.30 lane, no restart. PAUSED flag stays removed.
- Drift check rebased: watchdog_v2_drift_check.sh (single copy
  /root/build/v127_stage/) md5 aa314093 — EXPECT_IMAGE → v1.2.30, spec
  gate INVERTED (spec on the default lane = DRIFT; V1227_EXPECT_SPEC=1
  declares a sanctioned spec leg window; verdict block uses SPEC_OK). It
  had been crying 2 stale DRIFT lines every 10 min since v1.2.29 (frozen
  at v1.2.27 + spec-missing). Live verdict now:
  V1227_DRIFT_NONE posture=certified-swift-fp8. LAW: rebase the drift
  check after EVERY posture ship.
