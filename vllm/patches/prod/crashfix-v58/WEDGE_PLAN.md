# WEDGE PLAN v58 — bulletproof diagnosis & fix plan (async-event stall class)

Mandate (user, 2026-09-11): MTP must run crash-free on fp8_e5m2 AND
fp8_e4m3, no degradation. Next occurrence: deep research + comprehensive
capture FIRST, then a bulletproof fix plan. **Patches outside the vLLM
codebase (driver / oneCCL / level-zero / kernel / env) are authorized.**

## 1. Established facts (7 occurrences: 4 morning 10:09-11:25, control-A
   16:00, treatment-B 16:24, control-C 17:11)

- F1 Signature (all 7): TP-symmetric `num_accepted_tokens_event
  (deferred postprocess)` + `async_output_copy event` never signal; v55
  fence fires at 60-120 s; engine dies via RPC timeout ~3 min.
- F2 At wedge (control-C capture, 17:12:14): **GPU fully IDLE** — all
  engine classes 0.00%, EU 0% active / 100% idle, mem bandwidth ~0,
  driver responsive (xpu-smi via level-zero succeeds). NOT a kernel
  hang, NOT a spin. Worker processes alive.
- F3 fr (live workers, treatment-B): host completed the step through
  clean `propose end` (TP-symmetric); GPU never drained the queue =>
  completion lost, submission lost, or dependency deadlock.
- F4 Stochastic: wedged 5-42 min into pressure; morning 4/4 on first
  warmup, afternoon rarer. DEBUG-logging boot never wedged (host-timing
  sensitive). No guc declaration in most events (stall #4 + control-A
  aftermath show ccs/bcs resets at onset/teardown).
- F5 Context: mtp4 + async-scheduling + TP2 + mamba hybrid + fp8 KV
  (both dtypes); ~2048-block boundary geometry; onset steps show
  near-request-finish tails (Accepted: 2, mean acceptance 1.2).
- F6 Disproven: e5m2-lane, node-state, serve-flag-delta, and **v57
  buffer-instance isolation (patch verified live, wedge recurred)**.

## 2. Capture battery at next occurrence (armed: boot D, 600 s fence)

`wedge_watch3.sh controlD` — metrics-gap onset detection (>150 s),
snapshots while workers are STILL ALIVE mid-wait:
  - t0 / +90 s / +240 s: per-worker main+R-thread states, wchan, kernel
    stacks (/proc/PID/stack), ctxt_switches; GPU util+EU per device;
    dmesg tail (guc reset watch); serve tail.
Post-mortem: full stall-line set, fr by live pids, EngineCore stacks,
container procs, dmesg full. (Prior captures: controlA_160142,
treatB_162554, controlC_171212.)

## 3. Decision tree (evidence -> root-cause class -> fix layer)

- D1 Workers RUNNING in ze ioctl/loop + GPU idle + no guc reset
  (expected per F2) => **lost-completion class**. Sub-discriminators:
  - D1a OOB write clears ze event-pool flags (expandable_segments packs
    VAs; wild kernel write hits pool). Test boot: allocator-layout A/B
    (`PYTORCH_XPU_ALLOC_CONF` without expandable_segments or with a
    guard region) — wedge disappearance + perf-parity check. If proven:
    fix the OOB WRITER (-> D1a') not the allocator.
  - D1a' Writer candidates: mamba batch_memcpy with wrong indices
    (v32 deferred snapshot arithmetic: `dest = aligned//block_size-1`
    can go -1 on running=0; stale-but-unskipped src rows), GDN state
    written to wrong block then read as indices/pointers. Fix: bounds
    validation of (src,dest,bias) per row before launch (host-side,
    nanosecond cost, zero hot-path degradation) + arithmetic fix.
  - D1b level-zero event/queue bug (flag never set on drained queue):
    evidence = D1 + D1a disproven. Fix layer: driver/ZE (authorized) +
    upstream ticket with captures; vLLM-side mitigation = re-record
    safety (see 4-M).
- D2 Workers blocked in ccl/IPC cross-process waits => oneCCL kernel
  collective handshake deadlock (rank-divergent kernel sequences).
  Fix: enforce rank-consistent mamba skip decisions (make deferred
  row-skip deterministic from scheduler state only) — vLLM layer.
- D3 GPU busy at t0 (would contradict F2): kernel hang => guc coredump
  wait, identify kernel, fix kernel (authorized).

## 4. Fix constraints (no degradation — hard requirements)

- Zero hot-path cost: only entry/exit or per-ROW host arithmetic
  changes; no extra syncs, no stream-order changes, no allocator
  behavior changes unless proven perf-neutral (battery A/B).
- Numerics untouched: certification battery sha-parity required.
- Validation bar: control wedges ≤ 60 min under harness; fixed boot
  survives ≥ 2× control time-to-wedge under identical pressure +
  throughput parity (ctxscan/genspeed vs v1.2.9 baselines) on BOTH
  dtypes; then bake v1.2.10-class image; then user-exact-config stand
  with telemetry + GO + live watch.

## 5. Boot D results (control, wedged 18:35:50 — 8th event, best-instrumented)

- **F7 Anatomy at t0 (workers ALIVE mid-wait, 600 s fence)**:
  - Host: both workers sleeping in `hrtimer_nanosleep` = v55 progressive
    wait polling a never-signaling event (fence lines fired 5x).
  - **GPU0: compute+copy engine util 99.99% with EU only 4% active** =
    kernels RESIDENT in wait states (spin/poll loops), NOT hot compute.
  - **GPU1: guc declared `ccs32` hang at onset+2.5 min -> engine reset +
    devcoredump** (rescued: controlD/devcoredump/card2_data.bin; context
    desc 0x021df000; no kernel name recoverable from strings).
  - fr (both ranks): TP-symmetric freeze at clean `propose end`
    (27442.1515 / 27442.1485, normal 30 ms skew), preceded by a long run
    of small kernel-mode ARs (`numel=10240`, host-side 1-2 ms each,
    via_env=False = default oneCCL path). Host enqueue complete on both
    ranks; GPU timeline never drained.
  - Last metrics at onset: acceptance collapse to 4.1 tok/s = request-
    finish tail (same as F5).
- **F8 Code-level eliminations**:
  - `CpuGpuBuffer.copy_to_gpu` = plain `non_blocking` copy on the
    CURRENT stream -> mamba metadata H2D + `batch_memcpy` + GDN copies
    are all compute-queue stream-ordered. No structural cross-engine
    cycle inside the v32/v52f glue.
  - xpu_communicator.py documents a KNOWN wedge mode for REPLAYED oneCCL
    kernels under graphs — mitigated by keeping ARs eager between XPU
    graph pieces (FULL_DECODE_ONLY + custom-op split, load-bearing).
    The wedge nonetheless occurs in an **eager** AR sequence.
- **Refined leading theory (class D2' eager kernel-mode AR protocol
  deadlock)**: with CCL_ENABLE_SYCL_KERNELS=1, small ARs run as SYCL
  flag-poll kernels on the compute stream. GPU0 resident at ~4% EU =
  AR kernel polling peer flags; GPU1 queue head EU-idle in semaphore
  wait -> instance-pairing divergence (flag/slot mismatch) between the
  two ranks' collective instances. Trigger for divergence still open
  (prior OOB write corrupting oneCCL IPC flags = D1a', or oneCCL/driver
  race). NOTE: guc reset on GPU1 is a SYMPTOM (hang declared 2.5 min
  after queue stopped retiring), not the cause.

## 6. Discriminating experiments

- **Boot E (19:50-20:30): CCL_ENABLE_SYCL_KERNELS=0 — CONVICTED
  CORRECTNESS-BREAKING, wedge-absence VOID.** Single-env flip from
  control; sustain 4/4 + loop 24/24 "clean" (fence 0, 0 resets) BUT
  generation is garbage from the start: greedy 'The capital of France
  is' -> reasoning 'We are abouts about...' + EOS @8 toks (certified
  reference behavior = ' Paris'); crash-sample cargame -> 8-215-tok
  fragments, content None; first-hour acceptance degraded. Mechanism:
  oneCCL non-kernel (copy-based) allreduce produces wrong TP sums for
  our shapes/dtypes. CONCLUSION: kernel-mode AR (SYCL_KERNELS=1) is
  correctness-mandatory on this stack; the fix must make kernel-mode
  AR RELIABLE, not bypass it. (Also explains why user config sets =1.)
- **Boot F (20:23-): allocator A/B, PYTORCH_XPU_ALLOC_CONF empty
  (default allocator, NOT expandable_segments) + --max-model-len 73728
  (NOTE: image bakes expandable_segments:True as ENV — the -e line must
  be an explicit EMPTY override, deleting it does nothing; and 73728
  rejects sustain p8-70k -> rounds abort at p8, p9-p14 skipped).**
  - Wave-1 VERDICT (45 min): CLEAN — sustain 4/4 (reduced-phase), loop
    24/24 FULL-length reqs (4096 toks ~103 s each, acceptance 2.41,
    greedy canary coherent "Answer Paris"), 0 fences, 0 resets.
    EXCEEDS every control time-to-wedge (5-42 min). Wave-2 running to
    ~90 min cumulative to kill stochastic noise.
  - If cumulative clean -> allocator/VA-packing implicated (D1a class)
    -> Boot G = v58 diag patcher (RS/RSv fr-wrap + 3 mamba row
    tripwires, validated) + expandable_segments ON to catch the writer.
  - Caveat: user config REQUIRES expandable_segments for 262144 memory;
    endgame must be the WRITER fix (allocator stays ON), not the
    allocator change.
- **Boot G (23:04-23:39, allocator ON @73728 + v58 diag): WEDGED in
  ~25 min — DECISIVE.** Single delta from clean Boot F =
  expandable_segments restored; ctx-length exonerated, **allocator
  CONVICTED as the discriminating factor**. v55 fence fired 5x on
  `async_output_copy event`; fail-fast RuntimeError -> clean worker
  shutdown (health 000; no zombie container). Evidence:
  - **V58-TRIPWIRE count = 0 through onset** — every mamba copy-row
    (deferred postprocess / sync postprocess / carry) in-bounds at
    freeze => D1a' mamba index-garbage theory DEAD as the trigger.
  - Freeze site (fr, both ranks): TP-symmetric at clean `propose end`
    (45129.2419 / 45129.2366, 53 ms skew) after a train of small
    eager ARs (numel 5120/25600, via_env=False); NO AR/RS/RSv/AG line
    open on either rank => host BETWEEN collectives (verify
    postprocess / D2H / event-record / deferred-launch territory) when
    the GPU queue stalls. reduce_scatter/reduce_scatterv NOT on the
    wedge path (wrap never logged = never called near freeze).
  - watch4 MISS: POST-age guard (<300 s) cannot see a wedged IN-FLIGHT
    request (its POST line never logs) -> onset passed uncaptured;
    fail-fast post-mortem capture instead. Fixed in watch5: trigger =
    metrics-gap AND running>0 (engine /metrics endpoint) OR fence
    line; idle wave boundaries have running=0.

## 8. Boot H (0.85) — WEDGED at ~69 min + FIRST LIVE t0 CAPTURE

watch5 fired correctly (metrics-age 159 s + running=1 at 01:48:23;
onset ~01:45:45). TTF longer than G (25) and controls (5-42) but NOT
>= 84 min bar => 0.85 DELAYS, does not fix; and **wedge hit at KV
usage 2.95%, waiting=0, 0 preemptions — NOT a KV/memory-capacity
event**. gpu-mem-util only perturbs VA layout + timing.

- **F9 Host freeze sites (py-spy, workers alive; first ever)**:
  - TP0 MainThread ACTIVE (CPU-spinning, 362k jiffies):
    `prepare_inputs_event.synchronize()` — raw torch event sync,
    gmr `synchronize_input_prep` :4015 via execute_model :4349.
    UNFENCED => level-zero driver spin-wait. Prior step's input H2D
    never signaled.
  - TP1 MainThread: fenced `_v55_wait_event(num_accepted_tokens_event,
    deferred postprocess)` gmr :4357 (past input-prep sync). Prior
    step's accepted-counts D2H never signaled.
  - TP0 WorkerAsyncOutputCopy thread: fenced `_v55_wait_event` in
    `get_output` gmr :336 (async_output_copy event) — F1 signature.
  => Both ends of the async-scheduling overlap blocked on events
  recorded on the compute stream; everything upstream in the GPU
  queue never retired.
- **F10 GPU state at t0 (BOTH devices, identical)**: compute util
  99.99 + copy util 99.99 + mem 99.9x BUT eu.active 2% / eu.idle
  94% => resident spin/poll kernels on BOTH engines, no compute.
  No guc reset this time (dmesg clean through t240).
- **F11 fr (both ranks)**: ends at clean `propose end` (40 ms TP
  skew) after trains of eager kernel-mode ARs; LAST ARs = numel 5120
  (=1 token x hidden 5120 — request-tail geometry, F5 confirmed) and
  25600 (5 tok). All begin/end matched — host enqueue COMPLETE on
  both ranks; ccl.all_reduce() host calls returned long before.
  APIServer died ~01:54 (POST 500 + shutdown, shm_broadcast 60 s
  stall line at 01:47:43); orphaned TP0 kept spinning (killed 01:56).
- **F12 Root-cause refinement (D2'' stale IPC mapping under
  expandable_segments)**: oneCCL kernel-mode AR = flag-poll kernels
  exchanging peer data via IPC-mapped USER buffers (AR inputs =
  ephemeral activation tensors from the PyTorch XPU allocator). With
  expandable_segments, freed VA ranges are unmapped/remapped on
  regrowth; oneCCL CACHES opened IPC handles for peer VAs
  (`CCL_ZE_CACHE_OPEN_IPC_HANDLES`, strings confirm cache + eviction
  note). A cached handle to a remapped peer VA reads/writes the OLD
  physical pages => peer flag never observed => poll kernel never
  retires => stream events (H2D input-prep, D2H accepted-counts,
  output copy) never signal => v55 fence. Explains: allocator
  discrimination (F clean/G wedged single-delta), stochastic timing
  (needs a remap to hit a cached comm VA), request-tail geometry
  (1-token steps = fresh tiny AR buffers = allocator churn), Boot D
  GPU0-resident-poll (peer flag wait), host-complete enqueue, and
  why tripwires=0 (no vLLM-side index bug). Same class as CUDA
  expandable_segments-vs-IPC/NCCL aliasing (guarded upstream, NOT
  guarded on XPU).
- **F13 oneCCL knobs (libccl.so 2021.15, authorized layer)**:
  `CCL_ZE_IPC_EXCHANGE` (drmfd|pidfd; pidfd fd_manager symbols
  present), `CCL_ZE_CACHE_OPEN_IPC_HANDLES` (=0 disables open-handle
  cache -> fresh zeOpenIpcHandle per use -> stale-mapping dead),
  `CCL_ZE_CACHE_GET_IPC_HANDLES`, `CCL_ZE_CACHE` master,
  `CCL_ZE_TMP_BUF_SIZE`, `CCL_ZE_CLOSE_IPC_WA`.
- **Fix candidates**:
  - J (cheapest discriminator): `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`
    on wedge config — if clean >= 2x control TTF, stale-handle
    cache CONFIRMED as mechanism. Then measure hot-path cost
    (open per collective) — if acceptable, env-level fix in serve
    script; else K.
  - K: `CCL_ZE_IPC_EXCHANGE=pidfd` A/B (different fd lifetime).
  - L (vLLM-layer fallback if env knobs insufficient): persistent
    non-recycled staging buffer for small AR inputs/outputs in
    xpu_communicator.all_reduce (copy in -> ccl on fixed VA ->
    copy back; ~10-50 KB copies, µs-class on-stream, covers decode
    tail ARs numel<=cap; large prefill ARs keep direct path).
  - All candidates require numerics battery (Boot E lesson: oneCCL
    env flips can silently corrupt) + perf parity + >=2x control
    TTF on BOTH dtypes.



## 9. Boots J/J2/K2 — oneCCL env layer + memory forensics (F14)

- **J (both cache knobs off): init-fail** — `ze_handle_manager.cpp:42
  mem_to_ipc_handle: device_fd invalid` on both ranks at first
  collective. **J2 (OPEN off only): same fail** => under drmfd
  exchange both IPC handle caches are load-bearing; the
  CACHE_*_IPC_HANDLES=0 knob path is CLOSED.
- **Image gotcha #2 found**: v1.2.9 image ALSO bakes
  `CCL_ZE_IPC_EXCHANGE=drmfd` as ENV (oneCCL build default here is
  **pidfd**; our docker run explicitly re-sets drmfd). Deleting the
  -e line does nothing — explicit override required.
- **K2 (pidfd exchange, single delta from wedge config, 0.9 @73728,
  allocator ON): greedy canary CORRECT** ("Answer Paris" -> "
  Paris"); pressure armed 04:08; **CLEAN past the 2x bar (~89 min,
  0 fences) and stable at FULL device saturation** (see F14).
- **F14 memory forensics (user-directed; growth logger
  mem_growth_logger.sh, 5-min samples on live K2)**:
  - Device memory at steady state under 0.9: **d0 32,638.7/32,656
    MiB used = 17 MiB free; d1 11 MiB free (99.95%)**. vLLM's 0.9
    budget = 29.4 GiB => **~3.2 GiB/device consumed OUTSIDE vLLM
    accounting** (oneCCL runtime/IPC infra + ze driver objects +
    graph pools). This is the unaccounted overhead class.
  - Per worker: **10.5k VM mappings, 7,632 /dev/dri device maps**
    (avg ~240 KB) — thousands of small IPC/allocator maps; count
    STABLE at saturation on pidfd (5-min deltas = 0). Same ~10.5k
    count as wedged Boot H => mapping count is a saturation
    property, not a monotonic leak.
  - maps-size accounting is NOT reliable for device totals (VA
    20.5 GiB total / 1.8 GiB dri < torch reserved 29.4 GiB =>
    ze/umf pools under-enumerate); xpu-smi MEMORY is the source of
    truth.
  - **Mechanism synthesis**: unaccounted set fills from boot to
    saturation; at 0.9 the steady-state margin is ~17 MiB. Any
    transient extra device allocation (ccl tmp for an unusual
    collective, new IPC registration, graph scratch) forces a
    squeeze; under drmfd the squeeze (unmap/remap + cached IPC
    handles) is the stale-mapping window => wedge. 0.85 (4.9 GiB
    headroom) delays it 69 min; 0.80 (6.5 GiB) survives >=100 min;
    **pidfd survives AT 17 MiB free >=89 min** => pidfd either
    avoids the stale-path entirely or resolves squeezes safely.
  - User examples checked: prefix-caching device cost lives INSIDE
    the KV pool accounting (only host-side hash tables outside);
    chunked prefill has no persistent outside device buffer —
    neither is the culprit; the outside consumer is the
    distributed/IPC + driver + graph stack.
- **K2 perf parity (in-band)**: loop reqs ~93-108 s vs drmfd boots
  ~93-109 s; acceptance 2.33 mean — no degradation vs baselines.
- **K2 FINAL (06:17 UTC): LOOP_COMPLETE_NO_WEDGE at 129 min
  cumulative** (armed 04:08; sustain x4 + loop 24 + wave2 48 = 72
  sequential 4096-tok reqs), fence-hits=0 on every req, greedy
  determinism sha-identical (5ec03da1428857c9 x2, both "
  Paris"), math canary 17x23=391 CORRECT, sampled reasoning
  coherent. **= 3x the max drmfd control TTF (42 min) at FULL device
  saturation (17 MiB free). e4m3@73728 lane: FIX CONFIRMED.**
- **v1.2.10 BAKED** (ENV-only: `CCL_ZE_IPC_EXCHANGE=pidfd` over
  v1.2.9; sha256:341ca2fe1f60...; allocator env preserved). NOTE:
  user's verbatim docker run re-sets `-e CCL_ZE_IPC_EXCHANGE=drmfd`
  which would override the bake — Phase F must drop that line (or
  set pidfd explicitly).



## 10. Boot M — user EXACT lane dress rehearsal on baked v1.2.10

- Launched 06:17:50 UTC on **llm-scaler-exp:v1.2.10** (baked pidfd
  image; script also passes `-e CCL_ZE_IPC_EXCHANGE=pidfd`), env set
  identical to wedge config otherwise (0.9, allocator ON, fence 600s,
  CCL sycl kernels). Serve = user's verbatim lane: **e5m2 KV
  @262144, mtp4, async-scheduling, FULL_DECODE_ONLY**.
- v58diag applied (RS/RSv wrap + 3 tripwires), py-spy installed,
  watch5 armed, mem_growth_logger from t=0 (06:17:50, fresh-boot
  growth curve for the F14 question).
- HEALTH_OK ~140 s. pidfd + expandable_segments confirmed via
  /proc/1/environ; CCL_WARN lines all benign (transport/topo notices,
  no drmfd line). **KV pool 707,980 tokens @262144 (2.70x max
  concurrency)** vs 527,564 @73728-e4m3.
- **Canaries ALL PASS on user lane**: greedy Paris determinism
  sha 5ec03da1428857c9 x2 (bit-identical AND identical to K2 e4m3
  lane sha); math 17x23=391 correct; sampled reasoning coherent
  (2637-char game-design plan; content empty = 700-token budget
  consumed in think phase — normal, not corruption).
- Armed 06:22:19: sustain x4 + loop 24 + wave2 48 (chain 1488710),
  watch5 (bootM tag), poller -> NOFLAG_100MIN verdict ~08:02 UTC.

### Boot M CRASH at 31 s — F15: xe driver ENGINE RESET (new class, hardware truth)

- **watch5 fired 06:25:29** (metrics-age=157 s, running=2) — legit.
- **dmesg 06:22:50: `xe 0000:b1:00.0 Tile0:GT0 Engine reset:
  engine_class=ccs, guc_id=22` + device coredump, Reason:
  "LR job cleanup", Process python3 [1484308] = our TP0 worker
  (still alive — NOT process-exit cleanup).** GPU0 compute engine
  reset MID-RUN, mid-propose (fr freezes at "propose end" 69870.78).
- Death chain: reset kills the oneCCL kernel-mode AR persistent (LR)
  poll kernels + their semaphore completions -> both workers hang in
  `_v55_wait_event` (py-spy identical to Boot H F9 sites) +
  WorkerAsyncOutputCopy hang -> EngineCore RPC `sample_tokens`
  times out at 300 s -> **EngineDeadError 06:27:51, health 000,
  500s** -> **v55 fence fires exactly at 600 s (06:32:51) on BOTH
  workers** (fail-fast works on this class too).
- KV usage at crash: **4.3%** (NOT saturation; 230 MiB free d0) —
  the F14 saturation-squeeze story does NOT explain this event.
- Crash workload: sustain round 1, **p5-prefill (~3k-token prompt,
  max_tokens=32)** — p1-p4 completed. K2 (e4m3@73728) ran the SAME
  p5 clean x4. Deltas: e5m2 KV kernels + 262144 lane (pool 707,980
  identical — both fp8 dtypes are 1 B/token; only kernel paths
  differ) + eager prefill memory layout.
- **F15 reset history (dmesg since host boot Sep 11 10:58; 25
  resets by Boot M crash, 28 after zombie teardown — the rm -f
  itself spawned 3 more = GEM-close LR-job cleanup confirmed)**:
  - 11:23:24 b1+da ccs | 11:31:42 b1+da bcs  <- v55-era PROD crashes
  - 16:10:26 b1 | 16:14:52 da | 16:35:40/43 b1+da (quad)  <- prod
  - 18:38:10 da ccs | 19:50:21 b1 ccs+bcs + da bcs  <- v58 controls
  - 00:29:02 QUAD (b1+da ccs+bcs) = Boot G zombie teardown
  - 01:55:23 b1 ccs+bcs + 01:57:09 da ccs+bcs = **Boot H wedge**
  - 06:22:50 b1 ccs = Boot M crash
  **Every wedge event in the arc sits on engine resets; guc_id=22
  (GPU0 ccs) recurs in every mid-run event. This is the hardware
  layer beneath the "async-event stall" — the v50/v55/v58 wedge
  classes share it.**
- Reframe: wedge = xe "LR job cleanup" engine reset tearing down
  oneCCL persistent AR kernels mid-collective. What requests the
  cleanup mid-run (BO rebind? scheduler action?) is the open
  question — pidfd fixed the drmfd IPC flavor (K2 129-min proof)
  but the e5m2@262144 lane triggers it via another path at p5.
- Discriminators launched: **M2 = e4m3@262144** (max-len axis vs
  clean K2) — armed 06:38:13; **M3 = e5m2@73728** (dtype axis)
  queued. Crash exposure point p5 (~31 s) passed clean on M2.

## 11. Boots M2-M5 — trigger-axis discriminators (all pidfd, v1.2.10, 0.9)

| boot | lane | result |
|---|---|---|
| K2  | e4m3 @73728 + mtp4 | CLEAN 129 min, 0 resets |
| M   | e5m2 @262144 + mtp4 | **CRASH 31 s at p5** (engine reset) |
| M2  | e4m3 @262144 + mtp4 | **CLEAN: 4/4 rounds, FULL p1-p14, 0 resets, 0 fences** (35 min) |
| M3  | e5m2 @73728 + mtp4 | CLEAN: 4 rounds p1-p7 (= K2 baseline shape; p8+ 400-abort by design) |
| M4  | e5m2 @262144 NOSPEC (async kept) | **CLEAN: round 1 exit=0, full p1-p14 incl. 70k/131k/262k prefills** |
| M5  | e5m2 @262144 + mtp4 + VLLM_XPU_FP8_MQ=0 | **CRASH ~7.5 min (deeper than M's 31 s): dual ccs resets 07:45:07 both GPUs, EngineDeadError 07:50:30 → MQ kernel EXONERATED** |
| M6  | e5m2 @131072 + mtp4 (plain lane, MQ on) | **CRASH ~35 min: sustain 4/4 rounds CLEAN, wedge at sustain->wave1 transition** |
| M7  | e5m2 @262144 + mtp4 + blocks-override 600 (pool 574,037 < clean-73728's 606,208) | **CRASH ~10.5 min: round 1 clean, round 2 dead; reset 10:42:50 GPU0 ccs precedes onset → bearer is max_model_len-STRUCTURAL, not pool size** |
| N   | e4m3 @262144 + mtp4 (certification) | **CRASH ~71 min: sustain 2 + FULL loop 24 clean, wave2 req~27 wedge — e4m3@262144 ALSO crashes; M2's clean call overturned (35 min too short)** |

- **TRIGGER REVISED (M6): SPEC x e5m2, LATENCY DOSE-RESPONSE in
  max_model_len — not a hard 262144 boundary.** Crash latency:
  262144 -> 31 s (M, MQ on) / ~7.5 min (M5, MQ off);
  131072 -> ~35 min (M6, sustain-clean then wedge at transition);
  73728 -> CLEAN >=100 min (Boot I @0.80) + 4 rounds (M3).
  e4m3: clean 129 min @73728 (K2) AND 35 min @262144 (M2).
  (Earlier "SPEC x e5m2 x 262144" triangulation was an artifact of
  run duration — M6's 4 clean rounds outlived M/M5 crash points.)
- **M6 death chain (same class)**: wedge onset ~09:21 (= sustain
  round-4 completion / wave1 transition; 300 s before RPC timeout),
  EngineDeadError 09:31:04 (sample_tokens timeout), v55 fence
  09:36:04 TP1 num_accepted_tokens_event 600 s + TP0 step watchdog
  604 s, engine reset 09:35:45 GPU0 ccs guc_id=22. ORDERING DIFFERS
  from M/M5: reset FOLLOWS stall onset by ~14 min (cleanup of the
  wedged LR job, not the trigger) — at least two sub-mechanisms
  share the e5m2 x spec fingerprint.
- M-matrix lesson recorded: never call a lane CLEAN before the
  wave/loop workload class runs (M6 passed every crash point of
  M/M5 and died 25 min later in the loop class).
- Kernel-level facts: SPLITS is env-fixed (VLLM_FP8MQ_SPLITS default
  16, no context adaptation in this tree — the v51 "32/64" adaptive
  story does NOT apply here). The MQ gate (flash_attn.py:1394+)
  routes fp8 spec verify (uniform q 2..8) to triton_fp8_mq for BOTH
  fp8 dtypes (IS_E5M2 bitcast branch). chunked_prefill_paged_decode
  handles e5m2 and iterates by actual seq blocks.
- M2/M3/M4 teardowns were reset-free (idle close); only Boot M's
  zombie teardown spawned GEM-close resets (3) — LR kernels were
  resident only in the wedged case.
- guc_log_level raised 1 -> 5 (runtime) before M4/M5 for verbose
  GuC capture on any next reset. Boot M's coredump expired from
  sysfs after ~1 h (73699 "coredump has been deleted") — key
  sections extracted before expiry: Reason "LR job cleanup",
  live batch on ccs0, RING_INSTDONE 0xffdefffe, ROW_INSTDONE=0
  (spinning kernel holding engine).
- M5 design: single env delta VLLM_XPU_FP8_MQ=0 (documented v51
  rollback knob) -> spec verify routes back to v33 unified_attention
  3D path. Clean p5 => triton_fp8_mq IS_E5M2 branch convicted;
  crash => bearer elsewhere in spec path.
- **M5 VERDICT: crash class UNCHANGED by MQ rollback.** Armed
  07:37:41; round 1 survived the M crash point (p5, 31 s) and ran
  ~7.5 min to a deeper phase; dual ccs engine resets 07:45:07 on
  BOTH GPUs (b1 guc_id=22, da guc_id=32 — same class as Boot M);
  `TimeoutError: RPC call to sample_tokens timed out` ->
  EngineDeadError 07:50:30. Dual coredumps captured (both Reason
  "LR job cleanup", Processes = the two live TP workers 1595034 /
  1595165); key sections extracted before sysfs expiry (~1 h).
  **The v51 Triton MQ kernel (triton_fp8_mq e5m2 branch) is
  EXONERATED. The bearer is in shared spec x e5m2 plumbing.**
  Remaining suspects: draft-side e5m2 forwards (drafter KV is
  e5m2-matched per v49c policy), chunked_prefill_paged_decode e5m2
  mapping, e5m2 KV write/dequant in cache-reshard.
  Note: M5 survived LONGER than M (7.5 min vs 31 s) — MQ rollback
  may reduce (not eliminate) exposure, but 1 sample ≠ trend.
- Teardown reset ledger: host total 35 after M5 teardown (33 before,
  +2 bcs = copy-engine GEM-close noise, same signature as M4's
  active teardown; distinct from the mid-run DUAL-CCS crash
  signature).
- **M6 design (max-len threshold)**: e5m2+mtp4, max_model_len
  131072, everything else verbatim user lane (MQ kernel ON —
  single delta from Boot M is max-len). KV pool 664,337 tok
  (vs 707,980 @262144 — only 6% smaller), concurrency 5.07x.
  OUTCOME: sustain 4/4 rounds exit=0 clean 35 min (passed M's
  31 s and M5's 7.5 min crash points), then CRASH at the
  sustain->wave1 transition (loop workload class) — see death
  chain above. Teardown +2 bcs resets (host ledger 39).
- **M7 design (pool vs max-len)**: e5m2+mtp4 @262144 verbatim
  user lane + `--num-gpu-blocks-override`. Discovery at first boot:
  NATURAL pool = **740 blocks** (log: "Overriding num_gpu_blocks=
  740") — the printed "707,980 tokens" is a WEIGHTED hybrid count
  (full-attn blocks + GDN linear-attn state equivalents), so my
  1184 (= 606,208/512) premise was wrong: 1184 > 740 GREW the
  pool to 1,132,768 tok — misdesigned + over-allocation risk at
  0.9 saturation -> aborted before arming. REDESIGN: override
  **600 blocks** (~19% attention-pool cut, natural 740). Verdict
  key: fast crash (~31 s class) => bearer tracks max_model_len
  structurally (block-table width, position/spec-length math);
  slow crash (~35 min class) => pool-size/block-count axis;
  clean through loop class => pool clamp is a candidate fix.
  (Block-count ledger: 262144-e5m2-spec natural = 740; M6
  131072 natural = ? — printed 664,337 weighted.)
- **M7b OUTCOME: CRASH ~10.5 min.** Health OK 10:23, armed
  10:32:18, sustain round 1 exit=0 10:40:37, reset 10:42:50 GPU0
  ccs guc_id=22, wedge onset ~10:43:14, EngineDeadError 10:48:14
  (sample_tokens), dual v55 fences TP0 10:53:14
  (async_output_copy + num_accepted_tokens_event). No sysfs
  coredump generated this time (capture conditions differ).
  Printed pool 574,037 tok / concurrency 2.19x. **Verdict: the
  pool was SMALLER than the clean 73728 lane's 606,208 and
  262144 still crashed -> pool/capacity is NOT the gate. The
  bearer tracks max_model_len (or a tight correlate: block-table
  width 512 vs 256/144, spec plumbing that scales with max-len).
  Pool clamp is NOT a fix.**
- Crash-latency ledger @262144 e5m2+spec: 31 s (natural 740,
  MQ on), ~7.5 min (natural, MQ off), ~10.5 min (600 blocks, MQ
  on) — single samples, wide spread; @131072: ~35 min. Latency
  responds to multiple knobs (MQ, pool, max-len) — consistent
  with a rate/race modulated by workload mix, gated REQUIRED on
  e5m2 x spec.
- Teardown ledger after M7b: 47 (+3); M7a abort quad (84123 =
  10:20:03, both GPUs bcs+ccs) attributed to active teardown of
  the misdesigned over-allocated boot.
- **FIX DECISION REVOKED (Boot N, 12:11): the bearer is SPEC x
  max_model_len — NOT e5m2-specific.** e4m3@262144+mtp4+pidfd
  crashed at ~71 min (armed 11:00:32; wedge onset ~12:08:50,
  EngineDeadError 12:13:50 sample_tokens, v55 fences 12:18:50
  TP0 async_output_copy + TP1 num_accepted_tokens x2, reset
  12:18:30 GPU0 ccs guc_id=22 — reset-as-cleanup ordering like
  M6). Survived sustain 2 + FULL loop 24 + 27 wave2 reqs; M2's
  35-min "clean" was simply too short (M6 lesson, now repeated
  on e4m3). Teardown +2 (ledger 51).
- **REVISED BEARER MODEL (v2, current):** REQUIRED = mtp-spec x
  max_model_len >= 131072. KV dtype = rate modulator (e5m2
  31 s-35 min; e4m3 71 min @262144 — single sample). Pool size
  exonerated (M7b), MQ kernel exonerated (M5), nospec base path
  clean (M4, short sample). @73728 both dtypes clean >=100 min
  (K2 129 min e4m3; Boot I 100 min e5m2). 262144+mtp4 was NEVER
  long-soaked before this arc — and prod's Sep-11 crashes
  (16:10-19:50 resets) were this exact class: crashfix-v55's
  serve_user.sh ran 262144. The v55/v58 pidfd fix covered the
  73728-class wedge; the 262144-class remains open.
- **Delivery options (post-N):**
  (a) ROOT-CAUSE F15b (only path to 262144+mtp4) — in-code
      spec-step instrumentation around the propose-end -> sample
      window (Boot M freeze site) to name the hung kernel;
      SYCL_UR_TRACE / ZE loader trace viability check; 196608
      bisect if the curve matters.
  (b) INTERIM 262144 NOSPEC (async, both dtypes?) — M4 clean but
      only ~8 min; Boot O soak launched (~2.5 h, loop class).
      Perf cost: spec was 2.17x nospec @2k (v50).
      **VALIDATED: Boot O = BOOTO_NO_FIRE_180MIN (2026-09-12 16:10
      UTC) — 180 min, 0 resets over base 51, 0 EngineDeadError,
      sustain 2 rounds + loop waves clean. Lane P stood on it
      2026-09-13 (see section 7).**
  (c) INTERIM 73728 SPEC — proven both dtypes >=100 min; halves
      context.
  Phase F deferred until user picks; no 262144+mtp4 stand is
  possible today.
- **LIVE CONFIRMATION 2026-09-13 (~11:02 UTC):** host rebooted 10:07
  (reset counter cleared); user stood 262144+mtp4+e5m2 manually in
  lsv-test (v1.2.10, template bind-mounted, alias flag) — wedged
  ~49 min in: dual ccs resets GPU0 guc 22 + GPU1 guc 32 at 11:02,
  bcs teardown pair 11:04, serve left down. 3rd independent
  confirmation of the class (M, N, live prod lane).
- **Root-cause thread (OPEN) — leading hypothesis (F15b):** the
  reset's "LR job cleanup" IS the stuck kernel: an e5m2-branch
  attention kernel in the spec-only path HANGS (grid/branch scaled
  by max_model_len-correlated width, e.g. block-table entries
  144/256/512), the GuC later force-cleans it (reset), and the
  oneCCL AR kernels queued behind it never complete -> collective
  stall -> wedge. Fits: reset precedes stall (M/M5/M7b) or follows
  as cleanup (M6); nospec clean (kernel not in nospec set);
  e4m3 clean (different branch); MQ=0 still crashes (bearer in the
  NON-MQ spec verify path — _v34_mq3d_state e5m2 bitcast views
  flash_attn.py:1980-1995 — or the draft-side paged decode e5m2
  read). Next-session tools: per-kernel-launch watchdog build
  (name+shape logging around spec steps), 196608 bisect, XPU dump
  of the hung kernel (xpu-smi dump / umd capture before teardown).




## 7. Current status

- **BOOT Q ARMED 2026-09-13 17:29:44 UTC (F15b instrumented diag, user
  authorized root-cause arc):** EXACT Boot M lane (e5m2@262144+mtp4,
  pidfd, v1.2.10, fresh container) + patch_f15b.py — P1 _v55_wait_event
  wrap (all freeze sites self-naming), P2 input_prep sync marks, P3/P4
  execute_model/sample_tokens marks, P5 propose wrap + GPU-completion
  event, P6 per-AR completion ring (xpu_communicator). Watchdog dumps
  ring+py-spy+xpu-smi+dmesg at 45 s zero-progress — BEFORE the 300 s
  RPC timeout kills live state. Mirror dry-run green (all anchors
  unique, patched files compile). Lane P torn down for this boot
  (restore on fix completion; lane P poller killed). Pressure: sustain
  2 + loop 24 + 48 + 48. Round 1 exit=0 clean; baseline resets 7.
  Crash EXPECTED (class 31 s-50 min) → dump names the bearer.
- **LANE P STOOD 2026-09-13 11:32 UTC (prod service restored):**
  262144 + NOSPEC + e5m2 + async + high->xhigh alias
  (serve_user_e5m2_262k_alias.sh, template bind-mounted in
  lsv-test — docker cp of a bind-mounted file fails EBUSY, leave
  it). Standing on Boot O's 180-min clean soak. First live
  e2e proof of effort=high alias pending (post-warmup).
- **BOOT M (user lane e5m2@262144, pidfd, v1.2.10) CRASHED 31 s
  into pressure — F15 NEW CLASS: xe driver engine reset ("LR job
  cleanup") kills oneCCL AR kernels mid-collective** (see section
  10). dmesg reset history (35 total) maps onto EVERY wedge
  event since Sep 11 incl. prod crashes and Boots G/H — the
  hardware layer beneath the whole arc. NOT saturation (KV 4.3%,
  230 MiB free).
- **Discriminator matrix FINAL (section 11): bearer = SPEC x
  max_model_len >= 131072 — BOTH dtypes crash @262144 (e5m2
  31 s-35 min, e4m3 71 min; Boot N overturned M2's 35-min clean);
  both dtypes clean @73728 >= 100 min.** MQ kernel and pool size
  exonerated. Prod's Sep-11 crashes were this class (v55 lane
  ran 262144). NO 262144+mtp4 lane is deliverable today.
- **pidfd fix STANDS for its lane** (K2 e4m3@73728: 129 min
  clean; covers the 73728-class wedge) but not 262144+spec.
- Running: Boot O = 262144 NOSPEC e5m2 soak (~2.5 h, option b).
  Open threads: F15b instrumentation (option a), user decision
  on interim lane (b/c). Phase F deferred.
- **Boots J/J2 disposed** (cache knobs init-break) + memory
  forensics F14 (see section 9): ~3.2 GiB/device unaccounted
  overhead confirmed; saturation margin 17 MiB explains mem-util
  dose-response; pidfd stable AT saturation.
- **Boot I (0.80): CLEAN ≥100 min** — 2x bar met; dose-response
  0.9→5-42, 0.85→69, 0.80→100+ (headroom modulation, not capacity).
- **Boot H: WEDGED ~69 min with FIRST LIVE t0 capture** (freeze
  sites, both-end event waits, resident EU-idle poll kernels, KV
  2.95%) — see section 8. Stale-IPC-mapping theory (F12) now leads.
  Evidence: lce1/bootH/ (incl. pyspy_609/638, engine_metrics).
- **Boot F FINAL: CLEAN** — cumulative ~116 min pressure (63 loop reqs
  x ~103 s + 8 sustain rounds), 0 fences, 0 resets, greedy canary
  healthy. Exceeds 2x max control TTF (42 min).
- **Boot G: WEDGED ~25 min** (allocator convicted, D1a' dead, freeze
  between collectives) — see section 6. Evidence: lce1/bootG/.
- Boot E convicted + disposed (correctness break, not a fix path).
- v57 patcher retained in repo as disproven-experiment record; do NOT
  bake.

## 12. Side quest: reasoning_effort "high" (client compat)

Prod clients (litellm -> Claude Code) send `reasoning_effort: "high"`;
server 400s: "Unexpected reasoning effort high. Supported types are
xhigh (default), medium, and low."

- Root cause: validation lives in the MODEL's chat_template.jinja
  (/models/qwen3.8-27b-fp8, lines ~47-55), NOT in vLLM or litellm.
  Tiers only select a system-prompt instruction sentence; ZERO numeric
  coupling (no budget/max_thinking/thinking_tokens anywhere). xhigh IS
  the maximum tier this SKU ships — there is no "real" high to unlock.
- Fix: high->xhigh alias. Patched template
  /root/build/chat_template_qwen38_high.jinja (3-line normalizer
  inserted before the validation block; guard for invalid values
  intact).
- Validated offline (CPU render, in-container, zero GPU impact on
  Boot O): validate_effort_alias.py -> ALL_VALIDATE_OK (high renders
  byte-identical to xhigh; medium/low/default/preserve_thinking all
  pass; invalid 'ultra' still rejected).
- Deployment (NEXT boot — do not restart a soak lane for this):
  docker cp the template into lsv-test:/root/ + add
  `--chat-template /root/chat_template_qwen38_high.jinja` to the serve
  script (model dir is ro-mounted, so the flag route is required).
  `--default-chat-template-kwargs '{"preserve_thinking":true,
  "reasoning_effort":"xhigh"}'` stays unchanged. Long term: bake
  template+flag into the next image build.

## 13. Boot Q series — F15b forensics + Fix L discriminators (2026-09-13/14)

All boots = EXACT Boot M lane (e5m2@262144+mtp4, pidfd, v1.2.10,
oneCCL reduce path) unless noted. Evidence: lce1/bootQ*/.

- **Boot Q (f15b v1, ring 256):** TTF 30.4 min; trigger = prefill
  entering steady decode (prompt 165.8 tok/s @18:00:04; resets
  18:00:10 = b1 ccs guc 22 + da ccs guc 32); death chain resets →
  300 s RPC timeout → 600 s v55 fences. Host healthy post-reset
  (396 ARs/268 ms, latency ≤0.2 ms; workers alive-polling
  hrtimer_nanosleep), GPU eu.idle spin = stuck poll kernel. f15b v1
  watchdog MISSED (static any-DONE check never fires when completed
  events park in the window) and ring 256 spanned only ~3 s of
  prefill-class ARs (victim fell outside). fr logs disposed
  (graceful exit path).
- **Q2 (VIA_ALLGATHER discriminator, f15b v2 + ring 4096):**
  VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 (v29 FIX-1 allgather+local-add
  replaces the oneCCL reduce kernel). CRASHED ~90 s after arm —
  WORSE than control. v2 transition watchdog FIRED (5 dumps); fr
  logs SURVIVED (SIGKILL skips disposal); both ranks LOCKSTEP
  identical (9098 events each, same worst-gap instants) — no
  host-side divergence. Ring: first-PENDING = AR numel=7736320
  (prefill-class, mid-propose) + 10× 25600; wait=num_accepted_tokens;
  reset da ccs guc_id=32 @41941 (same signature).
  **VERDICT: via-allgather does NOT prevent the wedge (2× IPC data
  + clone churn, worse TTF). Disposed.**
- **Q3 (Fix L v1 = persistent small-AR staging @256 KiB, f15b v3):**
  arstage ACTIVE logged on both workers. WEDGED ON THE FIRST SANITY
  PAIR (~40 s into 2-req decode; ~5.5 min post-health). Deep ring:
  prefill-class ARs SURVIVED (7.7M-class DONE; decode reached
  72.2 tok/s, declining 47.9/43.2/35.2 → 0.0 over 40 s = wedge
  forming), freeze INSIDE draft-propose:
  sample_tokens → propose_begin → 3× AR numel=15360 PENDING →
  9× AR 5120 PENDING → propose_end/propose_gpu_done PENDING →
  execute_model → input_prep_sync done → v55wait
  num_accepted_tokens. All 5 dumps identical; pyspy: host parked
  in _v55_wait_event; reset da ccs guc_id=32 @46184 (same
  signature). **Victim class = 15360/5120 (15–30 KB), DEEP UNDER
  the 256 KiB cap; same class as Boot Q's victim (15360). Q2's
  7.7M was the prefill-phase instance — the wedge latches onto
  whatever ARs are in flight when the window opens.**
  Open question: did the victims take the staged path or a
  passthrough (v1 passed non-contiguous through; drafter ARs are
  RowParallelLinear matmul outputs → probably contiguous →
  probably staged → F12-as-scoped falsified)? → Q4 census.
- **Q4 (arstage v2 CENSUS, ARMED 2026-09-14 02:15 UTC):** v2 =
  per-class staged/direct counters keyed (numel, dtype, contig,
  stream) + periodic f15b ring marks (every 512 calls) +
  ARSTAGE_CENSUS line in every f15b dump + non-contiguous inputs
  now COPY-staged (v1 hole closed) + VIA_ALLGATHER guard (latent
  v1 copy-back bug killed). No silent uncounted direct path
  (reason=meta). Local sim green (ARSTAGE_V2_SIM_OK); mirror
  dry-run green (anchors unique on true v1.2.10 baseline,
  MIRROR_COMPILE_OK, census module verified in mirror). Boot
  01:00:15, HEALTH_OK ~01:02:40, BOTH sanity requests COMPLETED
  (already past Q3's wedge point). Armature: watch5 bootQ4 +
  sustain 2 → loop 24 → 48 → 48; poller 30×300 s BASE=19.
  Interpretation matrix at freeze: victims STAGED (single stream)
  → F12-as-scoped (user-buffer VA staleness) FALSIFIED → pivot to
  CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 under pidfd (J/J2 tested drmfd
  only) or expandable_segments:False; victims DIRECT non-contig →
  hole was real, now closed → graduation bar ≥61 min clean under
  full armature (=2× Boot Q TTF); multi-stream census → staging
  reentrancy race visible → per-stream buffers (L v3).
- **Q4 VERDICT (wedged ~02:17:42, ~5 min into sustain r1):** dump#1
  02:18:28, all 5 identical. CENSUS: every top-8 AR class
  (10240/15360/5120/76800 fp16) **100% STAGED / 0 direct /
  direct_last: none** on persistent never-remapped VAs —
  **F12-as-scoped FALSIFIED** (VA staleness cannot be the
  mechanism). Census ALSO shows **≥4 stable stream handles per
  class** issuing the same ARs (10240 under 4 streams:
  74649/53702/35904/28594) — matches gpu_model_runner side streams
  (async_output_copy / draft_token_ids_copy /
  num_valid_draft_tokens_copy / valid_sampled_token_count_copy) +
  main. Frozen tail = 3×10240 + 9×5120 PENDING mid-propose; reset
  da ccs guc_id=32 (same). → root-cause v3 = CROSS-STREAM
  COLLECTIVE ORDERING DESYNC: kernel-mode flag-poll assumes
  per-connection FIFO device order; ≥4 unsynchronized issuing
  streams → device order can differ across ranks → mutual
  poll-spin → LR-job engine reset. Explains spec-only bearer,
  lockstep hosts, staging irrelevant, TMP_BUF delay-only.
- **Q5 (Fix M = AR FUNNEL, boot 03:10:58, armed 03:24:10):**
  patch_arfunnel.py routes EVERY _all_reduce_impl through ONE
  dedicated per-device stream, ev_in.record(caller)→funnel.wait /
  run on funnel / ev_out.record(funnel)→caller.wait; capture/
  compile bypass; run()-exactly-once (local sim
  ARFUNNEL_SIM_OK). ACTIVE logged both workers; perf UNCHANGED
  (76–114 tok/s, acceptance 2.3–2.8 = funnel cost ≈ nil).
  **WEDGED 03:47:40 = TTF 23.5 min (control 30.4) — NO
  protection.** Forensics (dump ring semantics: PENDING/DONE =
  device event query at dump time, NOT host-blocking):
  (a) host NOT stuck in oneCCL enqueue — sailed through
  propose_end + next-step input_prep, parked at v55wait
  async_output_copy event (gpu_model_runner.py:4372);
  (b) last step's AR events (3×51200 + 10×10240) +
  propose_gpu_done ALL PENDING on BOTH ranks = funnel stream
  never executed them = device stopped;
  (c) engine resets 63632.456 = da ccs guc32 + b1 ccs guc22
  SIMULTANEOUS, ~1 s of doomed enqueues after;
  (d) fr rings: **AR numel sequence byte-IDENTICAL across ranks**
  (diff=0 over last 400) = host-order precondition HELD;
  (e) **ZERO all_gather in the last 400 KB of both fr rings** =
  no non-AR collective edge at freeze;
  (f) v55 600 s fence FAILED FAST as designed (RuntimeError,
  workers die loud; shutdown-synchronize still spins on dead
  device). **VERDICT: root-cause v3 FALSIFIED-as-scoped** —
  single-stream FIFO funnel with verified identical host order
  did not change TTF or signature.
- **Q6 (kernel-mode discriminator): CCL_SYCL_KERNELS=0.** The one
  common factor across every wedge variant (staging on/off,
  via-allgather, funnel on/off, TMP_BUF 0/1-delayed) is
  CCL_SYCL_KERNELS=1 (flag-poll kernel collectives) under
  sustained spec AR volume; nospec (1 stream, fewer ARs) never
  wedges; all TTFs cluster 23–33 min wall-time regardless of
  load phase (Q4 wedged 5 min into light sustain, Q5 3.5 min
  into heavy loop24 — wall-time-bound, not iteration-bound).
  Q6 = EXACT convicted lane + CCL_SYCL_KERNELS=0 ONLY (no
  funnel, no arstage — single variable). Outcomes: clean ≥61 min
  → kernel-mode flag machinery convicted + fix candidate (perf
  parity gate applies — K=0 may raise small-AR latency);
  wedges → kernel mode exonerated → pivot to ze/driver layer
  (xe heartbeat/reset domain) or IPC-handle layer
  (CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 under pidfd).
- Discipline notes: dmesg BASE must be recounted at arm time
  (dying engines emit teardown resets — Q3 death added entries);
  f15b dump cadence ≈46 s/stall, MAX 5; xe devcoredumps
  auto-delete before salvage (Q2's coredump_q2_da.bin is the only
  retained artifact); fr logs survive SIGKILL only.

### Q6/Q7/Q8 verdicts (2026-09-14) — root-cause v4: UNIDIRECTIONAL eager-AR loss

- **Q6 (CCL_ENABLE_SYCL_KERNELS=0): numerics-CORRUPT DOA.** Boots and
  serves, but oneCCL non-kernel AR computes wrong sums
  deterministically (Paris probe garbage; acceptance 1.02-1.26,
  8-10 tok/s; bcs guc36/26 resets dmesg 69787.96). Kernel-mode AR is
  correctness-mandatory (re-confirms Boot I); the fix must make it
  RELIABLE. Evidence lce1: q6_verdict.txt, f15b_dump_541/546_Q6.log,
  serve_full_Q6.log.
- **Q7 (pidfd + CCL_ZE_CACHE_OPEN_IPC_HANDLES=0): init-fatal.** Both
  workers die at first collective: UR_RESULT_ERROR_DEVICE_LOST
  (dmesg 71436.2 dual ccs resets + devcoredumps BOTH devices,
  auto-deleted 75272.7). With J/J2 (drmfd) the CACHE_* knob path is
  now closed on both fd flavors.
- **Q8 (convicted lane + f15b v4/P7): WEDGED ~28 min post-health**
  (boot 06:44:44, health ~06:47, freeze 07:14:51 = kernel 76066.7;
  sample_tokens RPC timeout 07:19:51; step-watchdog kill 07:24:52 =
  kernel 76667.76; poller BOOTQ_ENGINE_DEAD 07:40:30). Full evidence
  local: f15b_dump_540/546_Q8.log, f15b_pyspy_540/546_Q8.log,
  f15b_dmesg_Q8.log, serve_full_Q8.log, fr_540/546_Q8.log.gz (full
  27 MB rings) + 400 KB tails; lce1 mirrors.
  - **P7 never fired — WRONG RUNTIME CLASS.** The lane's drafter
    (/models/drafter, DSparkDraftModel, block_size 7) loads via
    auto_map/trust_remote_code from the CHECKPOINT-BUNDLED dspark.py,
    not the vllm registry file (patched forward provably on disk +
    anchor-verified, mirror apply green). Drafter-layer naming would
    need a copy+repointed instrumented drafter dir — deferred; not
    needed for the verdict. f15b stays v4.
  - **Dumps #1-#3 (06:47:19-06:48:51) = benign false-positives**: the
    262k sanity-prefill JIT/pause window (ring static 138+s, NO
    resets in dmesg), engine recovered; dump #4 healthy. Read f15b
    stall-dumps with this class in mind (compile pauses can look
    like stalls without any GPU hang).
  - **The wedge (dump #5 07:15:38):**
    (a) host AR order IDENTICAL cross-rank — diff=0 over last 400
    AR/propose entries; final deferred-AR enqueues paired within
    8 ms (3rd consecutive boot: Q5 funnel, Q8).
    (b) per-step EAGER-AR census at decode: exactly 1
    deferred-postprocess AR numel=25600 + 13 drafter ARs (3x25600
    selector + 10x5120 layers). Target decode ARs are graph-replayed
    (invisible to host) => the eager-AR surface is SPEC-ONLY —
    explains §11 nospec 180-min clean (zero eager ARs/step).
    (c) rank0 (da): first PENDING kernel = deferred-postprocess AR
    25600 (everything before DONE incl. prior propose_gpu_done);
    host main parked at synchronize_input_prep (gmr:4027), input-prep
    worker at v55wait async_output_copy.
    (d) rank1 (b1): ITS copy of the same deferred AR = **DONE**; first
    PENDING = first drafter selector AR 25600 inside propose; host
    parked at v55wait num_accepted deferred-postprocess (gmr:4372).
    (e) **NO GuC reset at the freeze**: both kernels spin 601 s until
    the step watchdog kills the workers; the 76667.76 b1 ccs guc22 +
    bcs guc26 resets are KILL CLEANUP. Flag-poll spinners look
    "running" to GuC — no hardware watchdog saves us. (Re-dates Q5's
    "resets at freeze" reading: kill-class, not cause.)
  - **VERDICT — root-cause v4: unidirectional eager-AR data loss.** A
    collective can only retire on rank1 if rank0 contributed =>
    rank1 RECEIVED rank0's chunk while rank0 never received rank1's.
    rank1's kernel retires and advances into propose; its next
    collective spins waiting for a rank0 that will never arrive
    (rank0's stream is stuck at its own spinning deferred-AR kernel).
    Dual spin = soft deadlock; 600 s watchdog kill. Q3/Q4 staging
    already falsified the AR-INPUT-buffer staleness flavor (victims
    100% staged through persistent buffers and still lost) => the
    stale artifact is oneCCL's own ze/IPC control state (flag/mailbox
    mappings; §9's 7,632 /dev/dri maps class) — consistent with the
    Boot F/G allocator conviction (expandable_segments unmap/remap
    aliasing onto ze mappings; the upstream-guarded class this stack
    lacks) and with margin-modulated TTF (0.9 17 MiB vs 0.85 vs 0.80).
- **Q9 (margin discriminator): convicted lane, single delta
  --gpu-memory-utilization 0.80.** Attacks squeeze FREQUENCY (F14:
  17 MiB steady margin at 0.9; 73728 lane: 0.85→69 min, 0.80→≥100
  min, allocator-off→clean ≥90). KV pool ~594k tok = 2.26x
  max_model_len (boots; OOM → Q9b = 0.85). Diagnostics Q4/Q8-proven
  inert: f15b v4 as-is + arstage v2 census (disjoint anchors; Q4
  co-ran both) + mem_growth_logger from t=0 + coredump salvage from
  BOOT + **AUTO-ARM at health** (Q8 lesson: manual arming lost the
  race by 20 min; pressure armature does not move TTF per Q5).
  Graduation: clean ≥100 min (3x control class). Wedge ≤61 min →
  margin falsified at 262144 → pivot: upstream aliasing-guard port
  (the real fix) or vendor pack with the Q8 unidirectional proof.
- **Q9 (util 0.80): WEDGED ~26.6 min post-health — margin FALSIFIED at
  262144, and the squeeze framing with it.** Boot 11:10:11, health
  11:12:34, freeze 11:39:13 (dump #1 11:40:01, entries 48s old);
  sample_tokens RPC timeout 11:44:13; v55 fence fail-fast 11:49:13;
  step-watchdog kill 11:49:16; poller BOOTQ_ENGINE_DEAD 11:47:39;
  auto-arm at health worked (ARMED 11:12:36, chain 182410, BASE=33) —
  full evidence captured live for the first time. KV pool 524,288 tok
  (exactly 2.0x maxlen; 9.11 GiB avail). Signature IDENTICAL to Q8:
  rank0 first-PENDING = deferred AR 25600, host at async_output_copy;
  rank1 at num_accepted (deferred postprocess) with drafter ARs
  PENDING; ARSTAGE_CENSUS at freeze = 100% staged / 0 direct /
  direct_last none (both ranks; deferred AR itself is fp16 staged
  17,298x) — input-buffer staleness falsified a THIRD time. fr host
  order identical (4th consecutive boot). **Mechanical falsifier:
  the mem curve was FLAT at 267 MiB (d0) / 675 MiB (d1) free from
  ~3 min post-boot to the freeze — the elastic unaccounted consumer
  fills in <3 min, and there was NO exhaustion/pressure event at the
  wedge.** Squeeze-at-margin is not the trigger; env-layer ladder is
  exhausted (staging/funnel/TMP_BUF/pidfd/drmfd/cache-knobs/K-mode/
  util). Evidence: f15b_dump_563/569_Q9.log, f15b_pyspy_563/569_Q9,
  f15b_xpusmi/dmesg_Q9, serve_full_Q9.log, fr_563/569_Q9.log.gz,
  bootQ9_memgrowth.log, watch5 t0/t90/t240 snapshots (lce1 + local).
- **Q10 (eager-AR volume discriminator): convicted lane, single delta
  VLLM_XPU_ENABLE_XPU_GRAPH=0, util back at 0.9.** Surviving models:
  (a) per-eager-AR rare race (~1e-6/AR; ~350-500k eager ARs to
  wedge; nospec immune because 0 eager ARs/step — matches Boot O),
  or (b) wall-time periodic hazard (driver/oneCCL housekeeping
  ~25 min). Graphs-off exposes the target decode ARs too (~148/step
  = 10.6x volume). Predictions: (a) wedge ~2-3 min post-health;
  (b) unchanged 23-33 min. (a) → fix = eliminate eager ARs
  (graph-capture drafter propose + deferred postprocess AR; nospec
  immunity generalizes), (b) → in-repo elimination cannot fully fix;
  vendor pack is the root-cause path. Armature = Q9's (auto-arm,
  salvage, mem logger; f15b v4 + arstage). Benign JIT-pause dumps
  expected with graphs off (known false-positive class).

### Q10 interim (2026-09-14 12:20): env delta INERT — relabeled Q10-rep; Q10b = serve-delta graphs-off

- **Three dmesg reset pairs REATTRIBUTED (all pre-arm, inside BASE=37;
  zero resets during Q10 runtime; poller correctly silent).** Anchor
  kernel 94034.73 = 12:14:19 UTC. K=90183.335 → **11:10:08** = Q9
  boot-phase transient on da (survived; Q9 health 11:12:34).
  K=92529.325 → **11:49:14** = Q9 step-watchdog kill cleanup (freeze
  11:39:13 + 601s; b1 — exact match). K=93000.558 → **11:57:05** = Q9
  container disposal teardown (repro_bootQ10.sh `docker rm -f`, da;
  launch echo 11:57:08).
- **VLLM_XPU_ENABLE_XPU_GRAPH=0 is INERT on this lane → Q10 ran as a
  THIRD replication of the convicted lane (relabel: Q10-rep).** Serve
  log: Worker_TP0 pid=561 / TP1 pid=567 executed "Capturing CUDA
  graphs (decode, FULL) 37/37" at 11:58:16-11:58:35 with the env var
  present (verified via `docker exec env`). Mechanism
  (platforms/xpu.py check_and_update_config): the `_unsafe_spec_tp_graph`
  branch (spec+TP2 → TORCH_COMPILE_DISABLE=1; its warning fired
  11:57:22/11:58:16/17) SHADOWS the `elif not
  envs.VLLM_XPU_ENABLE_XPU_GRAPH` → CUDAGraphMode.NONE fallback, and
  the serve command's explicit `--compilation-config
  '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` stands. The env gate is
  unreachable under spec+TP2.
- **Q10b (true graphs-off discriminator): serve-command delta ONLY** —
  `serve_user_q10b.sh` = serve_user.sh with cudagraph_mode
  FULL_DECODE_ONLY→NONE; env otherwise byte-identical to convicted
  lane (inert env line deleted). With NONE the same spec+TP2 branch
  still sets TORCH_COMPILE_DISABLE=1 (already set in the convicted
  lane), so the ONLY effective delta = graph capture off → target
  decode ARs run eager (~148 eager ARs/step vs 14). Predictions:
  (a) generic per-eager-AR race → wedge ~2-10 min post-health
  (10.6x volume partially offset by slower eager steps); (b) wall-time
  hazard OR race confined to the spec-AR subclass → wedge ~23-33 min
  unchanged. Scripts staged+verified on host: repro_bootQ10b.sh (no
  XPU_GRAPH env, cp of serve_user_q10b.sh), arm/poll Q10b, memgrowth
  log bootQ10b.
- Q10-rep runs to verdict first (3rd convicted-lane timing point;
  wedge window 12:22:34-12:32:34 post-health 11:59:34; at 12:14 it
  was healthy, 0 errors, traffic flowing: 487,986 prompt / 39,928
  gen tokens, acceptance 24,004/63,501 = 37.8%).

### Q10-rep verdict (2026-09-14 12:41): CLEAN 40 min — tight wall-time hazard FALSIFIED; Q10b launched (graphs truly off)

- **Q10-rep ran CLEAN 40m53s post-health** (11:59:34 → cut 12:41; at
  cut: health 200, 100,131 gen tokens, steady ~40 tok/s, 0 errors, 0
  dumps, resets still 37, poller silent). Effective config identical
  to Q8 (which wedged ~25 min) — the env delta is provably inert at
  all three read sites (envs.py default False = explicit "0";
  flash_attn.py:1166 / gpu_worker.py:768 getenv default "0"). Mem
  curve the whole run: d0 4.3 MiB / d1 63 MiB free, util 99.99/99.81
  — razor-thin margin for 40 min without a wedge (margin re-falsified
  on the convicted lane itself). **Verdict: the ~25-min wall-time
  periodic hazard model is FALSIFIED — onset is stochastic, consistent
  with the original 5-42 min band (F4) and the rare-race framing.**
  Convicted-lane timing observations now: Q8 ~25 min (wedge), Q9
  26.6 min (wedge, util 0.80), Q10-rep ≥40 min (clean, cut).
  Evidence: lce1/serve_full_Q10rep.log + bootQ10_memgrowth.log.
- **Q10b (TRUE graphs-off) launched 12:44:31, HEALTH_OK 12:46:36
  (~120s, faster — no capture), armed 12:46:36 chain=248572
  BASE=41** (+4 = Q10-rep disposal teardown at 12:44:31, pre-arm).
  Delta verified in-boot: `grep -c "Capturing CUDA graphs"` =
  **0** (Q10-rep: 5) — cudagraph_mode NONE took effect; env
  byte-identical to convicted lane. Interpretation rubric (eager
  steps slower, net wall-clock AR rate ~3-5x, per-step ~10.6x):
  **wedge ≤13 min post-health → generic per-eager-AR race**;
  **wedge in the baseline 5-42+ band → NOT volume-scaled →
  spec-AR-pair geometry (deferred AR + drafter pattern) or
  graph/eager-mix mechanism**; **clean ≥61 min → strong evidence
  against volume-scaled race flavors**. First f15b dump census
  should show the enlarged eager-AR population (target decode ARs
  now visible to arstage); benign JIT-pause dumps expected early
  (eager lane, known false-positive class).

### Q10b verdict (2026-09-14 14:42): CLEAN 115 min — GRAPH/EAGER-MIX is the mechanism (root-cause v5)

- **Q10b (all-eager, mtp4, 262144, util 0.9, one env identical to
  convicted lane; only delta cudagraph_mode NONE, verified: 0
  capture lines) ran CLEAN 115m20s+ post-health** (12:46:36 →
  14:41:56, still running at verdict as a free extended probe with
  armature live): health 200 throughout, 98,735 gen tokens at
  ~14 tok/s steady (≈3x slower than graphed, expected), 0
  TimeoutError/ASYNC-EVENT-STALL, 0 f15b dumps (not even benign
  JIT pauses), resets flat at BASE=41. Acceptance numerics sane:
  57,560/164,076 = 35.1% overall, healthy position decay
  26489/15852/9455/5764. Evidence: lce1/serve_full_Q10b_running.log,
  Q10b_metrics_114min.txt, bootQ10b_memgrowth.log (d0 115/d1 29 MiB
  free, util 99.7/99.9).
- **Statistical exclusion map (AR-count-equivalent to the graphed
  lane):** eager steps ~2.9x slower → total-eager-AR wall rate
  ~3.7x, spec-AR rate ~0.35x. 115 min clean ≡ **~425 min
  graphed-equivalent for the generic per-eager-AR race (EXCLUDED)**
  and ≡ **~40.3 min for the subclass per-spec-AR race (EXCLUDED —**
  past every observed mark AND the Q10-rep stochastic tail; the
  ~400k-spec-AR estimate passed at wall ~85 min). Wedge timing
  census: graphed lane wedges 5-42 min (7 original events, Q8 ~25,
  Q9 26.6; Q10-rep ≥40 clean = tail); all-eager ≥115 min clean;
  all-graphed nospec 180 min clean (Boot O).
- **Root-cause v5 (refines v4's unidirectional eager-AR loss): the
  wedge requires the MIX of graph-replayed target-decode AR kernels
  and EAGER oneCCL ARs from the spec lane (deferred postprocess AR
  + 13 drafter ARs) interleaving on the same oneCCL/ze IPC control
  state.** Graph replay re-programs ze/IPC state for the captured
  AR sequence; a subsequent/adjacent eager AR hits disturbed
  control state in a rare (~1e-6-class) window and loses data
  UNIDIRECTIONALLY (Q8: rank1 DONE / rank0 PENDING, same
  collective) → dual soft-deadlock. Pure lanes are immune.
- **Fix paths:** (A) eliminate the mix with graphs KEPT —
  graph-capture the spec propose span + deferred postprocess AR
  (in-repo precedent: whole-step XPU graph capture + width-aware
  dflash2 EMIT_K graphs); (B) TP1-replicate the MTP drafter
  (drafter_comm.dflash_tp1 generalization — kills 13/14 eager ARs;
  insufficient alone unless the deferred AR is also captured, and
  DFlash TP1-replication was CONVICTED CORRUPT for dflash2's conv
  layout — MTP layout differs, needs its own validation);
  (C) vendor escalation (authorized): oneCCL/ze robustness at the
  replay boundary, with the Q8/Q9 unidirectional proof + Q10b/
  Q10-rep timing census as the evidence pack. Next: code survey of
  capture boundaries / propose span / deferred AR site / MTP
  drafter TP structure (v10src copies).

### §13 addendum — lone-AR identification + Q11 design (15:10 UTC)

**Lone-AR ID (fr_tail_540_Q8.txt census, bs=2):** per-step eager-AR pattern =
`propose begin -> 3x AR 51200 (drafter selector/head, scale w/ bs) + 10x AR 10240
(drafter layer ARs, bs x 5120) -> propose end -> ONE lone AR 51200 -> (~30ms gap
covering the replay) -> next propose begin`. The 13 drafter ARs ride the DEDICATED
v27 drafter_tp comm (grouped_conv/drafter_comm lineage); the lone AR rides the
DEFAULT TP comm — the same comm whose ARs are GRAPH-REPLAYED in target decode.
It matches the Q8-convicted stuck collective (numel bs x 5 x 5120; bs1: 25600).
Source: target o-proj INT4 fast path — qwen3_next.py:914 (ntoks==1
esimd_gemv_int4 -> AR) / :928 (esimd_gemm_int4_pgrp -> AR, numel ntoks x 5120 =
bs x 5 x 5120 with ntoks = bs x (1+4) spec rows). This o-proj AR fires EAGER
(MTP forced-eager head span / non-captured pre/post-projection region) adjacent
to the replay = the ONLY same-comm replay+eager interleave in the step.
Falsified variant: mamba run_deferred_postprocess (mamba_utils.py:312-366)
contains NO collective — Triton batch-memcpy only; the "deferred postprocess
AR" v6 variant is dead. gmr:5486-5520 arena AR is boot-time only; async copy
spans are pure D2H.

**Root-cause v6 hypothesis (Q11 discriminates):** the EAGER oneCCL REDUCE
kernel on the default comm, interleaved with replay-disturbed ze/IPC state,
is the wedge trigger (v5 established the mix precondition; Q8 established the
unidirectional loss event; v6 names the kernel class).

**Q11 = convicted lane + VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1.** Single env
delta on the convicted lane (serve_user.sh, graphs FULL_DECODE_ONLY, util 0.9,
e5m2@262144, mtp4, async). Lever is in-tree (xpu_communicator_v10.py: VIA
block ~210-262, `_xpu_allgather_allreduce` :96; guard
`not torch.xpu.is_current_stream_capturing()` => captured ARs UNTOUCHED; all
EAGER ARs — lone o-proj + 13 drafter — reroute through allgather + local add,
numerics = same sum; STABLEBUF staging default ON under rows cap :34).
Predictions: (a) clean >= 61 min => eager oneCCL reduce kernel is the trigger
=> candidate prod fix; then perf battery (allgather vs reduce cost for 14
eager ARs/step) + numerics battery on both dtypes. (b) wedge in the 5-42 min
band => eager-reduce-kernel hypothesis falsified => deeper ze/IPC mechanism =>
pivot: dedicated-comm patch for the o-proj AR (drafter-coordinator style),
EagleSpeculator graph surgery (graph the propose span + o-proj AR), or vendor
pack (Q8/Q9 unidirectional proof + Q10-rep/Q10b timing census).

**Q10b disposal state:** cut at 15:07:43 UTC, health=200, 118,352 tokens,
~2h23m clean (margin beyond the 115m20s verdict poll). Final log
serve_full_Q10b_final2.log + Q10b_metrics_final.txt secured at cut.

### §13 Q11 verdict (16:21 UTC) — CLEAN 66m13s → eager oneCCL REDUCE kernel CONVICTED; VIA_ALLGATHER = candidate fix

Facts: launched 15:12:24, HEALTH_OK ~140s (graphs 37/37 FULL = convicted signature,
verified in serve log), BASE=41, ARMED 15:14:49 (chain=362623), VIA env verified
in container Config.Env + exec env; FUNCTIONAL proof: fr census 52,546x
"AR begin ... via_env=True" / 0x via_env=False at ~5 min mark — every eager AR
(lone o-proj 25600-class + 13 drafter) routes allgather+add; captured replay ARs
untouched (gate xpu_communicator:222-226). Soak: 15:14:49→16:21:02 = 66m13s
clean, health 200 throughout, resets=41==BASE flat, watch5 fence=0, poller
silent, 0 EngineDeadError, 151,282 tokens at steady ~41 tok/s aggregate (no
VIA perf collapse; STABLEBUF path [rows=bs·5 <= 512] avoids per-call clone).

Verdict: the ENTIRE 5-42 min convicted wedge band (all 3 prior replications
died inside it: Q9 30.5m, Q8-class, Q10-rep 40m53s) + the >=61-min
certification bar (2x control TTF) are cleared with ONLY the eager-reduce
kernel swapped out. Root-cause v6 CONFIRMED at mechanism-isolation level:
the trigger is the eager oneCCL REDUCE kernel path (kernel + its ze/IPC
control machinery) on the default TP comm interleaved with graph replay —
NOT the mix itself, NOT user buffers (F6), NOT eager-AR volume (Q10b), NOT
margin/squeeze (Q9), NOT the drafter-comm ARs (v27 dedicated comm clean).

Candidate prod fix: VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 (in-tree since v29,
documented in xpu_communicator:210-221 as exactly this workaround).
Numerics expectation: TP2 VIA sum = copy(g0)+add_(g1) — the same two-term
fp16 add as oneCCL reduce => battery shas should be BIT-IDENTICAL to
certified refs. Remaining certification gates: numerics battery (determinism
x3 + coherence + loopscan) on the LIVE Q11 engine + perf parity (bench3
depths / conc8 vs convicted-lane refs) + e4m3 dtype lane. Extended soak
continues during battery (battery traffic counts toward soak).

### §13 Q12 attribution boot (16:30-16:41 UTC) — P1 corruption PRE-EXISTING, VIA EXONERATED; raw-lane numerics census

Trigger: Q11 battery (f8ref q11via) passed determinism (distinct=OK,
cb8c3851b897/68332ec7c31b/05c88ff03b0c) and P2 (40-tok x6) was 6/6
identical+coherent, BUT coh_probe P1 (8x identical 6-tok temp-0 short
prompt, prefix-cache hit) = 8/8 DISTINCT GARBAGE ('.+=zuixianraphous',
'TheTheThe'...). Controls on Q11: cache-buster nonce (cache miss) = 4/4
identical coherent '**Paris**' continuation; repeated identical 40-tok
(cache hit, longer gen) = clean. => corruption = SHORT-generation +
prefix-cache-RESUME specific (spec-decode resume path), not general.

Q12 = Q11 minus VIA (only delta: env line removed; boot 16:27:42, health
~16:30, BASE=41, graphs 37/37): P1 = 8/8 DISTINCT GARBAGE AGAIN (' isUUUUU',
' is a 201', ' is xu0gu chang ['...), P2 = 6/6 identical+coherent, f8ref
deterministic x2 runs (0b21bb2d3c6a/cb95c5395fa2/95e24129958b both passes).
=> P1 short-request cache-hit corruption is a PRE-EXISTING raw-mtp4 lane
defect — the v50-convicted mtp4 k>=4 class, live in the raw user serve
(no tq4nc); NOT VIA-caused. VIA fix stands on stability + is not the
corruptor. (Historical: every coh-passing lane in the record ran tq4nc or
dflash7/mtp1; no raw-mtp4 P1 pass exists anywhere in the record.)

Open nuance (flagged): Q11-vs-Q12 f8ref hashes differ (both internally
deterministic; both P2/f8ref outputs coherent). Cross-boot knife-edge
flips on identical config are a documented lane property (v42.1 #18
class, v51 p5 flip), so single-boot-pair delta is not attributable; Q13
(VIA re-boot) will check Q11-hash reproducibility as the secondary signal.

Q12 bench3 baseline (under armature pressure, ~6-11 min post-arm, matched
protocol for Q13): ctx2k 397.6 / ctx16k 421.3 / ctx65k 348.8 tok/s decode;
conc8 360.6 tok/s aggregate; acceptance 0.745 (1881/2524) — spec decode
functional, corruption is resume-path-specific not general. Q12 left to
ride = free 4th no-VIA wedge replication (poller armed).

### §13 Q13b (VIA re-run) — numerics TRANSPARENT (bit-proof), decode perf PARITY; Q13a DEVICE_LOST = disposal artifact (17:12-17:36 UTC)

**Q13a crash (17:00-17:08):** boot at 17:00:47 force-disposed the STILL-WEDGED
Q12 (docker rm -f on frozen engine, no intervening reset). Crashed 17:05:50:
UR_RESULT_ERROR_DEVICE_LOST at gdn_attn.py:255 (block_table indexing), EngineDeadError,
+1 b1 ccs reset 17:06. Card self-recovered post-reset ("normal" in xpu-smi);
evidence in lce1/Q13_crash/. LESSON (disposal protocol): never docker rm -f a
wedged engine — quiesce/reset first (wedge leaves device state that faults the
next boot). Not a VIA data point (config = Q11 which ran 66 min clean).

**Q13b (relaunch 17:12:52, health ~17:14, ARMED 17:15:15 BASE=44):**
- f8ref hash-repro: 0b21bb2d3c6a/cb95c5395fa2/95e24129958b — BIT-IDENTICAL to
  Q12 (no-VIA). Q11 (also VIA) differs from both => hash variance is
  BOOT-stochastic knife-edge (documented v42.1 #18 class), NOT VIA. Direct
  bit-evidence: VIA boot == no-VIA boot on 160-token greedy outputs.
- bench3 parity (idle engine, 2 clean samples): ctx2k 397.0/402.8 vs Q12 397.6;
  ctx16k 491.3/416.6 vs 421.3; ctx65k 347.9/347.3 vs 348.8; acceptance
  0.741/0.745 vs 0.745 — EXACT parity on decode + spec acceptance. conc8
  erratic run-to-run on this lane (227/66/27/278/104 across states) — variance
  exceeds any VIA attribution; documented as lane noise (Q12 single sample 360.6).
  Cold-bench first-run numbers (228-292, acc 0.548) were JIT-contaminated
  (warmup 8.3s vs 0.4-0.9s warm) — superseded by idle samples.
- Lurch incident (~17:26-17:33): pkill of pressure clients mid-stream left
  long-decode zombie requests; engine lurched 155->1.5 tok/s windows, TWO f15b
  46s stall dumps (17:32:40, 17:33:26, wait=None, recovered), then drained to
  idle-healthy (running 0, KV 0%, tokens advancing throughout — NOT the wedge
  class; dumps archived lce1/Q13b_lurch/).
- Pressure chain re-armed 17:36:02 (chain=483616); certification soak target
  >= 19:15 (2h from arm).

### §13 FINAL — crashfix-v58 CLOSED: VIA_ALLGATHER certified as the mtp4+262144 wedge fix (19:48 UTC)

**Soak verdict:** Q13b poller expired clean — BOOTQ_NO_FIRE_150MIN resets=44
(19:45:23); at 19:48 health=200, 340,633 tokens, engine serving (38 tok/s,
1 req running). Soak 17:15:15 -> 19:45+ = 2h30m+ clean. via_env census at
cut: 1,806,657 True / 0 False (TP0), 1,806,709 (TP1). Combined with Q11:
~2.7M eager ARs through allgather+add, ZERO wedges, ZERO via_env=False.

**Certification summary (all gates green):**
- STABILITY: VIA boots Q11 66m13s + Q13b 150min-poller-clean (2h30m+) under
  the convicted pressure armature vs no-VIA replications Q8/Q9/Q10-rep/Q12
  ALL wedged in 10-42 min (Q12: ~10 min, metrics frozen 11,417, rings parked
  at propose_gpu_done both ranks). Cleanest A/B: identical boot script, one
  env delta.
- NUMERICS: f8ref Q13b(VIA) BIT-IDENTICAL to Q12(no-VIA)
  (0b21bb2d3c6a/cb95c5395fa2/95e24129958b); in-boot determinism exact x2;
  Q11-vs-Q12 hash delta attributed to boot knife-edge (two VIA boots
  differ, VIA==no-VIA pair matches). Spec acceptance parity 0.741/0.745 vs
  0.745.
- PERF: decode parity at all depths (idle 2-sample: ctx2k 397.0/402.8 vs
  397.6, ctx16k 491.3/416.6 vs 421.3, ctx65k 347.9/347.3 vs 348.8).
  conc8/ttft = high-variance lane noise (66-278 run-to-run), not
  attributable.
- MECHANISM (root-cause v6 CONFIRMED): with graph/eager mix intact
  (graphs 37/37 FULL, mtp4 eager head, lone o-proj AR adjacent to replay),
  swapping ONLY the eager oneCCL reduce kernel for allgather+local-add
  eliminates the wedge => trigger = the eager oneCCL REDUCE kernel path on
  the default TP comm interleaved with graph replay (v5 mix precondition +
  v8/Q8 unidirectional loss event + this isolation).

**PROD-FIX PACKAGING:** env-only — add VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1
to the serve stack (docker run -e ... on llm-scaler-exp:v1.2.10+; in-tree
since v29, xpu_communicator gate :222-226, STABLEBUF default-on cap 512
rows; captured replay ARs untouched). No image bake, no code change,
revert = remove env. Deployment note: after any wedge, NEVER docker rm -f
the wedged engine — quiesce/reset first (Q13a DEVICE_LOST lesson).

**SEPARATE ITEM (not fixed by VIA, pre-existing):** raw-mtp4 lane (no tq4nc)
serves CORRUPT short completions on prefix-cache-resume requests (P1:
8/8 distinct garbage VIA and no-VIA; cache-buster clean; v50 k>=4 class).
Prod combos that passed coh (v51/v52 cert lanes) all ran tq4nc. Fix paths:
adopt tq4nc-class drafter knobs in the user serve, or the v51 k-disposition.

**Evidence index (lce1/):** Q10b (final2 log, metrics_114min), Q11
(serve_full_Q11_via.log, fr_tail_Q11_via_TP0.log, metrics_atcut), Q12
(serve_full_Q12_wedge.log, fr tails both ranks, Q12_wedge/ 9 artifacts), Q13a
(Q13_crash/ 4 files), Q13b (bench3 cold/warm/idle/idle2/idle3, f8ref,
Q13b_lurch/ 3 files, serve_full_Q13b_final.log, metrics_final). Q13b left
running as the standing validated lane (armed, poller completed; watch5
passive).

## §14 — Q13b post-cert death (2026-09-14 20:04:42 UTC): NEW failure class on the VIA lane at 2h52m

Q13b (VIA, e5m2, mtp4, armed 17:36, sustained mixed pressure) died at
20:04:42 — ~3h after the 2h30m certification window closed. Timeline:

- Pressure healthy to the end: loop2 req33-45 all OK (4096 tok, ~100s each)
  through 20:02:17; spec acceptance 2.0-3.0, ~40 tok/s single-stream.
- 20:02:30-20:04:30 p1_matrix.py ran 100% CLEAN (15 mt-sweep cells miss/hit
  x3 + ~900-tok long-prompt cells + nonce + logprobs) CONCURRENT with req46
  (4096-tok decode mid-flight) — idle-vs-load P1 nuance below.
- 20:04:42 driver reset GPU1 (b1:00.0) ccs engine — compute hang, devcoreump
  created (coredumps_bootQ13/, bootQ13_coredump.out).
- fr tail (bootQ13/fr_tail400_t0.txt): last record = "propose begin → all
  eager draft ARs complete (via_env=True) → propose end" then silence. The
  hang is in the NEXT stage — the target verify forward (graph replay; no
  eager fr lines inside replay). This is a DIFFERENT signature from the
  pre-VIA wedge (rings parked at propose_gpu_done both ranks = eager draft
  oneCCL REDUCE hang, via_env=False tails).
- pyspy_t0: workers alive; MainThread + WorkerAsyncOutputCopy parked at
  _v55_wait_event (gpu_model_runner.py:65) — GPU event never signaled after
  the reset.
- Partial death: API server + EngineCore gone (req47 HTTP 500 → req48 conn
  refused); both workers R-state spinning 3h06 holding both render nodes
  (device mem full, util pegged — post-reset orphaned contexts) until
  SIGKILL + docker rm -f at 20:19:40 (normal teardown reset cascade, same
  benign dmesg signature as 17:12:45).
- watch5 armature worked as designed: WEDGE-ONSET 20:07:31 (metrics-age 159s),
  CAPTURED5 → bootQ13/ (dmesg/engine_metrics/fr_tail400/pyspy/serve/workers,
  t0/t90/t240). Host RSS stable ~9.97GB/worker; device mem stable-full (the
  pool) — NOT memory-driven.

Assessment: the VIA certification verdict STANDS for its class — 0 eager-AR
wedges across Q11 66m + Q13b 2h52m (~2.7M VIA ARs); the oneCCL draft-AR wedge
never recurred. This is a SECOND, rarer class (~1 event / 2.9h sustained mixed
load), site = target forward (replay). Trigger correlate: p1_matrix's novel
pattern — short prefix-cache-hit bursts + ~900-tok chase-prefill concurrent
with a 4096-tok decode — the same request family as P1 (mamba resume path).
Candidate mechanism (UNPROVEN): resume-path mamba state copy with wrong
accept-token bias — preprocess_mamba uses input_batch.num_accepted_tokens_cpu[i]
as bias for a request whose row slot may hold a STALE accepted count from the
row's previous occupant (fresh/resumed rows are not reset) — collect_mamba_copy_meta
then reads spec-state slot (bias+1) beyond what the GDN kernel wrote, or copies
from a wrong block id, corrupting GDN state/KV → wild access in a target kernel
→ hang (and, when it merely poisons numerics, P1 garbage). Matches v52e's
"zombie = all-NaN logits just past a 2048 multiple" bias diagnosis and v50's
k>=4 multi-row class. p1_matrix idle cells were CLEAN because sequential short
probes leave accepted=1 in the reused rows; garbage needs other spec traffic
to seed stale biases.

P1 refined understanding: idle matrix CLEAN even with 1 concurrent 4096-decode
(req46) — so P1 needs MORE than a single neighbor; gradient probe = p1_load.py
(3 then 6 background long-decodes + short cache-hit probes).

Next: Q14 probe boot (VIA, NO auto-arm, launched 20:22:19) → p1_load.py.
Evidence additions: bootQ13/ + bootQ13_{watch,loop2,memgrowth,coredump} +
coredumps_bootQ13/ + dmesg 20:04:42. Host cleanup 20:19:40 (workers killed,
lsv-test removed, watchers killed). tq4nc defined (this session): it is NOT an
env knob — it is --kv-cache-dtype turboquant_4bit_nc (TQ preset: 4-bit keys +
4-bit values + norm_correction, config.py:26-30; prod_restore127.sh:16,
serve_bench.sh:41 passes it as --kv-cache-dtype; image llm-scaler-exp:v1.2.5).

## §15 — P1 REPRODUCED ON DEMAND (Q14, 2026-09-14 20:42-20:50 UTC): trigger isolated + poison-persistence signature

Reproduction ladder on Q14 (VIA, e5m2, mtp4, 262144, no auto-arm):
- p1_matrix idle (Q13b): CLEAN — all mt 1-40, miss/hit, short+long prompt.
- p1_matrix concurrent with 1x 4096-decode (Q13b 20:02): CLEAN.
- p1_load (Q14 20:28): idle + 3x + 6x concurrent 300-tok decodes (temp 0):
  CLEAN 6/6 each phase. Simple batch>1 hypothesis FALSIFIED.
- EXACT armature chain (repro_sustain.sh 2 = dt_warmup_v53 mixed-stream
  pattern, then loop24): coh-P1 (PLAIN prompt "The capital of France is",
  mt=6, temp 0, x8) → GARBAGE from ~20:42 (~7 min into sustain round 1),
  distinct=8/7/6/8 per round continuously. loop24 had NOT started — the
  trigger lives in the dt_warmup_v53 pattern (stagger p11 / solo long decode
  p12 / concurrent p13 / 5-stream p14 mix), not in sampled 4096-tok requests.

Signature decomposition (diagnostic gold):
1. Under load, EVERY garbage output starts with the CORRECT first token
   (' is') — junk begins at token 2+. The request's FIRST verify step is
   correct; corruption hits its SUBSEQUENT steps = the mamba running-state
   carry (preprocess_mamba boundary carry), exactly where the bias mechanism
   operates. mt=1 probes are structurally immune: clean bit-identical logprobs
   x3 taken DURING the garbage window (20:47).
2. After pressure stopped (20:46:39), garbage PERSISTED and ESCALATED to
   token 1 ('R8\n/js,', '?/?/?/' — no correct first token), distinct=8 per
   round indefinitely (20:47-20:50+). Interpretation: the shared CACHED
   prefix block itself is now poisoned, and each finishing request
   RE-POISONS it with its own (different) garbage via the postprocess_mamba
   / run_deferred_postprocess boundary-conversion copy → self-perpetuating
   cache-poison cycle. Distinct-across-repeats despite temp-0 = each cycle
   writes different garbage back (pool-recycled content per allocation).
3. Two HTTP 500s in sustain_warmup.out during the window.
4. num_preemptions_total=0 — preemption/eviction NOT involved.

Static root-cause candidate (code-searcher verified, upstream-identical):
preprocess_mamba's new/resumed branch uses row-local
num_accepted_tokens_cpu[i] as the copy bias; the temporal copy then reads
block_ids[prev_state_idx + bias] — a speculative block that can be
never-written/pool-recycled garbage (mamba_utils.py:338; conv copy reads
offset bias past valid conv slots, :311/:325). Recycled-row staleness is
defended (add_request reset gpu_input_batch.py:466 + async remap forces
prev_idx<0 rows to 1, gpu_model_runner.py:1991-1994), but the poisoned-cache
signature now indicates the CORRUPTING WRITE path is postprocess_mamba /
run_deferred_postprocess (destination = shared cached block) fed by a bad
source read on the carry. Live capture = next step.

Q15 = capture boot (launched ~20:53): same lane + VLLM_LOGGING_CONFIG_PATH
per-module DEBUG on vllm.v1.worker.mamba_utils (logcfg_v52f.json) so the
in-tree v52f PRE-COPY / POST-COPY / DEF-PP lines (slot/prev/curr/bias/
computed/sched) log during a fresh reproduction. Baseline coh probe →
sustain round → watch onset → correlate bias lines with first garbage.

Evidence: lce1/p1_coh_under_arm_Q14.out (all rounds incl. escalation),
lce1/bootQ14_pressure.out, lce1/sustain_warmup.out, lce1/p1_load_Q14.out,
lce1/p1_matrix_Q13b.out.

## §16 P1 ROOT-CAUSED + FIXED — uniform-decode misroute of k+1-token first prefills (Q15/Q16/Q17)

§15's cache-poison theory is OVERTURNED. P1 is a length-keyed code-path
defect: a solo FIRST prefill of exactly num_spec_tokens+1 tokens is
misclassified as a uniform decode batch.

Q15 (capture boot, idle, virgin) falsifications:
- First-ever plain request = '9000*2' garbage with NO load — "load" was
  coincidental (first contact happened under load on earlier boots).
- mt=1 also drifts (' **',' **',' Paris') — token 1 itself wrong; §15's
  "mt=1 structurally immune" was situational. Logprobs probe: returned
  token == its own top logprob (top5 'cu'/':H'/':I'/'wu'/':T' — a
  code-text distribution) → the FORWARD is corrupted; token selection
  and assembly are fine.
- Lowercase "the capital of France is" (same 5 tokens, different first
  token) ALSO drifts — not keyed to one prompt identity.
- Alias probe (dragons/chemistry/legal/cooking topic request then fresh
  probe): garbage does NOT track topic vocab; wrong tokens ECHO THE
  PROBE'S OWN PROMPT (' quite', ' color', exact-case 'The') →
  intra-request/state-slot signature, not cross-request KV aliasing.
- Length sweep: broken set = EXACTLY {5} within 4..18; all other
  lengths bit-stable with miss==hits.
- /reset_prefix_cache absent; engine log silent (no errors/warnings).

Q16 = k=3 CONFIG-FLIP CONFIRMATION (fresh boot, same lane, only
num_speculative_tokens 4→3): broken length moved 5→4, and ALL 5-token
prompts (incl. the classic "The capital of France is" → ' Paris'
-0.49 bit-identical x3) are healthy. Boundary is exactly
prompt_len == num_spec_tokens + 1, deterministic under the flip.
Also explains why the P1 probe prompt was "special": it is 5 tokens on
the prod k=4 lane. The classic P1 prompt IS k+1 by accident.

ROOT CAUSE (construction-convicted in the live image v1.2.10):
gpu_model_runner.py _is_uniform_decode (:4067):
    uniform_decode = (max_num_scheduled_tokens == uniform_decode_query_len)
                     and (num_tokens == max_num_scheduled_tokens * num_reqs)
with uniform_decode_query_len = 1 + num_spec_tokens (:1016, v42d). A
SOLO first prefill of exactly k+1 tokens passes BOTH shape tests
(max_sched=P=k+1, num_tokens=P=P*1). No is-prefill/first-chunk guard
exists. The step is then built as a decode/verify batch: decode FA2
routine, decode cudagraph dispatch (FULL_DECODE_ONLY), and — the killer
on this hybrid GDN model — state handling as resume-from-mamba-slot.
The request's mamba state slot has never been written (preprocess_mamba
is a no-op for fresh sub-block prompts: prev=-1 → no copy), so the GDN
layers scan forward from stale pool memory:
- zeros in the slot → correct init → CLEAN (Q13b idle early probes,
  quiet boots) — explains per-boot variance and idle-clean runs;
- dirty slot (cudagraph capture dummy-runs write junk state into pool
  slots; load recycling rotates real other-request states) → garbage
  that looks like a continuation of OTHER text (code fragments, echoes);
- each garbage output writes garbage state back → self-perpetuating
  poison cycle + drift-across-repeats at temp 0 (§15's escalation).
Secondary fit: p1_batch — the k+1 prefill stays broken when co-scheduled
with a LONG prefill (they landed in separate steps; and co-scheduling
with real decode rows still satisfies the uniform test). mt=1 garbage
the first forward IS the corrupted step. Engine log silent: no asserts.
Upstream note: the same shape-only classifier ships upstream; for
nospec lanes udql=1, so 1-token prompts are the analogous (rare) case.

FIX (v58 P1, patch_v58_p1.py, applied in-container at boot, Q17):
in _determine_batch_execution_and_padding, immediately after the
uniform_decode classification:
    if uniform_decode and force_uniform_decode is None and num_reqs > 0:
        if (input_batch.num_computed_tokens_cpu[:num_reqs] == 0).any():
            uniform_decode = False   # + throttled INFO "v58 P1 ... fired"
Rationale: first-chunk prefills are the ONLY rows with
num_computed_tokens == 0; every real decode/verify row has
num_computed >= 1 (prompt length >= 1), so legitimate uniform-decode
batches keep the optimization. Chunked-prefill continuations
(num_computed > 0) also keep it — their state slot was written by the
previous chunk, so the decode shape was already valid there.
Capture/dummy-run callers pass force_uniform_decode != None → unaffected.
Input freshness verified at the call site (:4439): _prepare_inputs
(populates num_computed per row, gpu_input_batch.py:377) runs BEFORE
_determine_batch_execution_and_padding; the adjacent cascade-attn call
already uses the same num_computed_tokens_cpu[:num_reqs] idiom.

Q17 VALIDATION (boot = Q16 + patcher + k back to 4 = prod spec config):
- Patch APPLIED+VERIFIED (anchor count 1, marker x2).
- Guard FIRES live exactly on the misroute, both ranks:
  "v58 P1 uniform-decode prefill guard fired (#1): reqs=1 udql=5".
- Boundary probe: ALL lengths STABLE CLEAN; top-3 logprobs BIT-IDENTICAL
  to healthy-lane references (' Paris' -0.49; ' is' -0.59/-1.59/-3.08)
  → fix restores the numerically-correct path, zero drift.
- Soak (16 fresh k+1 prompts x2 + plain x3 + controls): PASS bad=0,
  ' Paris.\nThe capital of' bit-identical x3, NO poison cycle.
- f8ref q17fix: distinct=OK, hashes cb8c3851b897/68332ec7c31b/05c88ff03b0c
  = bit-identical to the certified VIA-lane reference (Q11/Q15).
- Perf spot: 49.4 tok/s solo mt=256, 44.8 aggregate conc4 mt=128 —
  inside the historical lane band; guard-fire delta during decodes = 0.

Disposition: P1 root-caused and fixed; fix validated; bake into the
prod image with the next bake (patch_v58_p1.py is boot-applied until
then). Evidence: lce1/{p1_tokenvar_Q15,p1_logprob_Q15,p1_len5probe_Q15,
p1_lenset_Q15,p1_plogprob_Q15,p1_alias_Q15,p1_len5probe_Q16,p1_batch_Q16,
p1_shared_Q16,p1_len5probe_Q17,p1_soak_Q17,f8ref_q17fix}.out,
lce1/bootQ1{5,6,7}.out, serve_user_q16.sh (k=3), patch_v58_p1.py.

## §17 — Boot B (Q18, tq4nc + VIA + P1 fix): fix VALIDATED on the armature lane; LANE CRASHED at 23 min (§14-class event #2) → tq4nc prod-candidacy BLOCKED

Q18 = Q17 lineage with ONLY `--kv-cache-dtype turboquant_4bit_nc`
(serve_user_tq4nc.sh; preset in-tree config/cache.py:28; all other
flags identical: 262144, mtp4, async, prefix caching, FULL_DECODE_ONLY,
f15b+arstage+P1 patchers, VIA env, v1.2.10). Boot 05:08:56, health
~160 s, PROBE_BOOT_READY 05:11:43.

P1-fix validation on tq4nc (ALL GREEN):
- Idle: boundary probe all lengths STABLE; top-3 logprobs essentially
  IDENTICAL to the e5m2 healthy lane (' Paris' -0.45; ' is'
  -0.58/-1.58/-3.07) → no tq4nc numeric cliff. Guard fired live
  (reqs=1 udql=5 both ranks). Soak PASS bad=0 (' Paris.\nThe capital
  of' x3 bit-identical). f8ref q18tq4nc distinct=OK (e899790d3635/
  f167d905a10b/d84100508821; differs from e5m2 hashes as expected).
  Perf: solo mt=256 55.5 tok/s (+12% vs Q17's 49.4); conc4 36.8
  agg (single sample, high-variance lane — not a gate).
- Under armature (repro_sustain 2 = dt_warmup_v53 p1-p14 x2 +
  p1_coh_watch 30 min + 2 mid-load fresh-family sweeps):
  ROUND 1 CLEAN (exit 0, 9.5 min, fence-hits=0); coh rounds 1-13 ALL
  distinct=1; mid-load sweep 1 (05:21, during round-1 pressure) ALL 8
  families STABLE with top-3 logprobs == idle values. ZERO garbage
  anywhere — the §15 on-demand P1 signature is GONE with the fix.

CRASH (~05:26:49, ~7 min into round 2, uptime 23 min):
- Engine hung mid-step → 280 s later `TimeoutError: RPC call to
  sample_tokens timed out` → EngineCore fatal → API 500s (sweep-2 +
  coh requests queued behind the hung step, all 500'd at death);
  v55 fence (ASYNC-EVENT-STALL, async_output_copy 600 s) and worker
  step watchdog (605 s) fired post-mortem. dmesg: ccs engine reset
  GPU0 (da:00.0) + device coredump — saved (Q18_crash/devcoredump).
- Fatal step captured by dump_input (SchedulerOutput): MIXED batch =
  2547-token CHUNKED-PREFILL continuation of a ~41k-token chat prompt
  (num_computed 38912) + ONE 5-token verify row (num_computed 5,
  output 1, spec [-1,-1,-1,-1]); new_block_ids_to_zero=[486]; KV
  usage 7.7%. Hang inside execute_model/async_output_copy of this
  step on GPU0 compute (ccs), NOT in oneCCL (VIA env active, no
  via_env=False, fence class = async producer dead).
- P1 guard ORTHOGONAL to the fatal step: both rows have
  num_computed>0 (not first-chunk k+1 prefills); the mixed shape
  (max_sched 2547 != udql 5) was never uniform-decode anyway.

Attribution & disposition:
- §14-class event #2 (target-forward compute hang + ccs reset +
  devcoredump). Earliest onset yet: Q13b (e5m2+VIA) died at 2h52m
  under ≥ comparable pressure; Q11 66 min clean; Q14/Q15 idle+probe
  boots clean. tq4nc lane died at 23 min with 18 min of pressure —
  strong signal of tq4nc-correlated fragility (TQ kernels visible in
  JIT monitor: _tq_full_dequant_kv, _tq_mq_decode_stage1/_fwd_stage2),
  though 1 event cannot fully separate lane-vs-chance.
- tq4nc prod-candidacy BLOCKED pending root-cause of this class; NOT
  a mandate dtype (mandate = e5m2 + e4m3). P1 fix UNAFFECTED
  (validated on this lane too).
- Evidence: lce1/Q18_crash/ (serve_full_Q18.log, dump lines 902-907,
  devcoredump_card2_q18.bin, first500/first_errors/via_census/
  stall_lines/fr_tail), lce1/{p1_len5probe_Q18,p1_len5probe_Q18_load1,
  p1_len5probe_Q18_load2,p1_soak_Q18,f8ref_q18tq4nc,q18_perf,
  p1_coh_Q18,sustain_Q18,q18_armature}.out, lce1/bootQ18.out,
  serve_user_tq4nc.sh, repro_bootQ18.sh.

## §18 — Boot C (Q19, e4m3 + VIA + P1 fix): idle ALL GREEN; died at ~61 min pressure = §14-class event #3 → class is dtype-agnostic, now the dominant open thread

Q19 = Q18 lineage with `--kv-cache-dtype fp8_e4m3`
(serve_user_e4m3_262k.sh = single-line diff vs user serve). Boot
05:38:45, health ~140 s.

IDLE VALIDATION (ALL GREEN):
- Boundary probe: all lengths STABLE; top-3 logprobs ≈ e5m2 AND tq4nc
  lanes (' Paris' -0.45; ' is' -0.58/-1.57/-3.06) → no numeric cliff
  across all three KV dtypes. Guard fires live (reqs=1 udql=5, both
  ranks). Soak PASS bad=0. f8ref q19e4m3 distinct=OK
  (52f598e7d38a/6b1c26403bfc/95e24129958b — prompt-3 hash
  BIT-IDENTICAL to the e5m2-lane certified reference).
- bench3 idle x2: ctx2k 465.2/457.4, ctx16k 380.8/381.2, ctx65k
  300.0(JIT-warm)/358.5, conc8 128.4/372.2 (= idle VIA samples for
  the conc8 study), acceptance 0.743/0.739 (e5m2 parity 0.741/0.745).
  Perf solo 54.4 tok/s (+10% vs Q17 e5m2 49.4).

SUSTAIN CERTIFICATION (sustain 6 + coh watch 65 min): rounds 1-4
CLEAN (~44 min), 51 coh rounds bit-stable (the one "distinct=2" at
death = empty-reply artifacts, NOT garbage), sweep1+sweep2 mid-load
ALL STABLE, fence-hits 0 → DIED mid round 5 at ~61 min of pressure.

Death forensics:
- Hang onset 06:39:28 == dmesg ccs ENGINE RESET GPU1 (b1:00.0) +
  devcoredump created (kernel auto-deleted it at 07:42:40, 1 h TTL —
  not saved). EngineCore RPC sample_tokens timeout 06:44:32 (the 500
  wave; round-5 exit=1, coh empty-replies). v55 fence 06:49:32 on
  BOTH num_accepted_tokens_event (deferred postprocess) AND
  async_output_copy — the classic F1 signature pair, fail-fast worked
  as designed (workers exited cleanly; NO wedged-engine state).
- Fatal step (dump_input): a LEGITIMATE uniform-decode verify step —
  2 chat decodes (num_computed 450/494, num_output 393/437) each
  scheduling 5 tokens = udql, total 10, graph-replay decode region,
  KV usage 4.6%. The P1 guard was correctly PERMISSIVE here (real
  decode rows, num_computed>0) — guard orthogonal to the hang.
- NOT a oneCCL-leak: ccs compute reset at hang onset (same class as
  Q13b/Q18), not a parked eager-AR.

§14-CLASS INCIDENCE TABLE (VIA on, mtp4@262144, v1.2.10,
repro_sustain-class pressure):
  e5m2   Q13b  died 2 h 52 m (target replay propose-end→silence)
  tq4nc  Q18   died 23 min    (mixed 2547-chunk + verify, ccs GPU0)
  e4m3   Q19   died ~61 min   (uniform-decode verify, ccs GPU1)
→ dtype-agnostic rare compute-hang class; the §1-§13 oneCCL wedge
  remains FIXED by VIA (0 eager-AR wedges across all VIA boots);
  e5m2 is the longest-lived lane.

CAPTURE LESSON (hard, for next event): the fr ring files
(fr_602/fr_608.log, 29 MB, mtime == hang onset) and f15b stall dumps
(f15b_dump_602/608.log, 06:41) live IN-CONTAINER and were LOST — the
capture script's copy loop silently failed before teardown ran. Next
event: docker cp fr_*.log f15b_dump_*.log FIRST, verify non-empty,
THEN teardown; read /sys/class/drm/*/device/devcoredump/data
immediately (1 h kernel TTL).

Disposition: e4m3 idle/correctness/determinism/perf = GREEN;
e4m3+VIA sustained-pressure crash-free = NOT ACHIEVED (1 h 01 m).
Mandate "crash-free on both dtypes" now hinges on §14-class
root-cause (open thread). Prod stands on e5m2+VIA+P1fix (longest-lived
lane + wedge fixed + P1 fixed). Evidence: lce1/Q19_crash/
(serve_full_Q19.log, dump_section, dmesg_tail, via_census, ps),
lce1/{p1_len5probe_Q19,p1_len5probe_Q19_load1,p1_len5probe_Q19_load2,
p1_soak_Q19,f8ref_q19e4m3,bench3_Q19_s1,bench3_Q19_s2,q19_perf,
p1_coh_Q19,sustain_Q19,q19_armature,bootQ19}.out.

## §19 — Boot D (Q20 conc8 study) RESOLVED; prod-restore attempt Q21 hit §14-class event #4 ON the e5m2 lane (best forensics yet); Q21b STANDING as prod; engagement closure

BOOT D (Q20 = e4m3, NO-VIA control, dispose-while-healthy protocol):
bench3 x2 idle — conc8 129.0/373.4, acceptance 0.743, ctx2k 474.7/460.2
vs Q19 VIA samples conc8 128.4/372.2, acc 0.743 → EXACT PARITY.
CONC8 STUDY CLOSED: the conc8 run-to-run spread (66-373 across the
engagement) is LANE NOISE, not VIA-attributable. VIA costs nothing at
conc8 and stays (it is the wedge fix, §13). Q20 disposed healthy
09:09:03, never near its no-VIA wedge window.

PROD RESTORE ATTEMPT (Q21 = Q17 lineage exactly: serve_user.sh
e5m2 + VIA_ALLGATHER + v58 P1 patcher, mtp4@262144, v1.2.10):
- Boot 09:17, health ~140 s. Idle validation ALL GREEN: kv fp8_e5m2;
  P1 marker 2; boundary 8/8 STABLE; soak PASS bad=0; F8REF q21e5m2 =
  EXACT certified refs cb8c3851b897/68332ec7c31b/05c88ff03b0c; bench3
  403.4/481.7/340.9 conc8 124.3 acc 0.743; solo 50.3 tok/s; VIA census
  (fr rings, NOT serve log — via_env lives in _fr.log) 31431 True /
  0 False both workers, symmetric. (Note: image /tmp contains STALE
  Sep-6 fr rings baked at image build — census must target the
  current-boot pids.)
- Sustain round 1 (battery spot, launched 09:25:34): round itself
  exited 0 with fence-hits=0 at 09:37:01 — but the ENGINE DIED at
  09:36:41. §14-CLASS EVENT #4, ON THE e5m2 PROD LANE, ~6 min into
  battery pressure (~14 min after boot). The round's green counters
  are an artifact: hang onset 09:31:36 → RPC timeout 09:36:41 (305 s,
  the class signature) → round closed before the v55 fence (600 s).

EVENT #4 FORENSICS (§18 capture lesson applied — best evidence of
the engagement, all artifacts saved BEFORE teardown):
- fr_574/fr_580 rings (6.0 MB each) frozen at hang onset (mtime
  09:31): AR begins == AR ends == 88757 on BOTH workers — ZERO
  unpaired allreduces → the VIA/allgather path (and AR generally) is
  EXONERATED for the §14 hang class. Both rings end "propose end"
  170676.918x → silence: the hang is BETWEEN propose-end and the
  first verify-forward AR, i.e. in the target-verify COMPUTE region.
  Matches Q13b's "target replay propose-end→silence" signature.
- devcoredump SAVED (first physical artifact of the class,
  devcoredump_card1_Q21.bin, 518 KB, card1 = b1:00.0, ccs reset
  guc_id=22 — IDENTICAL gpu+guc_id to Q19's reset; Q18 was GPU0).
  GuC engine reset = recovery attempt; the reset workload never
  completes → host wait never signals → RPC timeout.
- Fatal step (dump_section_Q21.txt): single legitimate verify step,
  5 tokens, spec [-1,-1,-1,-1], num_computed 2763 (>0, short ctx),
  KV usage 2.6% — P1 guard correctly permissive; NOT long-context;
  NOT first-chunk; same step class as Q19's fatal step.
- No f15b stall dumps — CONSISTENT: f15b watches the AR ring; no
  unpaired AR existed. The class is a compute hang, not comm.

§14-CLASS INCIDENCE (final): e5m2 Q13b 2h52m / tq4nc Q18 23m /
e4m3 Q19 61m / e5m2 Q21 14m — dtype-agnostic, time-to-death highly
variable (14m-2h52m), 4/4 under repro_sustain/dt_warmup battery,
0/many in idle validation, 0 in weeks of real prod traffic. Working
root-cause (evidence-backed, thread open): a compute kernel in the
target-verify forward hangs a ccs engine (GPU1 twice, guc_id=22
both times); driver resets the engine; the in-flight TP step never
completes; EngineCore RPC timeout (300 s) kills the lane. Next
session's entry point: devcoredump_card1_Q21.bin + full rings at
host lce1/Q21_crash/ (fr_574/580.log, 88757 paired AR records).

PROD RESTORE-2 (Q21b = same lineage): boot 09:41, health ~140 s,
idle certification ALL GREEN — F8REF exact certified refs, boundary
STABLE, soak PASS, bench3 403.6/482.1/348.3 conc8 127.9 acc 0.743,
solo 50.5/conc4 143.6, VIA 31546/0 both workers, P1 marker 2.
NO battery re-run (4/4 §14-class events are battery-correlated; the
battery is artificial worst-case pressure, the class has never hit
idle validation or real traffic). Q21b LEFT STANDING as prod.

ENGAGEMENT CLOSURE (v58 work order):
1. VIA_ALLGATHER prod deployment — DONE, standing (§13 fix + §19
   conc8 parity: zero VIA cost at conc8).
2. e4m3 spot-validation — idle/correctness/determinism/perf GREEN
   incl. cross-dtype bit-identity (§18); sustained crash-free NOT
   achieved (§14-class #3); blocked on the open class root-cause,
   which is dtype-agnostic (e5m2 itself is not immune — event #4).
3. Conc8 clean study — CLOSED: VIA vs no-VIA exact parity, spread =
   lane noise (§19/Q20).
4. P1 defect — ROOT-CAUSED + FIXED + VALIDATED on all three KV
   dtypes (§16-§18); patch_v58_p1.py boot-applied pending next bake.
Open thread for next session: §14-class compute-hang root-cause
(devcoredump + frozen rings in hand). tq4nc remains BLOCKED (§17).
Evidence: vllm/patches/prod/crashfix-v58/*_Q21* (15 files incl.
devcoredump_card1_Q21.bin, fr_*_tail500_Q21.log, dump_section,
serve_full_Q21.log); host lce1/Q21_crash/ (full rings) +
lce1/{bootQ21,bootQ21b,sustain_Q21,f8ref_q21e5m2,bench3_Q21,
p1_soak_Q21}.out.

§20 — §14-CLASS ROOT-CAUSE: UPSTREAM PLATFORM RACE IDENTIFIED
(GSD-12919 / intel compute-runtime #939); fr-ring forensics: sudden
cliff, zero precursor; lever matrix + Q22a design

A. DEVCOREDUMP ANALYSIS (A1) — class signature uniform:
- Q21 dump (card1=GPU1/b1:00.0, 518923 B): "Reason: LR job cleanup,
  guc_id=22". Contexts: ccs22 (class 5 = compute), Timeout
  9223372036854775807 ms (INFINITE = long-runner context), Job
  seqno=149, finished=0 — a compute batch that never completed.
  GuC fw LOADED 70.44.1 vs WANTED 70.49.4 (mismatch). Kernel
  6.17.0-1010-intel, module xe, python3, PCI 0xe223 (BMG).
- Q18 dump (card2=GPU0/da:00.0): "Reason: LR job cleanup, guc_id=32"
  — IDENTICAL reason string on the OTHER physical card.
- => 3/3 dumps with reason string identical; both cards; e5m2/e4m3/
  tq4nc. NOT a single-card hardware defect. xe GuC engine reset
  triggered by a wedged long-runner compute context.

B. FR-RING TIMING FORENSICS (A2, full rings 88757 AR records each) —
RACE, NOT DEGENERATION:
- AR begins == AR ends == 88757 BOTH workers (comm layer re-exonerated).
- Step periods: med 58.4 ms, p90 67.3; session-sixths medians FLAT
  (56.0/60.1/57.6/57.9/59.3/57.9) — zero drift. LAST 30 periods
  before the hang: 47.9-67.1 ms — statistically identical to
  baseline. The 4549th step completed normally (~57 ms), its propose
  ARs paired to the last record (final AR end 0.3-2.4 ms before
  propose-end), then ABSOLUTE SILENCE.
- Only 13 ARs >50 ms in the whole session, all at IDENTICAL wallclock
  instants on both workers (+271/+603/+612/+753 s) = client round
  boundaries, not GPU events.
- No verify-phase AR records exist anywhere in the ring: verify runs
  as GRAPH REPLAY (ARs baked into the captured graph; the fr wrapper
  only sees eager ARs). Hang onset therefore sits INSIDE graph
  replay = maximal-rate tiny-kernel dispatch, no host gaps.
- Analyzer: .tmp-tq/fr_timing.py (scp'd host /root/build).

C. UPSTREAM IDENTIFICATION (A3) — exact match, GSD-12919
(intel/compute-runtime issue #939, OPEN, needs-feedback):
Xe2 ccs engine reset; devcoredump reason IDENTICAL ("LR job
cleanup"); workload = MoE inference, "each layer dispatches many
small per-expert matmul kernels back to back"; TIMING-SENSITIVE —
survives under SYCL_UR_TRACE=2 (slowed submission); reporter's GuC
fw update did NOT fix; no confirmed fix in any NEO release through
26.27.39122.11 (thread asked reporter to test 26.18 — never
answered). Multiple independent Battlemage B580/B50 hang reports.
Our fit is total: 4/4 deaths under max-rate battery (verify of tiny
rows on a MoE = burst of tiny expert GEMMs in graph replay), 0 at
idle, dtype-agnostic, time-to-death 14m-2h52m random.

D. STACK + LEVER MATRIX:
- Container: NEO intel-opencl-icd 26.14.37833.4 (IGC 2.32.7, L0
  loader 1.28.2, gmmlib 22.9), IPEX 2.11.0+xpu; NO submission knobs
  set (no SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS etc.) —
  NEO defaults govern the doorbell cadence.
- Host: kernel 6.17.0-1010.10 = NEWEST offered (kernel lever
  UNAVAILABLE); linux-firmware 2.29 → 3.1 upgrade available (would
  align GuC 70.49.4; needs REBOOT); xe.force_execlist param present.
- NEO releases 26.18/26.22/26.27 = the untested #939 data point.
- Levers ranked: (1) NEO 26.14→26.27 container-local deb swap —
  cheap, reversible, no host impact, NEO owns submission batching /
  doorbell cadence (race participant); (2) L0 immediate-command-list
  flip — container env, rewrites doorbell pattern; (3) GuC fw
  alignment — host+reboot, LOW prior (#939 reporter unaffected);
  (4) xe.force_execlist=1 — removes GuC entirely, PERF-RISKY (gate);
  (5) kernel — unavailable; (6) submission rate-limiting — REJECTED
  (degradation). Containment sidecar: auto-relaunch watchdog on
  EngineDeadError/health-000 (mitigation only, not root fix).

E. HYPOTHESIS (final): under maximal-rate tiny-kernel submission
(dflash verify graph-replay, MoE expert GEMMs) the xe/GuC
submission path on BMG occasionally wedges a ccs context (Job
finished=0); kernel heartbeat fires engine reset ("LR job
cleanup"), both workers' contexts die, the in-flight TP step never
signals, EngineCore RPC times out ~305 s → EngineDeadError → lane
dead. Root cause lives in kernel/GuC/NEO submission stack — open
upstream (GSD-12919), NOT fixable in vLLM code. Our evidence
(3 identical devcoredumps + frozen rings + a reproducing battery)
is exactly what the open upstream thread lacks.

F. Q22a DESIGN (fix-validation, container-local, no host impact):
- Lineage: Q21b exactly + in-container NEO overlay BEFORE serve:
  docker cp 6 debs (ocloc/opencl-icd/ze-intel-gpu1/gmmlib 26.27
  + IGC core/opencl 2.38.2) + dpkg -i + version census logged.
- GATES (all green before any battery): health; NEO census
  26.27.39122.11; P1 marker 2; VIA census symmetric; f8ref e5m2 ==
  certified cb8c3851b897/68332ec7c31b/05c88ff03b0c; boundary STABLE;
  bench3 + q17_perf within certified band (NO DEGRADATION rule).
- BATTERY: 6-round repro_sustain + p1_coh_watch (Q19 armature),
  §18 capture protocol armed (rings FIRST, devcoredump immediate).
  PASS = ≥3 battery-hours 0 resets (observed MTTF under battery
  14m-2h52m) → restore standing on Q22a. Recur → confirm dump
  reason-string identical → lever 2 (cmdlist flip) → host levers
  (fw align, execlist) which REQUIRE host reboot sign-off.
