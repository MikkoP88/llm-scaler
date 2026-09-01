# Ticket B (draft) — oneCCL: cross-post to uxlfoundation/oneCCL #212 / #215

Status: **POSTED 2026-09-01** —
#212: https://github.com/uxlfoundation/oneCCL/issues/212#issuecomment-5495759967
#215: https://github.com/uxlfoundation/oneCCL/issues/215#issuecomment-5495760342
(reframed per the v31 conviction: our hang was NOT oneCCL; posted as a
triage datapoint + timing-interaction evidence). Only #213 is already
fully covered upstream.

---

## Comment for #212 — your stale-IPC-handle hang reproduces at scale in vLLM
TP=2 serving on 2x Arc Pro B70, with full device-side telemetry

**RESOLUTION UPDATE (2026-09-01, final — please read first): our livelock was
ultimately convicted OUTSIDE oneCCL.** A discriminator matrix on the same
stack convicted it to **vLLM's torch-inductor-compiled piecewise decode path
x speculative decoding** (details below): booting with `TORCH_COMPILE_DISABLE=1`
while KEEPING XPU graph capture on — i.e. the same oneCCL 2021.15 executing
every eager collective in the same TP=2 decode loop — survives the full 65k-
context provocation battery completely clean, while the inductor-compiled
configuration wedges at the first 65k chunk (and at chunk ~563 without
capture). So we are NOT reporting your stale-IPC-handle bug; we are posting
because (a) our telemetry signature is indistinguishable from yours at the
device level (Compute 100% + Copy 100% local spin, collective never retires),
so anyone reproducing our signature on a TP serving stack will land on your
issue, and (b) the oneCCL-side interaction datapoints below (version-
independence across 2021.15/2021.17.2; `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`
accelerating our onset ~40x; `CCL_SYCL_ALLREDUCE_TMP_BUF=1` delaying it
4-6x) show the IPC handle cache and tmp-buffer paths shift the onset timing
of a compiled-region defect without being its cause — possibly the same
timing sensitivity your reporter hit. Triage recipe for anyone with our
signature: one boot with `TORCH_COMPILE_DISABLE=1`, graphs still on — if the
hang vanishes, it is the compiled path, not oneCCL.

We have been chasing a multi-month production livelock with what appears to
be this exact mechanism class, on the same hardware (2x Arc Pro B70, BMG G31,
8086:e223). Our evidence corroborates the "collective never retires, progress
path spins" behavior and adds three things: (1) a trigger characterization
(piecewise CUDA-graph replay x speculative decode x cumulative large-context
traffic), (2) a permanence characterization by IPC exchange path, and (3)
full device telemetry at the live freeze.

**UPDATE (2026-09-01, before posting): we upgraded to oneCCL 2021.17.2 and
the livelock SURVIVES it.** Same stack with oneCCL 2021.17.2 (apt
`intel-oneapi-ccl-2021.17-2021.17.2-5`, SYCL RT still 2025.3.2): `drmfd` is
removed in 2021.17 (the env value is hard-refused at boot), `pidfd` now
works on kernel 6.17 (the #213 fix is effective for us), and the standard
provocation still wedges — both workers parked at the same host sync, both
GPUs at Compute 100% + Copy 100%. Two deltas vs 2021.15: the wedge now
SELF-HEALS under pidfd after some minutes (but the first post-recovery
request returns degenerate text, so it is not operationally better than the
permanent hang), and your workaround `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` —
unbootable on our 2021.15 — BOOTS on 2021.17.2 and makes onset ~40x WORSE
(wedge at the FIRST 65k-context decode step, vs ~1.3k chunks cumulative on
2021.15). The open IPC handle cache is load-bearing for survival on this
workload; the spin is not simply "the cache being open". Full arm data in
the post-log.

**Stack**: vLLM 0.21.1.dev0+gad7125a43 (xpu) + vllm-xpu-kernels
0.1.8.3.dev0+g3cab97a, torch 2.11.0+xpu (ProcessGroupXCCL, dist backend name
`xccl`), oneAPI 2025.3.2 / oneCCL **2021.15**, NEO 37833, kernel
6.17.0-1010-intel. Model: Qwen3.8-27B (GDN hybrid) fp8 + MTP drafter, TP=2,
tensor-parallel all_reduce on the decode path (~35 AR/step at bs<=128).

**Signature (8 auto-captured live wedges + 1 transient with full telemetry)**:
 both GPUs Compute Engines 100% + Copy Engines 100%, GPU util 22%, EU ~14%
active / 66% stall, HBM reads ~80 GB/s, PCIe only 33 MB/s — a LOCAL spin
polling peer state, fabric idle. Host ranks parked at the next device sync
(five distinct sites across wedges; most commonly a 10 KB `.cpu()` D2H in
`_update_states_after_model_execute`). Under `CCL_ZE_IPC_EXCHANGE=sockets`
the SAME storm drains after >=120 s and the engine self-recovers; under
`drmfd` it is permanent until worker kill. (Relevant to #213: pidfd is
unsupported in our build too — `CCL_WARN| pidfd is not supported, fallbacks
to drmfd exchange mode` on both TP workers at init.)

**Trigger (necessary ingredients, from a 12-arm matrix)**: piecewise XPU
graph capture (FULL_DECODE_ONLY pieces) x speculative decode x cumulative
multi-context traffic (>=32k-ctx requests accelerate it sharply; hazard is
acquired per boot, idle gaps do not repair it). Capture accelerates onset
~500x (wedge at chunk 1 with capture vs chunk ~563 for compile-without-
capture — so capture is an accelerator, not a necessary ingredient; the
necessary ingredient is the inductor-compiled region).
Eager collectives are exonerated: 76,414/76,414 perfect AR begin/end pairing
per rank at one wedge.

**Workaround arms we ran against our serving wedge (2026-08-31, results)**:

1. `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` (#212's workaround) is **unbootable on
   2021.15**: both TP workers die at the first all_reduce in
   `init_device` with
   `CCL_ERROR ze_handle_manager.cpp:42 mem_to_ipc_handle: condition device_fd != ccl::utils::invalid_fd failed`
   — under BOTH `drmfd` and `sockets` exchange. The handle manager cannot
   mint IPC handles without the cache on this version; the workaround is
   version-gated (your reporter is on 2021.17). **On 2021.17.2 it BOOTS
   (init passes) — and the wedge onset accelerates ~40x**: provocation
   wedges at the first 65k-ctx decode (1 chunk) instead of ~1.3k cumulative
   chunks. See the 2026-09-01 post-log entry.
2. `CCL_SYCL_ALLGATHERV_TMP_BUF=1` (#215's workaround) is **hard-refused on
   BMG in 2021.15**:
   `allgatherv_sycl.cpp:112 allgather_sycl_single_node: EXCEPTION: To run
   on BMG, CCL_SYCL_ALLGATHERV_TMP_BUF must be set to 0`
   (i.e. our env's `=0` pins are a hard requirement, not tuning).
3. `CCL_SYCL_ALLREDUCE_TMP_BUF=1` alone (allgatherv left 0) **boots and
   serves at full speed** — provocation: pass 1 fully clean (19/19), pass 2
   on the same boot wedged at 488 chunks (~8k cumulative) with the identical
   capture signature (w213816). Delays onset ~4-6x, does not fix. Details in
   the post-log below.

**Supporting asks (echoing yours)**: evict-on-free IPC handle invalidation
should not require `ZE_ENABLE_TRACING_LAYER`; a timeout in
`ccl_executor::wait()` would convert permanent wedges into recoverable
errors for TP serving.

---

## Comment for #215 — additional datapoint: 2021.15 + SYCL RT 2025.3.2 hangs
in TP serving; our env pins TMP_BUF=0

Your version matrix says 2021.17.2 + SYCL RT 2025.3.2 = PASS. We hung on
oneCCL 2021.15 + SYCL RT 2025.3.2 (same RT as your PASS row) in vLLM TP=2
serving on 2x Arc Pro B70 — but only under piecewise graph replay +
speculative decode (details in the #212 comment). **Resolution update: our
hang was ultimately convicted to vLLM's inductor-compiled decode path x
speculative decode, NOT oneCCL** (`TORCH_COMPILE_DISABLE=1` with graph
capture still on = fully clean through our 65k battery on 2021.15), so your
PASS row stands as far as we can tell; our `ALLREDUCE_TMP_BUF=1` result
(below: delays onset 4-6x, does not fix) is a timing interaction with the
defective compiled region, not evidence against your workaround. Two notes
for your matrix:

- Our serve env sets `CCL_SYCL_ALLREDUCE_TMP_BUF=0` and
  `CCL_SYCL_ALLGATHERV_TMP_BUF=0` (historical perf tuning predating these
  hangs) — i.e. the opposite of your workaround. We will test =1.
- We also cannot use docker `--cpuset-cpus` NUMA pinning: oneCCL init dies
  with `base_thread.cpp pthread_create EINVAL` (worker affinity computed
  from full topology lands outside the cpuset), and `CCL_WORKER_AFFINITY`
  rejects any in-set list ("expected at least N values" scales with
  topology). Happy to file that separately if useful.

---

## Post-log

- 2026-08-31: draft created (v29d session). NOT posted — awaiting maintainer
  review of wording + the CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 / TMP_BUF=1 arm
  results, which would make the comments far stronger (or unnecessary).
- 2026-08-31 (arms run): cache-off unbootable on 2021.15 (both exchange
  modes); ALLGATHERV_TMP_BUF=1 hard-refused on BMG; **ALLREDUCE_TMP_BUF=1
  alone: provocation pass 1 (fox 6x128 @65k + long_exp 3x2048 @65k +
  canonical 10x4096) 19/19 CLEAN at full graphs speed (14.7-15.0 tok/s),
  zero watcher captures — vs the same battery wedging every historical k4
  boot (A-matrix ~1.3k chunks; prov_k3 fox iter 1). BUT pass 2 on the SAME
  boot WEDGED: canonical iter 2 at 488 chunks (~8k cumulative chunks),
  capture w213816 shows the identical signature (Compute 100% + Copy 100%
  device storm, worker parked in the spec drafter propose path).**
  Conclusion: the tmp-buf path DELAYS onset ~4-6x but does not fix the
  livelock — consistent with in-place SYCL-kernel allreduce exposing the
  caller's (graph-replayed) buffer to the IPC handle cache more often than
  a oneCCL-owned tmp buffer, but the staleness mechanism itself survives.
- 2026-09-01 (upgrade arm run, image llm-scaler-vllm-adv:v27-ccl1717 =
  v27 + apt intel-oneapi-ccl-2021.17-2021.17.2-5, SYCL RT 2025.3.2 kept):
  - Boot 1 — **pidfd** (drmfd is REMOVED in 2021.17: env_parser hard-refuses
    it; #213's pidfd works on our kernel 6.17, no fallback WARN): healthy
    boot, all workers on 2021.17 libs. Provocation: fox 6/6 clean, long_exp
    iter 1 WEDGE after 39 chunks, canonical RC=7. Capture w073435: both
    workers at `_update_states_after_model_execute` (gmr:1489), Compute 100%
    + Copy 100% both GPUs. NEW behavior: self-heals after ~minutes, but the
    first post-recovery request returns degenerate text ("!!!!!!!!").
    k4 logits corruption (#12) persists on this boot (coh P1 distinct=2).
  - Boot 2 — pidfd + **CCL_ZE_CACHE_OPEN_IPC_HANDLES=0**: passes init_device
    (2021.15 died there) → the workaround is live on 2021.17.2. Provocation:
    fox iter 1 WEDGE after ONE chunk, long_exp iter 1 WEDGE after one chunk,
    canonical RC=7 (w075336, same signature/site). Cache-off makes onset
    ~40x FASTER than baseline — the open IPC handle cache is load-bearing
    for survival on this workload.
  - Verdict: the livelock is oneCCL-version-independent on this stack
    (2021.15 wedges; 2021.17.2 wedges in both cache modes; 2022.x = HANG
    per #215's own matrix). #215's 2021.17.2 + SYCL RT 2025.3.2 PASS row
    does not cover our trigger (piecewise graph replay x spec x large ctx).
    Draft remains unposted pending user go-ahead; it now carries this
    datapoint as the strongest single comment for #212/#215.
- 2026-09-01 (v31 GPU window — CONVICTION, oneCCL EXONERATED): a 7-arm
  discriminator matrix on v31 (k4->3, 2021.15, same 65k battery):
  compile+capture WEDGE @1 chunk; +GDN split identical (REDUNDANT — those
  ops are already in the default split list); +`moe_ops::moe_forward_full_
  fp8_block` split identical; +ALL custom ops split (every
  `custom_esimd_kernels_vllm::*` gemm/gemv/norm + `_xpu_C::fp8_gemm*`)
  identical; compile-no-capture WEDGE @563 chunks; **capture-NO-compile
  (`TORCH_COMPILE_DISABLE=1`, whole-step XPU graph kept) = fox 6/6 +
  long_exp 3/3 @65k + canonical 10/10 ALL CLEAN @17.7 tok/s, coh bit-stable**;
  fully eager clean @9.9. The wedge is the inductor-compiled piecewise path
  x spec decode; oneCCL, XPU graph capture itself, and every custom kernel
  are exonerated. Fix shipped in our image v31.1 (gate disables only the
  compile path for spec+TP>1). The oneCCL-side arms above (version axis,
  IPC cache off, TMP_BUF=1) are retained as timing-interaction datapoints
  for the compiled-region defect. Comments reframed accordingly and POSTED:
  #212 https://github.com/uxlfoundation/oneCCL/issues/212#issuecomment-5495759967
  #215 https://github.com/uxlfoundation/oneCCL/issues/215#issuecomment-5495760342

