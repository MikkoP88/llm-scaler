# qwen38-dflash v29 — graphs+MTP wedge: live capture, named mechanism, exonerations

Image `llm-scaler-vllm-adv:v29` = `v28dbg` (flight recorder) + three overlays.
Verdict up front: **the KNOWN_ISSUES #11 graphs-x-spec residual is a
rank-desynced DEVICE-SIDE LIVELOCK — now identified as a oneCCL SYCL-kernel
collective spin (HBM-polled, fabric-idle) whose permanence depends on the
CCL_ZE_IPC_EXCHANGE path: drmfd = permanent death, sockets = self-healing
(>=120 s freezes, engine recovers)**. async scheduling, allocator mode,
prefix caching, mid-serving recompiles, PCIe/NUMA/governor, and eager
collectives are all convicted-innocent by direct arms or fence-safe
measurement. No graphs+MTP arm is certifiable as a fix; v29 ships the
evidence, the transport characterization, and an auto-recovery watcher.
Production stays on the v27 nospec config.

## What the image changes (vs v28dbg)

| Tag | File | Change | Off-switch |
|-----|------|--------|------------|
| FIX-1 | `xpu_communicator.py` | VIA-allgather stable output buffer: per-call `clone()` replaced by a per-shape cached buffer for contiguous decode-shaped tensors (rows <= 512); ragged prefill keeps the clone. Kills ~tens of allocs/spec step against `expandable_segments:True` exactly where pieces place their buffers. | `VLLM_XPU_VIA_STABLEBUF=0` |
| FIX-2 | `gpu_worker.py` | `_xpu_capture_stab(phase)`: `synchronize + empty_cache + synchronize` before the compile/warmup loop AND before cudagraph capture — every boot starts capture from a defragmented pool. | `VLLM_XPU_CAPTURE_STAB=0` |
| EVID | `llm_base_proposer.py` | `propose()` becomes a flight-recorder marker wrapper around `_propose_impl` (decorator preserved) — with the v28dbg AR markers this partitions each step's eager collectives drafter-vs-target in `/tmp/fr_*.log`. | — |
| FIX-3 | `frw4.sh` (host) | Stall watcher: on 50 s /metrics stall with requests running, capture xpu-smi (both GPUs) + py-spy of the **TP workers** + fr/engine tails into `/root/build/wedge_cap/w$TS/`; `restart` mode then re-rolls the boot. `once` = capture and exit (v28dbg behavior). | — |

Zero cost: fox/long_exp/conc walls identical to the v28dbg baseline
(2.9 tok/s fox, 137-142 s long_exp, conc 20/20).

## The mechanism (from 7 auto-captured live wedges)

Captures in host `/root/build/wedge_cap/` — w080544, w091903, w095947,
w103714, w110059, w112233, w123120. Consistent signature:

- **Both GPUs**: Compute Engines 100% + Copy Engines 100%, GPU util 22%,
  EU ~10% active / ~48% stall — a tiny-kernel/copy storm that never
  retires (livelock, not lockup: engines spin, work doesn't drain).
- **Hosts rank-desynced, one region apart**: one rank parked at the
  mamba-align D2H sync in `_update_states_after_model_execute`
  (`gpu_model_runner.py:1489`, `num_accepted_tokens.gpu[:n].cpu().numpy()`,
  GIL released = blocked sync); the other a region ahead inside the
  drafter `_propose_impl` (wedge #3 variant: inside an fp8
  `apply_block_scaled_mm` op call that never returned). Which rank leads
  flips between wedges — symmetric race. Across all 7 wedges the parked
  regions span FIVE distinct code sites (mamba-align `.cpu()` sync,
  drafter `_propose_impl`, fp8 GEMM op call, MTP `embed_input_ids` op
  call, GDN `gdn_attn.build`) — the parking spot is incidental
  (wherever the next device sync lands); the device-side storm is the
  only invariant.
- **Eager collectives exonerated**: 76,414/76,414 perfect AR begin/end
  pairing per rank at wedge #1. They are downstream victims, not causes.
- **fr-log caveat**: the buffered `/tmp/fr_*.log` writer never flushes at
  freeze, so its tail can under-report host progress (wedge #2's tails
  showed both ranks mid-propose while py-spy showed rank 0 pre-propose).
  **Trust py-spy over the fr tail.**

## Arm matrix (v29 image, k4 MTP, XPU graphs on)

| Arm | Boot | Battery | Back-to-back canonicals | Verdict |
|-----|------|---------|--------------------------|---------|
| A0 base | v29a0 | 8/8 + 4/4 + 20/20 clean | WEDGE #1 (484 chunks), 3 clean, WEDGE (479) | susceptible |
| A2 no `--async-scheduling` | v29a2 | 8/8 + 4/4 + 20/20 clean, zero wall cost | WEDGE @485 | async exonerated |
| A1 `expandable_segments:False` | v29a1 | 8/8 + 4/4 + 20/20 clean | WEDGE @471, unique-prompt WEDGE @532, 180 s-gap run clean x3 then WEDGE @464 | alloc + prefix-cache exonerated; gaps delay |
| A3 `TORCH_LOGS=recompiles` | v29a3 | (skipped) | 14/14 clean, ~10.7k chunks; all 20 recompiles in warmup, 0 in serving | recompile theory dead |
| A4 replicate of A3 | v29a4 | (skipped) | 8/8 clean, ~6.6k chunks (identical chunk sequence = deterministic gen) | TORCH_LOGS not special |
| A5 plain, then battery | v29a5 | none first: 4/4 canonicals clean (~3.2k chunks); then fox 6/6 + long_exp 1-2 clean | long_exp iter 3 WEDGE after 1 chunk | **within-boot proof: hazard is ACQUIRED, not boot-lottery** |

## B-phase arms (device/comm level, standard provocation = fox 6x128 + long_exp 3x2048 @65k + canonical 10x4096)

| Arm | Delta vs default (graphs ON, k4, drmfd) | Provocation result | Verdict |
|-----|------------------------------------------|--------------------|---------|
| B1 `VLLM_XPU_ALLOW_COMM_IN_GRAPH=1` | collectives captured inside pieces | 3/3 temp-0 requests return DETERMINISTIC GARBAGE ("ariusLATмоropp羞羞支配...") | **numerically broken — disqualified without provocation** |
| B2' `VLLM_XPU_ENABLE_XPU_GRAPH=0` | compile-without-capture (pieces still compiled) | 19/19 clean, output coherent | wedge-free but decode **-40%** (8.7 vs 14.7 tok/s canonical) |
| B3a `CCL_ZE_IPC_EXCHANGE=sockets` (2 boots) | IPC handle exchange host-sockets instead of DRM fds | run1: 19/19 clean; run2: fox iter1 + long_exp iter1 each froze >=120 s (watchdog WEDGE, in-flight requests died), then ENGINE SELF-RECOVERED and served 10/10 clean canonicals at full speed | livelock still triggers; **drmfd = permanent death, sockets = self-healing** |
| B3b docker `--cpuset-cpus <odd>` (+`--cpuset-mems 1`) | host NUMA pin | 3 boot attempts, all die at oneCCL init: (a) `base_thread.cpp pthread_create EINVAL` — worker affinity computed from full topology lands outside the cpuset; (b) `CCL_WORKER_AFFINITY=1` rejected ("expected at least 2 values"); (c) `=1,3` rejected ("expected at least 4 values" — the parser's demand scales with topology). Crash logs `v29b3b_crash.log`, `v29b3b2.log`, `v29b3b4x.log` | docker-level pin **incompatible with this oneCCL build** — NUMA-locality lever untestable without a oneCCL fix |

## w144515 — first full-telemetry capture (frw5, B3a run 2, transient variant)

Captured at the self-healing freeze (14:45:15, ok=1 gen=49 run=1):

- **Both** TP workers parked at the SAME site as wedge #2:
  `_update_states_after_model_execute` (`gpu_model_runner.py:1489`, mamba-align
  `.cpu()` sync), stacks identical across two py-spy dumps 5 s apart. Under
  sockets the freeze is NOT rank-desynced — both hosts wait together.
- Device storm identical to the permanent wedges (3 stats samples, 2 s apart):
  Compute Engines 100% + Copy Engines 100% + GPU util 22%, EU array
  14% active / **66% stall**, GPU mem read **80 GB/s** — but
  **PCIe only 33 MB/s rising to 68 MB/s** by sample 3 (recovery traffic
  starting). The storm is a LOCAL SPIN against HBM (a collective kernel
  polling a peer flag), not data transfer. The fabric is idle while both
  engines peg: livelock confirmed device-side, fabric healthy.

## GPU<->GPU transport characterization (P4, fence-safe timing)

Copies bypass Event/synchronize fencing on this build (P4a: impossible
13 TB/s readings) — every number below is wall-time around N iterations with a
per-iteration `.item()` readback fence on the destination tensor. Matmul
anchor validates the harness (49-86 TFLOPs depending on governor).

| Path | Latency / Bandwidth | Notes |
|------|--------------------|-------|
| same-GPU D2D 256 MiB | 240 GB/s xfer (~480 GB/s HBM traffic) | HBM healthy |
| cross-GPU direct copy | **9.5 GB/s** both directions | `can_device_access_peer=True` but slower than via-host => "P2P" is host-routed |
| via pinned host | **12.2/12.8 GB/s** per direction (D2H/H2D) | the real fabric path; ~Gen3 x16 class |
| oneCCL (xccl) AR 8x5120 (decode, k1) | 56 us raw / 114 us fenced | ~35 AR/step => ~2 ms/step, ~3% of 68 ms step |
| oneCCL AR 8x25600 (decode, k4) | 59 us raw / 109 us fenced | |
| oneCCL AR large-message ceiling | **9.7-9.8 GB/s bus** | == raw P2P copy ceiling; ccl already on the fastest path |
| `CCL_ZE_IPC_EXCHANGE=sockets` | +11 us small-AR latency, same BW | free at serve level (14.5-15.0 tok/s canonical) |
| `CCL_ENABLE_SYCL_KERNELS=0` | **474 us small AR, 3.9 GB/s ceiling** | 2.5x BW / 8x latency regression — never ship |
| `CCL_TOPO_P2P_ACCESS=0` | **HANGS** the first collective (init OK) | disqualified |
| mamba-align D2H sync (10 KB `.cpu()`) | 19.7 us | gmr:1489 sync cost is negligible |
| governor `powersave`->`performance` | matmul anchor 49->86 TFLOPs (launch-bound), fabric unchanged | CPU freq affects host dispatch only |

Env provenance: torch 2.11.0+xpu ships the ccl backend as **`xccl`**
(`oneccl_bindings_for_pytorch` does not exist as a module; `dist` backend list:
gloo/nccl/**xccl**/ucc/mpi). Serve env: `CCL_ZE_IPC_EXCHANGE=drmfd`
(oneCCL **default is pidfd** — pidfd remains an untested cell),
`CCL_TOPO_P2P_ACCESS=1`, `CCL_ENABLE_SYCL_KERNELS=1`.

## Hazard model (final, replaces v28dbg boot-lottery)

- **Acquired per-boot state, driven by cumulative multi-context decode
  traffic.** A5 is the proof: the SAME boot ran ~3.2k short-ctx chunks
  clean (benign), then after the fox battery (128k ctx) + 2 x long_exp
  (65k ctx) wedged immediately at the next decode start (chunk 1).
  A0/A2/A1 (battery first) wedged the canonical barrage at ~1.3k
  cumulative chunks; A3/A4 (no large-ctx traffic at all) stayed clean
  through 6.6-10.7k short-ctx chunks. Large-context traffic is the
  accelerant; short-ctx-only traffic accumulates far more slowly.
- **Idle gaps do not repair it** (A1: 180 s gaps stretched survival ~2.6x
  then wedged). The state persists for the boot's lifetime.
- v28dbg's "boot-lottery" observation is reinterpreted: fresh boots start
  LOW-hazard; the two clean fresh-boot batteries there had not yet
  crossed threshold.
- The user's production pattern (>=32k contexts) hits it fastest —
  matching the historical "user arm >=32k effectively deterministic"
  record (wedge at prefill->decode handoff).


## Scripts

- `frw4.sh [once|restart] <logname>` — capture/recovery watcher. py-spy
  targets `Worker_TP` procs (EngineCore parks in shm `get_response` and
  is the WRONG target — wedge #1 lesson).
- `frw5.sh [once|restart] <logname>` — full-telemetry watcher (frw4 + per
  event: `xpu-smi dump`, 3x stats samples 2 s apart = storm rate, dual
  py-spy 5 s apart, numastat, /proc/interrupts, dmesg, link sysfs, worker
  /proc status). Fired once at w144515.
- `prov.sh <tag>` — standard provocation chain (fox 6x128 -> long_exp
  3x2048 @65k -> canonical 10x4096) -> `/root/build/prov_<tag>.out`.
- `dma_suite.py` — fence-safe DMA probe (anchor matmul, P2P/via-host/D2D
  copies, 10 KB D2H sync latency). Run inside the container.
- `dma_ccl.py` — 2-rank oneCCL (backend `xccl`) AR/AG bench at decode/prefill
  shapes via `torchrun --standalone --nproc_per_node=2`.
- `tele_platform.sh` / `tele_pci.sh` — passive platform telemetry (BDFs,
  NUMA, IOMMU, THP, governor, IRQ, drm/iommu groups).
- `bw_probe.py` — P4a timing-methodology probe (documents the copy-fencing
  gap; matmul anchor validated).
- `serve_boot_pin.sh` — serve boot with docker cpuset pin (documents the
  oneCCL pthread_create EINVAL incompatibility).
- `analyze_fr.py` — partition fr-log ARs by propose region (drafter =
  12.00/call deterministic; ~23.4 target-side/step).
- `wedge_repro.py [N]` — canonical barrage, 90 s watchdog, exit 7 = wedge.
- `wedge_repro_u.py [N]` — unique prompts per iter (prefix-cache control).
- `wr_gap.py [N] [gap_s]` — idle-gap control (host `/root/build`).

## Where this leaves #11

Mechanism, now with device-side telemetry: **a oneCCL SYCL-kernel collective
spin-livelock** — the wedging collective kernel polls peer state in a tight
HBM loop (80 GB/s reads, engines 100%, EU 66% stall) while almost nothing
crosses PCIe (33 MB/s). The host parking sites (gmr:1489 etc.) are just where
each rank blocks next. The fabric, NUMA placement, governor, eager
collectives, and capture-adjacent host config are all measured healthy or
exonerated.

- The trigger requires **piece capture replay + spec decode + multi-context
  history** (A-matrix) — B2' (no capture) survives the identical provocation,
  proving capture is a necessary ingredient.
- The permanence is a function of the **IPC handle exchange path**: drmfd
  (the stack's override; oneCCL default is pidfd) wedges forever; sockets
  drains eventually (w144515: >=120 s freezes, in-flight requests die, engine
  recovers, full-speed serving resumes).
- Unshippable as-is: B1 (garbage), B3b (won't boot), SYCL_KERNELS=0 (2.5x
  slower), P2P_ACCESS=0 (hangs).
- Candidate posture, ranked: (1) **v27 nospec** (prod today) — no spec, no
  wedge; (2) **graphs + sockets** — full 14.7 tok/s decode but self-healing
  multi-minute freezes under >=32k-ctx loads; needs the frw auto-restart
  watcher and client retry to be operable; (3) **compile-no-capture (B2')** —
  clean, coherent, -40% decode; the safety fallback if spec must stay on.
- Untested cells worth one arm each: oneCCL default `pidfd` IPC exchange;
  `VLLM_XPU_MAX_CAPTURE_SIZE=0` (uncapped capture list) and a
  verify-region-only capture disable (would need an image change, env knob
  does not exist today).

