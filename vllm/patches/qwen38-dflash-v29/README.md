# qwen38-dflash v29 — graphs+MTP wedge: live capture, named mechanism, exonerations

Image `llm-scaler-vllm-adv:v29` = `v28dbg` (flight recorder) + three overlays.
Verdict up front: **the KNOWN_ISSUES #11 graphs-x-spec residual is a
rank-desynced DEVICE-SIDE LIVELOCK, not a host-config bug** — async
scheduling, allocator mode, prefix caching, and mid-serving recompiles are
all convicted-innocent by direct arms. No graphs+MTP arm is certifiable as
a fix yet; v29 ships the evidence, the hazard model, and an auto-recovery
watcher. Production stays on the v27 nospec config.

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
- `analyze_fr.py` — partition fr-log ARs by propose region (drafter =
  12.00/call deterministic; ~23.4 target-side/step).
- `wedge_repro.py [N]` — canonical barrage, 90 s watchdog, exit 7 = wedge.
- `wedge_repro_u.py [N]` — unique prompts per iter (prefix-cache control).
- `wr_gap.py [N] [gap_s]` — idle-gap control (host `/root/build`).

## Where this leaves #11

The surface is now the device-side piece-replay itself: piecewise-captured
verify-region pieces + interleaved eager drafter traffic, livelocking both
compute and copy engines. Next concrete steps (untried): capture the
livelock with a device-side profiler (xpu-smi dump + oneCCL/ZE traces at
wedge time, not just utilization), an arm with `VLLM_XPU_ALLOW_COMM_IN_
GRAPH=1` (collectives inside pieces — changes the interleaving), and an
arm with piece capture disabled for the verify region only (decode graphs
elsewhere). Shipping graphs+MTP requires one of those to pass the
canonical barrage + a soak.
