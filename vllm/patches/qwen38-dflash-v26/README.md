# qwen38-dflash-v26 — GDN spec-kernel ragged-batch fix: the #11 wedge root
cause, fixed in the GPU kernels

`llm-scaler-vllm-adv:v26` = validated v25 image + a rebuilt
`vllm-xpu-kernels` wheel whose two GDN speculative kernels no longer walk
ragged spec batches rectangularly. This is the **kernel-level fix** for the
**>=32k long-context MTP wedge** (KNOWN_ISSUES #11) that no host-side arm
(v23 barrier, v24 peer rendezvous, v25 barrier+tiny-gather, NG/DTP1/MALL/E4)
could stop.

## Root cause (empirically confirmed)

The XPU GDN spec kernels in vllm-xpu-kernels @3cab97a (+ our base patch):

```
csrc/xpu/gdn_attn/causal_conv1d.hpp     :: causal_conv1d_spec_kernel
csrc/xpu/gdn_attn/gated_delta_rule.hpp  :: gated_delta_rule_spec_kernel
```

walked a **rectangular** `batch_id * num_spec_tokens + t_local` index over
`spec_token_indx` and over the q/k/v/b/a scratch that
`gdn_attn_interface.cpp` allocates at the **actual ragged size**
`{spec_token, ...}` (`spec_token = spec_token_indx.size(0)`, NOT
`num_spec_decodes * num_spec_tokens`). Meanwhile the producer
(`vllm/v1/attention/backends/gdn_attn.py`) deliberately builds **ragged**
spec metadata:

- **final verify steps**: the scheduler clamps the draft length near the
  generation-end budget, e.g. a 1-of-5 tail produces
  `spec_token_indx = arange(min(1*5, 1)) = [0]` — length 1, not 5;
- **drafter replay steps**: `num_decode_draft_tokens = accepted - 1`, so
  with acceptance E[len] ~ 1.9, ~80% of drafter steps are ragged;
- **cudagraph padding rows** flatten the tail of `spec_query_start_loc`.

On every ragged step the kernels:

1. read `spec_token_indx` **out of bounds** → garbage `global_t` → wild
   writes through `core_attn_out[global_t]` / `z[global_t]`;
2. write q/k/v/b/a rows **past** their `spec_token`-sized allocations.

The interface's own NOTE claimed the sub-kernels "walk per-request ranges
via `spec_query_start_loc`" — aspirational: the kernels never received that
pointer (the non-spec kernels do, and are correct).

**Deterministic standalone repro** (`repro_gdn_spec_oob.py`, this dir): one
`torch.ops._xpu_C.gdn_attention` call with the exact 1-of-5 verify-tail
metadata on the v22 wheel faults the device on the FIRST iteration —
`UR_RESULT_ERROR_DEVICE_LOST`, dmesg
`xe ... Engine reset: engine_class=ccs` + `Xe device coredump has been
created`. That is the serve wedge (the DTP1 py-spy capture showing both
ranks blocked enqueueing `gdn_attention` was this exact path), reproduced
with no serve in the loop. Ragged steps also explain the length
correlation (long prompts → more verify-tail + drafter-replay raggedness
before the fault lands somewhere fatal) and why every host-side arm
failed: the corruption is upstream of anything the host can order.

## Fix (`gdn_spec_fix.patch`, 3 files, +64/-15)

Both spec kernels now walk the per-request ranges the interface NOTE always
described (plumbing: `spec_query_start_loc` is passed through the
launchers, which already received it from the interface):

```cpp
const int token_start = query_start_loc[batch_id];
const int token_count = query_start_loc[batch_id + 1] - token_start;
for (int t_local = 0; t_local < token_count; ++t_local) {
  const int token_id_local = token_start + t_local;   // conv kernel
  const int t              = token_start + t_local;   // delta kernel
```

plus an upper clamp for the `num_accepted_tokens`-selected initial-state
column:

```cpp
if (init_col > num_spec_tokens - 1) init_col = num_spec_tokens - 1;
```

(the stock code only clamped the bottom). For full batches
`query_start_loc[b] == b*num_spec_tokens`, so indices are **unchanged**;
the full-batch reference outputs are bit-identical across wheels (checked
by the repro's Case F cross-wheel comparison).

## Files

- `gdn_spec_fix.patch` — the kernel fix (applies to the patched tree; for
  reference, since the image ships the fixed wheel + fixed tree).
- `causal_conv1d.hpp`, `gated_delta_rule.hpp`, `gdn_attn_interface.cpp` —
  the fixed sources, refreshed into the image's provenance tree
  `/llm-scaler/vllm/vllm-xpu-kernels`.
- `build_vxk_wheel.sh` — rebuilds the wheel from the patched tree inside
  `intel/omix:0.1.0-devel-ubuntu24.04`, replicating the Dockerfile stage-1c
  recipe (torch==2.11.0+xpu, uv venv, `pip wheel --no-build-isolation
  --no-deps`, `MAX_JOBS=16` on this 64-core/376 GB host).
- `repro_gdn_spec_oob.py` — the deterministic wedge repro / wheel
  regression test. Exit 0 = clean; 1 = OOB writes or device fault.
  Case F saves `full_ref.pt` for cross-wheel bit-comparison (place the
  previous wheel's reference at `full_ref.prev.pt` to auto-compare).
- `serve.sh`, `bench_cargame.sh`, `cargame_client.py`, `monitor3.sh` —
  unchanged from v25.

## Build

The wheel is not committed. On the build host (10.20.3.64):

```
# tree with base patch + gdn_spec_fix.patch applied at /root/build/vxk
/root/build/kernel_fix/build_vxk_wheel.sh          # -> /root/build/wheels/*.whl
cp /root/build/wheels/vllm_xpu_kernels-*.whl .     # stage next to Dockerfile
docker build -t llm-scaler-vllm-adv:v26 .
```

## Wheel verification (v22 vs v26)

| case | v22 wheel (buggy) | v26 wheel (fixed) |
|---|---|---|
| F full 5-token batch | finite, reference saved | bit-identical, finite |
| R ragged 1-of-5 (50 iters) | **device fault on iter 0** (UR_RESULT_ERROR_DEVICE_LOST, xe ccs engine reset) | 0/50 phantom writes |
| R2 ragged 5+2 of 2x5 (50 iters) | (device already lost) | 0/50 phantom writes |
| verdict | BUGGY WHEEL | CLEAN |

## Serve validation (2026-08-30, host 10.20.3.65, image v26)

**The kernel fix is real but it is NOT the serve-wedge fix.** The OOB
walks device-fault the GPU deterministically on ragged shapes (proven
standalone) and are now gone; the ≥32k serve stall, however, persists on
the fixed wheel. Arm matrix (one boot per arm; probe = `wedge_probe.py
32k:3`; MTP k4 tq4nc unless noted; "healthy" tok/s = real 64-token
decodes, excluding the instant-EOS `ntok=1` artifact):

| arm | spec | graphs | KV | wedge rate | healthy tok/s |
|---|---|---|---|---|---|
| stock | MTP k4 | FULL_DECODE_ONLY | tq4nc | **2/3** | 27.9 |
| oneCCL SYCL kernels off (`CCL_ENABLE_SYCL_KERNELS=0`) | MTP k4 | FULL | tq4nc | **2/3** | — |
| ESIMD GDN spec off (`DISABLE_ESIMD_GDN_SPEC=1`) | MTP k4 | FULL | tq4nc | **1/3** | — |
| KV dtype `auto` (fp16) | MTP k4 | FULL | fp16 | **1/3** | 25.1 |
| k=1 (verify q_len=2) | MTP k1 | FULL | tq4nc | **1/3** | 27.6 |
| PIECEWISE capture | MTP k4 | PIECEWISE | tq4nc | **1/3** | 18.6 |
| compiled MTP head (`VLLM_XPU_MTP_EAGER_HEAD=0`) | MTP k4 | FULL | tq4nc | boot crash | — |
| comm-out-of-graph (`VLLM_XPU_ALLOW_COMM_IN_GRAPH=0`) | MTP k4 | FULL, colls eager | tq4nc | 0/5 @32-67k, **1/1 @133k** | 9-17 (short-ctx 78) |
| **enforce-eager** | MTP k4 | none | tq4nc | **0/11 @32k, 0/1 @133k, 0/1 @262k** | 15-20 |
| **no spec** | none | FULL | tq4nc | **0/3** | 28.2 |

Refined signature (py-spy, 3 rounds over one wedge; consistent with the
v22-era native captures in KNOWN_ISSUES #11): the victim rank's host
blocks at the align-mode `.cpu()` D2H in
`_update_states_after_model_execute` (gpu_model_runner.py:1489) — per
the v22 native stacks it cannot even be SUBMITTED (in-flight window
full of never-retiring collective kernels); the peer rank runs a step
ahead inside the eager MTP head. Both GPUs show Compute+Copy engines
100% at ~22% EU util; no xe reset. Wedged requests are lost (freed only
on client disconnect); the engine itself recovers and serves later
requests.

Conclusion: **the serve wedge requires spec x (any) cudagraphs x ≥32k
context.** It is independent of the GDN spec kernels (fixed here), the
ESIMD fused kernel, the TQ kernels, and the oneCCL kernel transport.
The trigger surface is the graphed spec pipeline against the eager
drafter under TP=2 at long context. Mechanism identified post-matrix
via oneCCL debug tracing (captured collective replay vs the drafter's
interleaved eager collectives — see the Mechanism block in KNOWN_ISSUES
#11); `VLLM_XPU_ALLOW_COMM_IN_GRAPH=0` clears ≤67k but not 133k, so a
second long-ctx trigger remains. Not fixed in v26; tracked as
KNOWN_ISSUES #11 (workarounds there).

**Performance note (32k, first-64-token windows):** nospec+graphs
28.2 tok/s > MTP+graphs healthy 19-25 > MTP eager 15-20. At ≥32k, MTP
is a net loss on this setup even when it does not wedge; the
recommended long-context configuration is **no spec + FULL_DECODE_ONLY
graphs** — both the only clean and the fastest arm. Full-length no-spec
battery on this image (2026-08-30 late): **0 wedges in 10 probes** —
real decodes ~33k: 27.75/27.93, ~67k: 23.60/23.64, ~133k: 18.45/18.49,
~262k: 12.71 tok/s (TTFT 184.7 s) — plus canonical car-game correctness
PASS (coherent HTML+canvas). Details: `vllm/PERF_TUNING.md` (v26
no-spec battery section).
