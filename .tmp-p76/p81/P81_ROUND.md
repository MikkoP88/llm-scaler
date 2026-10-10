# P81 — GDN Core Argument Freeze-Frame (CLOSED 2026-10-05)

**Goal.** Discriminate the wedge mechanism class at the P80-verdict site: snapshot the
host-visible argument image of every GDN core dispatch (metadata only — shape/stride/
dtype/data_ptr, **zero device syncs** so the race is not perturbed); on wedge, the last
snapshot = the hung call's argument image.

- args dirty → the fork's GDN call construction is wrong at depth.
- args clean → hazard lives behind the op (per-layer state registry via `self.prefix`,
  or the L0/UMD launch path) — next probe surface.

## Instrument

`_p81_argsnap.py` (env `P81_ARGSNAP=1`): line-buffered `/root/p81_argsnap.<pid>`,
~260 B/line, two sites in `gdn_linear_attn.py` (inserted after the P80 rungs, anchor
idempotence `llm-scaler P81`):

| Site | Fires | Captured |
|---|---|---|
| `sycl` | before `torch.ops.vllm.gdn_attention_core_xpu(` | core_attn_out, z, qkvz, ba, prefix, num_tokens, _use_buf |
| `conv` | before the ESIMD gate `handled = …` | qkvz, ba, core_attn_out, z, `_gdn_conv_esimd_ok` |

P80 ladder re-applied from a **v2 patcher** (d2h 8-space anchor + fattn 2-line anchor
fixed at source — no repair scripts needed); P80 md5s reproduced byte-identical
(tracer dc166b8d, gdn 22a0ac3a, rejsampler 01d20b79; attention c4fe0bd9 = v2 insert
shape). Posture = exact P80 clone (`boot_p81.sh` = boot_p80.sh + `P81_ARGSNAP=1`):
spec mtp×4, TP2, gmu 0.8, port 8000. Monitor = p80_monitor clone + argsnap banking.

Site-mix datum (warmup): decode rows=5 (1 seq × 5) → ESIMD path handles it (conv lines,
no sycl follow-up); chunked-prefill 8192 → conv+sycl pair (SYCL by design — prefill keeps
the upstream kernel); storm verify rows 512 > 128 → SYCL. The wedge surface is the SYCL
path either way.

## Execution

- 09:34 boot (health 200 ~09:38). Monitor premature-fire trap at boot (armed too early,
  health still 000, tree=0 → exited) — rearmed post-health. Smoke OK.
- r1 storm 09:40 → **CLEAN `ok=108/108 wall=1893s`** (wedge did not fire).
- r2 storm 10:15 → **CLEAN `ok=108/108 wall=4732s`**.
- r3 storm 11:37 → **WEDGE at T+55** (12:32:12). Monitor banked 3 rounds;
  devcoredump card1 harvested in-window (512,130 B, md5 cfb990c4…); serve_full.log banked.

Intermittence datum: 2 clean full legs then a wedge on the third, same vehicle/posture
as P80's T+25 wedge. Host-pacing masking ruled unlikely (P80 wedged with the heavier
per-call event instrument active).

## Capture — the freeze-frame

Both ranks in lockstep, last argsnap lines (per worker, `#` = call index):

```
240455-240480  512-row spec-verify step walking GDN layers 46→62   (clean, ~150 ms/layer)
240481  12:32:10.7  conv T(7880,8192) T(7880,48) T(7880,24,128)×2 … layer.0  True
240482  12:32:10.7  sycl … 'layers.0.linear_attn' num_tokens=7880 _use_buf=False
240483  12:32:12.3  conv … layer.1
240484  12:32:12.3  sycl … 'layers.1.linear_attn' num_tokens=7880 _use_buf=False
<FILE ENDS — layer 2 never fires>
```

Ladder correlation: TP1 (1865) last natural completion `gdn_core#299360` @12:32:11.94;
`STALL head=gdn_conv#299361 pend=[gdn_conv#299361, gdn_core#299362]` = layer-2's rungs
queued behind the hung op, **never completed, not even at reset**. TP0 (1859): same hang
point; the 12:32:43 reset force-completed its 2 queued events (P80 pattern).

**Hung call (both ranks): the layer-1 `gdn_attention_core_xpu` of a 7880-token
chunked-prefill**, deep-storm engine state (T+55, 63/108 rows done, deep tail).

## Verdict — CLEAN ARGS LAW

Every host-visible field of the hung call's image is **nominal**: shapes/strides/dtypes
identical in structure to the ~100k healthy GDN core calls banked across 2.5 h (including
thousands of same-size 7880/8192-token prefill chunks), pointers in the expected
per-chunk allocation sets, layer-name sane, flags consistent. **The fork's GDN call
construction is exonerated.** The hazard lives behind the op — the launch-time dependent
reads/allocations: the per-layer state the op resolves via `self.prefix` (mamba/GDN state
pool) and/or the L0 submit path.

**PREFILL-CHUNK SURFACE LAW (refines P80):** the lethal launch is a big GDN core call on
a **chunked-prefill chunk** (7880 tokens) against a deep-churned engine state — not a
decode/verify step. P80's "spec verify path" association was circumstantial (posture);
the verify traffic stretches deep-step wall and pool churn (exposure-time hazard law),
but the hung kernel is a prefill-shape core call. Nospec production safety unchanged.

**Post-reset L0 signature (new datum):** the first host-side L0 call after the silent
stall raised `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY (39)` — in
`spec_timing.py:96 → torch/xpu/streams.py record` (an event record) — while the op itself
raised nothing. P80's wedge showed pure DEVICE_LOST ×3, no OOM. Coherent reading: silent
driver-side allocation/submit failure at kernel submit (batch half-built, CS waits on a
memory-dependent semaphore = P80's descriptor-fetch stall, all compute idle) → reset →
subsequent L0 calls error (OOM vs DEVICE_LOST varies). Counter-datum kept honest: P80
wedged on a fresh boot's FIRST storm at T+25, so the ceiling pressure narrative cannot
be strictly uptime-accumulation — the race is per-storm deep-pool state.

## Consequence — fix surface (both in our fork)

1. **Launch-headroom guard** (engine-level): cap pools so deep-storm transient activation
   sets (~230 MB per 8k chunk) + driver submit temporaries always fit — testable by
   memory-pressure fields in the next probe.
2. **State-registry pinning**: per-layer GDN state buffers out of the churn pool
   (v130–v133: wholesale mapping death between turns) so the op's launch-time state reads
   can never race an invalidated mapping.

**P82 (next probe):** extend the snap with device memory pressure
(`memory_allocated/reserved` + free-device bytes — host-side, no stream dependency) at
each GDN call → the hung call's line then directly confirms/refutes the allocation-
ceiling mechanism; optionally also snapshot the state-registry buffers if Python-visible.

## Artifacts

- Host bank `/root/build/lce1/P81_wedge_123248/`: 3 rounds × (pyspy, ladder tail,
  **argsnap tail**, serve tail, dmesg), `container_root_snapshot/` (full 240k-line
  argsnap files + ladders), `serve_full.log`, `devcoredump_card1.bin` (512,130 B,
  md5 `cfb990c472cfb868a0ddb2f6006186a4`), worker_pids 1859/1865.
- Kit (this dir): `patch_p81_argsnap.py`, `patch_p80_event_ladder.py` (v2, site bugs
  fixed at source), `boot_p81.sh`, `p81_monitor.sh`. In-image md5s: `_p81_argsnap`
  5245ebc0, gdn_linear_attn (P80+P81) 684465d4, attention (v2) c4fe0bd9.
- Storm jsonl: 1,274 rows total (r1 +110, r2 +79, r3 +63-at-wedge; baselines
  `.pre-p80.bak` 865, `.pre-p81.bak` 969).
- Traps: monitor must be armed AFTER health 200 (boot-window false death); spec-decode
  `dump_input.py` fires at wedge teardown (expected noise, not a second error class).

## Post-round

Lane dead at capture (engine down 12:32:45, workers at-exit) → host reboot per
standing directive → chain auto-restores certified v1.2.29 nospec. Watchdog un-paused.
