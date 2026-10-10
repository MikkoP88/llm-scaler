# P80 — Verify-Path Wedge Probe: Event-Ladder Capture (CLOSED 2026-10-05)

**Goal.** Capture the MTP spec-verify wedge with a Python-side instrument that names the
earliest kernel boundary whose completion never arrives — the attribution the v79 py-spy
frames (both ranks parked at the rejection-sampler D2H) could not make, and the confirmation
(or refutation) of the R1 device capture's suspicion (`gdn_attention_core_xpu` never retiring).

**Vehicle.** v79a posture on the certified container: `boot_p80.sh` (clone of `boot_v79_x4.sh`)
= `DFLASH2_EMIT_K=1` + v79a drafter loop-skip (host-bind proposer) + `P80_TRACE=1`,
spec `{"method":"mtp","num_speculative_tokens":4}`, TP2, gmu 0.8, FULL_DECODE_ONLY graphs.
Load: `wsc_pressure_v134_leg7_depth.py` (N_TURNS=6 depth storm, 108 requests).

## Instrument — the event ladder

One `torch.xpu.Event()` recorded per instrumented rung on the current stream; a daemon
thread (250 ms tick) completes events head-first and heartbeats to a per-pid line-buffered
file (`/root/p80_ladder.<pid>`). If the head event's age exceeds `P80_STALL_SECS` (60), one
`STALL head=<name>#<seq> age=… pend=[…]` line names the earliest unretired kernel boundary
plus what is queued behind it.

Rungs (all exception-swallowing, capture-guarded so cudagraph capture stays clean):

| Rung | Site |
|---|---|
| `fwd` | bind `gpu_model_runner.py` `sample_tokens`, after ephemeral-state unpack (step boundary) |
| `d2h` | `rejection_sampler.py` `parse_output`, before `output_token_ids.cpu().numpy()` |
| `gdn_core` | `gdn_linear_attn.py`, immediately before `torch.ops.vllm.gdn_attention_core_xpu(` |
| `gdn_conv` | `gdn_linear_attn.py`, before the ESIMD conv+core decode attempt |
| `gdn_chunk` | `gdn_linear_attn.py` `fi_chunk_gated_delta_rule`, before the flashinfer chunk launch |
| `fattn` | `attention.py` `unified_attention_with_output`, after layer-name resolve |

Ladder files are separate from `serve_full.log` and line-buffered with explicit flush —
immune to the SPECTIMING BUFFERED-FLUSH law (readable mid-run; the monitor's minute
heartbeats tracked them live through the whole storm).

`p80_monitor.sh` (host, detached): 15 s health loop; on health loss with workers alive =
WEDGE WINDOW → banks 3 rounds 45 s apart (py-spy dumps of both workers via comm-scan
`VLLM::Worker*`, `docker cp` of `/root/.`, ladder tails, serve tail, dmesg).

## Execution

- 07:40 host reboot → chain auto-restore (v1.2.29 nospec) → certified (digest bit-exact,
  0 v79 markers, nospec posture, clean dmesg).
- Watchdog re-paused; graceful TERM of the certified serve (tree exit verified);
  P80 patchers applied (`image` in-container, `bind` on the host-mounted runner file —
  all `py_compile`-certified); `boot_p80.sh` → health 200 ~08:13.
- 08:21 leg7 depth storm launched; monitor armed. Steady ladder flow both ranks
  (~200–230 lines/min/worker), zero stalls through the whole shallow band.
- SPECTIMING arc: `step_wall 682 ms` (T+8) → `1122 ms` (T+12) → `1213 ms` (T+16) →
  `2671 ms` (T+24; tforward h≈2.07 s / d≈2.28 s — the deep tail IS the target-verify
  forward, as the v79 DEEP-STEP ECONOMICS law predicted).

## Capture

```
08:46:16.6  BOTH ranks: last completed event = gdn_core#107599, pend=0   ← last healthy moment
            (gdn_core rung records immediately BEFORE the gdn_attention_core_xpu launch)
08:46:16.6 → 08:47:29   72 s of zero completions on both ranks
08:47:28   xe ccs engine reset, guc_id=22, devcoredump created   ← driver LR-cleanup RESPONDS
08:47:29   reset force-completes the 6 queued events (HB burst done 107599→107605), then
           STALL head=fattn#107606 age=72.4 pend=[fattn#107606, 107607, 107608]
08:47:29   EngineCore sample_tokens → collective_rpc timeout → shutdown begins
08:47:33   health 000; monitor fires, banks 3 rounds (workers still alive)
```

py-spy at capture time: TP0 in the shutdown path stuck at
`_cleanup_profiling_kv_cache → synchronize`; TP1 parked at
`_to_list ← _bookkeeping_sync ← sample_tokens` (the D2H symptom); TP1's tracer thread
stuck **inside `Event.query()`** holding the GIL — the broken-context signature, and the
reason TP1's ladder file stops mid-silence while TP0's kept its (post-reset) final lines.

## Verdict — GDN CORE KERNEL LAW

**The wedge is `torch.ops.vllm.gdn_attention_core_xpu` non-retiring at deep context**
(step_wall ≈2.7 s band, 30k+ computed). The last event that ever completed on both ranks
is `gdn_core#107599` — recorded immediately before that op's launch; everything behind it
jammed (`fattn` head + `fattn×3` pending = downstream full-attention entries enqueued before
the Python threads parked). The rejection-sampler/`.cpu()` parks are traffic-jam symptoms,
not the hang. This independently confirms the R1 device capture (hung batch VA) from the
Python side.

**The guc ccs reset is the RESPONSE, not the cause** — the completion silence starts 72 s
before the reset; the reset force-completed the queued events and left the stream
permanently broken (`fattn#107606` never completes even post-reset).

Fix surface is now single-target: the GDN core kernel's deep-state path. Drafter (exonerated
v79a), rejection sampler, and full attention are downstream/clean.

**Open caveat:** the v79b scheduler trim was wiped by the chain's container recreate and not
re-applied, so this leg may have run full ×4 width (mid-park reset at +72 s is
crash-class-flavored vs the v79 wedge legs' clean mid-storm). The verdict is
width-independent — the wedge fires at both widths — but the next wedge-repro leg should
re-apply the trim to separate crash-class response from pure wedge death.

## Devcoredump analysis — WEDGE == CRASH UNIFICATION LAW

Parsed `devcoredump_card1.bin` (ASCII) against the P73 crash-class capture
(`WD_crash_132400_card2.devcoredump`):

| Field | P73 crash (card2, ccs32) | P80 wedge (card1, ccs22) |
|---|---|---|
| Reason | LR job cleanup, guc_id=32 | LR job cleanup, guc_id=22 |
| Job | seqno=2509, finished=0 | seqno=927, finished=0 |
| IPEHR | 0x61050001 | **0x61050001** |
| RING_INSTDONE | 0xffdefffe | **0xffdefffe** |
| SAMPLER_INSTDONE[0..31] | 0xffffffff | 0xffffffff |
| ROW_INSTDONE[0..31] | 0x00000000 | 0x00000000 |
| SC_INSTDONE[0..31] | 0xffffffff | 0xffffffff |
| INSTDONE_GEOM_SVGUNIT[0..31] | 0xffdfffff | 0xffdfffff |
| DMA_FADD − ACTHD | 0x35C | 0x3FC |
| Owner | python3 [8290] | python3 [15706] |

**Every meaningful field identical.** The CRASHFIX-76 crash class and the GDN wedge are
ONE hardware event: the same batch stall at the same instruction phase (same IPEHR), the
same stalled-unit signature, the same ~0x35C–0x3FC walk into the batch buffer. The only
presentation difference is what happens AFTER the LR-cleanup reset — lone reset (wedge/limp)
vs bcs job timing out behind it → GT reset → DEVICE_LOST (crash death, per the P72
SURVIVED-RESET LAW).

**Mechanistic reading:** all 32 DSS shader cores AND samplers report done/idle
(`0xffffffff`) while the engine-level INSTDONE has clear bits — the command streamer is
stalled at a CS-level dependency with zero compute in flight. The parser had walked only
~0x35C–0x3FC bytes of the batch (kernel setup/descriptor-fetch phase, before/at first
dependent reads). This is NOT a spinning kernel — it is a launch-boundary memory-dependency
stall, deterministic enough to hit the same IPEHR every time. Combined with the ladder
verdict (stall begins at the `gdn_attention_core_xpu` launch boundary at 30k+ computed),
the suspect surface is that kernel's early dependent reads — state/binding/descriptor
buffers at deep context (race with invalidation, or a fence whose write never landed).

Caveat: the dump's VM-state section captured `error: -19` (no page tables), and the owning
process is gone (container recreated), so the batch VA cannot be resolved to a buffer
post-mortem; kernel identity rests on the ladder, which is sufficient. GuC fw 70.72.1
(wanted 70.49.4) — chronic mismatch, unchanged.

**Consequence for the program:** crash-class fix and wedge fix are the SAME fix. Nospec
production posture remains correct. The root-fix surface is the deep-context launch path of
`gdn_attention_core_xpu` (argument/state buffers), not the drafter, sampler, or scheduler.

## Artifacts

- Host bank: `/root/build/lce1/P80_wedge_084733/` — 3 rounds × (pyspy, ladder tail, serve
  tail, dmesg), `container_root_snapshot/` (full 275 KB ladder files), `worker_pids.txt`
  (2122/2128), `serve_full.log` (438 KB, full boot+storm+death), `devcoredump_card1.bin`
  (512,554 B, md5 `7d0ba865772ab62687a8f9afd284f9fb`, harvested from sysfs card1 within the
  ~6-min window — P73-style hung-batch-VA analysis now possible on a wedge-caught dump).
- Patcher/scripts (this dir): `patch_p80_event_ladder.py` (modes `bind`|`image`),
  `repair_p80_rejsampler.py`, `patch_p80_fattn.py`, `boot_p80.sh`, `p80_monitor.sh`.
- Patched md5s: `gdn_linear_attn.py` 22a0ac3a…, `rejection_sampler.py` 01d20b79…,
  `attention.py` 1b5c6ad8…, `_p80_tracer.py` dc166b8d…, bind `gpu_model_runner.py` 62804f9f…
- Storm jsonl: `wsc_pressure_v134_leg7.jsonl` 969 rows (P80 leg appended rows 866–969;
  pre-P80 baseline saved as `.pre-p80.bak` at 865 rows).

## Post-round

Lane was left dead (workers zombie-parked) → host rebooted, watchdog un-paused; certified
chain auto-restores v1.2.29 nospec production posture. Next candidates: devcoredump VA
analysis (hung batch VA vs P73's), GDN core kernel deep-state review, R2 (async drafter
host glue, §11.7 bar propose ≤7 ms / greedy ≤1.0 ms).
