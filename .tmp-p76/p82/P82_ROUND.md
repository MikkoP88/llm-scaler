# P82 — Device Memory-Pressure Freeze-Frame (CLOSED 2026-10-05)

**Goal.** Pick the P81 fix lever by direct measurement: snapshot host-side device
memory pressure (torch allocator + **driver view** via `mem_get_info`) on every
SYCL GDN core dispatch — the wedge surface. On wedge, the hung call's own line
either shows a squeezed device (allocation-ceiling → **launch-headroom guard**)
or a normal one (→ state-registry pinning).

## Instrument

`_p82_argsnap.py` (env `P82_ARGSNAP=1`): P81 argument snap +, on sycl-site
calls only, `mem[a=<torch alloc MB>,r=<torch reserved MB>,f=<driver free MB>,
t=<device total MB>]`. All queries host-side — zero stream commands, zero
device syncs (race not perturbed). `memquery_us=1508` measured once at open
(~2 % of the worker thread at storm sycl rates; P80 wedged under far heavier
per-call instrumentation). P80 v2 ladder re-applied first (one-pass, certified
md5s reproduced). P82 module md5 `4b879770`, gdn (P80+P82) `fa3e42ea`,
markers=4 at lines 1300/1301 + 1310/1311 (same sites as P81).

**Reference-frame law:** `a/r` are process-wide SUMS across both TP cards
(boot-steady 17.5/19.1 GB → wedge-time **66.5/70.9 GB** ≈ 33/35.5 GB per card,
card total 32,656 MB); `f/t` are the CURRENT device's driver view — `f` is the
per-card discriminator.

## Execution

Boot 13:11 (health 200, spec mtp×4, KV fingerprint 476,451 ✓). Monitor armed
post-health (P81 trap law). Depth-storm legs:

- r1 13:16 → **CLEAN `ok=108/108 wall=2207s`** (jsonl 1274→1384)
- r2 13:54 → **CLEAN `ok=108/108 wall=4057s`** (→1494)
- r3 15:07 → **WEDGE 15:55:35** (T+~48, 63/108 rows; →1557)

## Live mid-run discovery — the floor is brushed constantly

5 min into r1, healthy sycl calls already ran at `f=305/120 MB` (boot was
11.9 GB free; torch reserved grew 19.1→31.6 GB with the deep storm). Over the
round: **f=0 exact on 3,100 / 2,496 calls per worker; f<100 MB on
49,959 / 52,963 = 54–58 % of ALL sycl calls** (91,625 each worker). The engine
spends the majority of deep-storm life under 100 MB of driver headroom, and
most launches there still succeed — the ceiling alone is not sufficient; the
kill needs the confluence (big-chunk launch + submit-time allocation need +
no headroom + no racing free). This is the intermittence mechanism, measured.

## Capture — wedge anatomy (all times 2026-10-05)

| Time | Event |
|---|---|
| 15:51:28 | ccs engine reset guc_id=32 (card2) — **SURVIVED in limp**: workers keep executing queued steps; health dies |
| 15:53:59 | monitor fires wedge window, 3 banking rounds |
| 15:54:04–08 | 4× "exec queue reset detected" (limp cascade) |
| 15:55:35.305 | **both ranks lockstep: last sycl line = layer-53 `gdn_attention_core_xpu` launch, 7173-token chunked-prefill chunk** |
| 15:56:35 | ladder `STALL head=gdn_conv#263464 age=60.0 pend=[gdn_conv#263464, gdn_core#263465]` — next-layer rungs never complete |

Freeze-frames (seq 214018 both ranks, 1 ms apart):

- **TP0 (1689): `mem[a=66488,r=70946,f=38,t=32656]`** — last-60 f-series
  ends `…12,12,12,16,16,16,0,0,38,38`
- **TP1 (1695): `mem[a=66488,r=70946,f=65,t=32656]`** — ends `…451×19,
  111,95,95,65` (a 451→65 descent in ~1.3 s walking layers 49→53)

L0 post-reset flavor this round: `UR_RESULT_ERROR_DEVICE_LOST (20)` on TP1
(P81 saw OOM 39; both flavors are aftermath, not cause). py-spy r3 park frame
is INSIDE the GDN block: `tensor_model_parallel_all_reduce ← linear.forward ←
_gdn_core_and_output (gdn_linear_attn.py:1423)` — the output projection
waiting behind the hung core.

## VERDICT — allocation-ceiling mechanism CONFIRMED

The hung launch fired at the bottom of a rapid driver-free descent (38/65 MB
and falling ~300 MB/s) after the allocator had consumed the card
(reserved ≈ 35.5 GB/card vs 32.7 GB total → driver squeezed). Coherent chain:
**deep-storm allocator growth → driver-free floor (<100 MB for the majority
of calls) → occasional ccs reset (survivable limp) → a big GDN prefill-chunk
launch whose driver submit transient cannot be satisfied → silent submit
failure → CS waits on a memory-dependent semaphore → wedge/crash** (same
unified class as P73/P80/P81 — WEDGE == CRASH).

- P80's fresh-boot T+25 counter-datum RESOLVED: the allocator grows
  19→35 GB/card *within a single depth storm* — no uptime accumulation needed.
- P81's L0 OOM-at-event-record tell now reads as the driver-side allocation
  failure surfacing at the next L0 call.
- Lever 2 (state-registry pinning) not contradicted, but the wedge needs no
  mapping-invalidation story — the f-descent at the hung call is direct
  evidence for the ceiling.

## Fix surface — launch-headroom guard (P83 design)

1. **Cap allocator growth**: `torch.xpu.set_per_process_memory_fraction`
   (or PYTORCH_XPU allocator cap) at engine init so per-card reserved ≤
   total − headroom (e.g. 1–2 GB), forcing torch OOM- reclaim inside the
   allocator instead of starving the driver.
2. **Reserve a driver-side emergency pool** at boot (allocate-and-hold a
   headroom buffer outside the allocator; release/retry on submit-failure
   paths if reachable).
3. **Chunked-prefill admission control at depth**: bound concurrent deep
   prefill chunks (~230 MB transient per 8k chunk × concurrency) when
   `mem_get_info` free < floor — engine-level, no kernel change.

Production nospec note: the same machinery drives the nospec lane's allocator
under deep load; the guard is posture-general (spec only raises exposure
frequency — exposure-time hazard law stands).

## Artifacts

- Bank `/root/build/lce1/P82_wedge_155359/`: 3 monitor rounds (pyspy/ladder/
  argsnap/serve/dmesg tails), `container_root_snapshot/` full 214k-line
  argsnap files, `devcoredump_card2.bin` (512,663 B, md5
  `001b7858cd2e32bdc6700d915428ad8b`; dismiss EPERM after read = harmless),
  worker pids 1689/1695.
- Kit (this dir): `patch_p82_argsnap.py`, `boot_p82.sh`, `p82_monitor.sh`.
- Storm jsonl baselines: `.pre-p82.bak` 1274; wedge at 1557 (r1 +110,
  r2 +110, r3 +63). Legs: `p82_leg7_r1/r2/r3.out`.
- Traps: `a/r` are cross-card sums (never read per-card); thousands of
  healthy f=0 calls (don't treat zero as the kill signature — the kill is the
  sustained descent + big launch); monitor armed only after health 200.

## Post-round

Workers exited pre-reboot (tree=0); host rebooted with watchdog armed →
chain auto-restores certified v1.2.29 nospec production posture.
