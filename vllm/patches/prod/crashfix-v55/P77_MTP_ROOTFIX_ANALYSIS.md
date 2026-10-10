# P77 — MTP Deep-Analysis: Root Fixes for the Spec Crash Class, Stalls, and Perf Deficit

**Date:** 2026-10-02 · **Directive (user, binding):** *"do not use any barrier, only
root fix is accepted, all wedge has to fix and stalls"* · **Standing laws active:**
ZERO-DEGRADATION (only improvements), reboot-after-crash, SHIP-MEASUREMENT (.so sha
`1d9dcf4e…` unless a kernel ships), no subagents, spec MTP + XGrammar-2 stay SUPPORTED.

**Lane state while writing:** v1.2.29, spec-opt-in posture live with MTP×3 +
chat template (container `0b6a2687b621`, KV 481,725 = spec pool, `VLLM_XPU_SPEC_DRAFT_BARRIER=0`
per directive — never reintroduced), sync scheduler, health 200, zero resets.

---

## 1. What this analysis adds over CRASHFIX-76

CRASHFIX-76 closed the *posture* question (nospec default, spec opt-in) and banked the
trigger discrimination (spec 3/3 death at depth, nospec 5/5 clean, exposure-time scaled).
It explicitly did **not** name the defect inside the MTP path. This round re-opens the
forensics with four new evidence classes:

1. **Full barrier-history reconciliation** (repo archaeology, §2) — every historical
   justification for `VLLM_XPU_SPEC_DRAFT_BARRIER` is void; it is not and never was a
   root fix. Consistent with the user directive: it stays `0` forever; no barrier appears
   anywhere in this program.
2. **The v52f stall law (user-observed, live-confirmed)** — stalls correlate 1:1 with
   `mamba_utils.py` PRE-COPY / DEF-PP lines. Code anatomy (§3) names the defect: a
   serial 1 KB-chunk triton copy kernel moving multi-MB GDN states **on the critical
   path of every step**, decode *and* chunked prefill.
3. **Live MTP×3 acceptance telemetry** (§5) — drafter quality is healthy (58-70% draft
   acceptance); the 3× perf deficit is machinery, not the model.
4. **Upstream corroboration** of the exact crash signature on identical hardware
   (immich #23462: BMG, python, `Reason: LR job cleanup`, fw-mismatch against the same
   kernel-wanted 70.49.4) — the class is not fork-specific (§4.3).

---

## 2. Barrier history — closed, permanently

Timeline (all banked in-repo; sources: `wedgefix-v60/README.md`,
`image-bake-keepers-v37/README.md`, `wedgefix-v75/PATCH_STACK_ANALYSIS.md`,
`perf-v89/DECODE_SPEED_ANALYSIS.md`, `perf-v123/patch_barrier_v123_bake.py`):

| Era | Event | Verdict on the barrier |
|---|---|---|
| v2x (Aug) | #11 oneCCL wedge era; device-drain barrier added, default ON | mitigation only |
| v31/v31.1 | **#11 root cause convicted: inductor piecewise × spec ≥32k** — fixed BY CONFIG (`TORCH_COMPILE_DISABLE=1` for spec+TP2, still load-bearing today) | original justification VOID |
| v33/v37 | barrier measured strictly better OFF (+13.7…+28.5% vs the drain); default flipped OFF | OFF keeper (short-ctx certified) |
| v60 (Sep 21) | CC-intensive deep usage re-opened wedge symptoms; drain re-enabled every-step + async forced OFF | confounded — see v88 |
| v62 (Sep 22) | host-flock mode 2 (µs-class); labeled "ROOT FIX" | attribution later superseded |
| v75–v88 | **real root cause found: int32 pool-offset overflow in GDN conv kernels (ids ≥4097)**; fixed by int64 wheel v1.2.21. Barrier proven inert against this class (v1.2.20 BARRIER=2 died deterministically; v61 crashed with drain ON) | never protected against the dominant class |
| v123 (Sep 28) | default OFF (user directive; T2b measured identical perf) | OFF posture — matches directive |
| CRASHFIX-76 (Oct 2) | spec depth deaths 3/3 **on barrier=0**; nospec clean | barrier irrelevant to this class too |

**Conclusion (directive-compliant):** `VLLM_XPU_SPEC_DRAFT_BARRIER=0` stands in every
posture, and no step of this program may reintroduce host- or device-side serialization
of the draft path. Everything below is defect removal, not scheduling mitigation.

---

## 3. The stall class — ROOT-CAUSED (code anatomy)

### 3.1 The user law

> "stalls happened always when this appear on log → `v52f PRE-COPY` / `v52f DEF-PP`"

Live confirmation on the MTP×3 lane (13 min of CC traffic): **3,194 PRE-COPY + 134
DEF-PP lines (~4.1/s)**, both TP ranks logging each event twice. Log lines themselves are
not the cost — they mark the copy event.

Full-session census (`p77_stallscan.py`, 15:12→20:19, `.tmp-p76/`): PRE-COPY 3,276,
DEF-PP CROSS (src≠dest) 54, DEF-PP INPLACE (src==dest, bias>0) 94, **DEF-PP NOOP
(src==dest, bias=0) 64**. One precision on the user's flagged sample
(`DEF-PP … src=6 dest=6 bias=0`): `collect_mamba_copy_meta` opens with
`if src_block_idx == dest_block_idx and accept_token_bias == 0: return` — that exact
class is **already skipped** (no pointer entries, no kernel launch); the v52f log line
prints before the guard fires. The stall witnessed by such lines is therefore NOT the
self-copy itself — it is whatever shares that step: the blocking
`num_accepted_tokens_event` wait that precedes DEF-PP consumption (see §3.2), the
PRE-COPY real carries that fire at the same boundary steps, and the chunked-prefill
entry itself (7k-token chunks pausing decode interleave — the line class is the
boundary-step marker).

### 3.2 The machinery (all paths in-tree vLLM code, NOT custom ESIMD)

- `vllm/v1/worker/mamba_utils.py` — `preprocess_mamba()` (log site :221, the PRE-COPY):
  for every request whose running-state block index moved (every `block_size=1024`
  tokens of progress — including **each chunk of a chunked prefill**: live proof
  `computed=7168 sched=1016` → `computed=8184 sched=276` one second apart), collects
  one copy entry per (GDN layer × state kind) and issues them.
- `run_deferred_postprocess()` (log site :345, the DEF-PP): partial→full block
  conversion copy, runs at **execute_model entry** (v32 deferral).
- Both terminate in `do_mamba_copy_block()` → `batch_memcpy()` →
  **`batch_memcpy_kernel` (triton)**:

```python
@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)              # ONE program per copy entry
    ...
    for i in range(0, size, BLOCK_SIZE):  # serial 1 KB chunks
        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)
```

**Defect:** one triton program (a single CU) serially copies an entire GDN state —
conv + SSM state across all GDN layers of a 27B hybrid — in 1024-byte loop iterations.
A multi-MB state is chewed at 1 KB/iter by one thread block while the step's forwards
wait: `preprocess_mamba` is called in the model runner before every forward
(`gpu_model_runner.py:4661`), on the default stream, and under MTP×3/×4 the draft
forwards + verify queue **behind** it. Under a 20-60 request storm with continuous
boundary crossings (decode *and* prefill chunks), this is a self-inflicted copy storm
on the critical path.

Secondary costs in the same path: per-event `copy_to_gpu()` H2D of the pointer tables,
and — at every step that carries deferred rows — the **blocking host wait**
`_v55_wait_event(num_accepted_tokens_event)` at `execute_model` entry
(`gpu_model_runner.py:4529-4541`), which releases only after the previous step's
device work + D2H retire. The DEF-PP log line prints immediately after that wait
releases: the line is a *witness* of the sync boundary, which is why stalls and the
line co-occur even when the copy itself is the no-op class.

### 3.3 What each copy actually moves (spec semantics, `layers/mamba/mamba_utils.py`)

- `get_conv_copy_spec`: `state[src_block, offset:]` — essentially the **full conv slot**
  of the source block (the accepted offset only trims leading rows).
- `get_temporal_copy_spec`: `state[block_ids[cur + num_accepted − 1]]` — the **whole
  SSM slot of a bias-selected block**. With `num_accepted − 1` pointing past the blocks
  actually written by the GDN kernel (the spec-block region right after the running
  block), this reads garbage — **the v52e NaN-poison law, confirmed at spec level**.
  The structural clamp is therefore precisely: clamp `cur + bias` to the last written
  block.

Each real event = one entry per (GDN layer × {conv, ssm}) ≈ 96 entries × ~917 KB/slot
(per the banked hybrid anatomy) ≈ **tens of MB per event per rank**, each entry chewed
by ONE triton program in serial 1 KB iterations. In-storm rate ~4 events/s per rank.

### 3.4 Measurement limits (log-scan verdict, `p77_stallscan.py`)

Inter-line timestamp analysis of the full session CANNOT convict a class at 1 s
resolution: the ≥2 s gap windows are almost all **inter-request idle** (logger shows
`Running: 0 reqs` before the gap; the next line is the new request's first PRE-COPY —
hence the 841/946 PRE-COPY adjacency is a base-rate artifact), and in-serving cadence
is bounded by the 10 s logger tick (per-class next-line p90 = 2-6 s for ALL classes).
The per-step device tax lives below log resolution → the decisive instrument is a
timing probe (§6, FIX-0), not more log forensics.

---

## 4. The crash class — unified hypothesis, ranked

Banked signature (two full devcoredumps + B1 leg, identical): ccs engine reset, reason
**"LR job cleanup"** (GuC Long-Running job cleanup), **ACTHD == RING_BBADDR** (batch
START never retires), **0x54 B unretired**, job timeout **infinite**, RING_INSTDONE
`0xffdefffe`, cross-card / cross-guc_id / cross-kernel-set, spec-only at deep context,
exposure-time scaled; nospec 5/5 clean at the same shape.

### 4.1 H-R2a (primary): the mamba copy storm IS the long-running-job class

- The `batch_memcpy_kernel` batches are, by construction, the **longest-running ccs
  jobs this stack submits** (single-CU serial copies of multi-MB states) and they are
  submitted at high cadence from two sites per step under spec.
- "LR job cleanup" + infinite timeout is GuC handling of a long-running context whose
  job must be cleaned up; the hang at **batch START with a tiny unretired tail** fits a
  copy batch that was queued behind/alongside deep-forward work and never got its first
  dispatch retired.
- **Cross-kernel-set invariance explained:** the B1 leg (all 15 `DISABLE_ESIMD_*`)
  swapped the custom kernels for upstream — but `batch_memcpy_kernel` is in-tree triton,
  untouched by those knobs → same-class crash.
- **Spec-only + exposure-time explained:** spec multiplies the machinery: 5 forwards
  per step (4 draft + 1 verify at ×4), `num_speculative_blocks` extra state blocks with
  bias-shifted copies, DEF-PP deferred copies each step, and 2-3× longer walls at the
  death shape → more copy batches, longer ones, more LR-cleanup events per deep-second.
  Nospec: 1 forward/step, no spec blocks, no bias reads → 5/5 clean.
- **FW-independence explained:** both 70.44 and 70.72.1 exhibit it (banked law) — a
  KMD policy interacting with our job profile, not a fw regression.

### 4.2 H-R2b (competing): upstream deep-forward kernel non-retirement

P73's original read (attention/GDN deep-forward kernel stops retiring; fr traces "never
mid-AR/mid-propose"). Still viable — the fr traces were host-side stacks, and the dump
carries no kernel names (GUC-MODE HWSP LAW), so the hung batch was never actually named.
**Discriminator:** FIX-A (§6) at the death shape. If the crash class survives with the
copy storm eliminated → H-R2b convicted and the surface is upstream xe/kernel
(§6 FIX-C + the existing upstream handoff). If clean → H-R2a convicted and the root fix
already shipped.

### 4.3 Upstream corroboration

immich-app/immich#23462: same `Reason: LR job cleanup` coredump class, Intel
**Battlemage**, python ML workload, `GuC firmware 70.65.0 (kernel wanted 70.49.4)` —
same kernel-wanted version as ours (loaded 70.72.1). The class exists outside this fork
on the same GPU family; our workload is simply an efficient trigger. Any driver-side
resolution there applies to us (watch item).

---

## 5. The perf deficit — decomposed (why spec ran ~3× slower)

Live MTP×3 telemetry (20:12-20:13): mean acceptance length 2.75-3.11 / 4, per-position
0.80/0.67/0.58, draft acceptance 58-70% — **drafter quality is NOT the deficit.**
Decomposition, ranked by measured evidence:

1. **Sync-scheduler host chain** — the post-verify handoff (parse → IPC → scheduler →
   launch) serializes every step on this lane (async forced OFF by WEDGEFIX-A;
   `VLLM_XPU_ALLOW_ASYNC` absent). v89 measured async = **+85-110% solo, ~2× per-stream
   under contention** on the spec path, battery-clean after the #29 root fix
   (`VLLM_RPC_TIMEOUT=60000`, standing in our env). The biggest idle lever.
2. **Copy-storm stall mass** (§3) — fires in decode and prefill; at storm depth it is
   continuous. Directly reduces both throughput and tail latency; also the crash driver
   under H-R2a.
3. **CHURN-DESTRUCTION** (banked law) — spec draft-setup allocation pops (new=5-6/step)
   guillotine cached prefix mappings → prefill recompute inflates storm walls. Structural
   levers named at v133 close-out: two-free-list redesign, size-aware eviction
   (weeks-scale).
4. **Intrinsic draft overhead** — small at this acceptance; NOT a target.

---

## 6. Root-fix ladder (no barriers, ranked by cost/evidence)

### FIX-0 — timing instrumentation FIRST (decisive, zero-risk)
Wrap `do_mamba_copy_block()` (device time via events + host wall) and the
`num_accepted_tokens_event` wait with per-class ms accounting, p50/p99 rolled per
minute into the log. Splits the user-observed stall into its three candidate components
(copy-kernel time / sync-wait time / prefill-interleave) BEFORE any code change, and
provides the FIX-A before/after gate. Pure telemetry, no semantics change.

### FIX-A — parallel `batch_memcpy` kernel (THE stall root-fix; crash discriminator)
Rewrite the triton kernel: 2-D grid — one program per 16 KB chunk (or vectorized 16 B
accesses at 1 KB granularity), preserving exact copy semantics; keep the same buffers
API. ~10-line kernel change, in-tree python, no .so impact (triton JIT), no numerics
change (pure memcpy). Optional second step (only if needed after measurement): issue the
copy on a side stream with an event before the first forward — NOT a barrier; it is
removing our own serialization of an internal dependency.
**Gates:** ZERO-DEGRADATION storm parity-or-better (vs nospec band 330/372/277 s and
spec band 1089/1029/1042 s), stall-mass metric (PRE-COPY-adjacent stall windows → ~0),
depth-shape ×3 (class-kill under H-R2a; discriminator for H-R2b).

### FIX-B — copy-volume reduction + structural bias clamp
- The `src==dest && bias==0` skip **already exists** (`collect_mamba_copy_meta` entry
  guard) — verified live; no work needed there.
- **Prefill-chunk coalescing:** PRE-COPY fires at every 1024-token crossing of every
  chunked-prefill chunk (live: same request `computed=7168 sched=1016` then
  `computed=8184 sched=276` one second apart = two full-state carries back-to-back);
  coalesce so one carry serves the completed prefill.
- **Structural bias clamp:** clamp the temporal read `cur + bias` (and the conv
  offset) to the last block actually written by the GDN kernel — kills the v52e
  NaN-poison family by construction instead of by detection (v52f history ring stays
  as telemetry).

### FIX-C — driver hygiene (cheap, evidence-gated, no lane risk)
- Align GuC fw to kernel-wanted **70.49.4** (backup exists: `.zst.bak-7044`; the
  mismatch is unsupported territory and the upstream report shares it).
- Evaluate `xe` job-timeout/heartbeat parameters so a wedged job becomes a recoverable
  error instead of DEVICE_LOST — *failure-containment* for whatever class remains after
  FIX-A, not a fix by itself; adopt only with parity evidence.
- Track kernel 6.18.x xe fixes (job-cleanup/fence fixes landing upstream); upgrade via
  the standing host-reboot discipline when a candidate is justified.

### FIX-D — async-scheduling adoption (perf, after crash closure)
One controlled leg: `VLLM_XPU_ALLOW_ASYNC=1` + standing `VLLM_RPC_TIMEOUT=60000` on the
FIX-A posture → full C1/C2/C3 matrix. v89's battery + #29 root fix are the precedent;
the WEDGEFIX-A research-only invariant is superseded *only* by passing this matrix
(ZERO-DEGRADATION, no crash). Expected: recovers a large fraction of the host-chain tax.

### FIX-E — structural churn work (weeks-scale, last)
Two-free-list / size-aware eviction per the v133 close-out. Only after FIX-A/B/D
re-measure the remaining gap.

---

## 7. Validation matrix (per fix or stacked posture; fresh host per crash)

| Gate | Criterion |
|---|---|
| C1 standard ×3 | 72/72 each; wall ≤ nospec band (330/372/277 s) — improvements only |
| C2 depth ×3 | 108/108 each, zero resets/harvests — the class-kill gate |
| C3 regression | spec + XGrammar-2 + template asserts, health 200, KV fingerprint |
| C4 stall-mass | stall windows adjacent to PRE-COPY/DEF-PP events → ~0 (new telemetry: log per-step copy bytes + step gap) |
| C5 ship | bake per SHIP-MEASUREMENT law; chain rebase both copies; watcher rearm |

---

## 8. Open questions / next actions

1. **FIX-0 timing probe** on the live lane (copy-kernel ms / wait-event ms / step
   boundary) — first move of the next round; produces the C4 baseline.
2. FIX-A implementation + C1/C2 legs (decisive for H-R2a vs H-R2b) — the next round.
3. If C2 still dies with stalls gone: live GPU-VA mapping leg (A4-style) to name the
   hung batch's allocation; H-R2b surface → expand the upstream handoff with FIX-A
   exclusion evidence.
4. Watch immich #23462 / intel-xe list for the driver-side resolution of the
   LR-job-cleanup class (same hw family).
5. Measure: per-step copy bytes histogram (sizes_np sum) — quantifies the storm for the
   FIX-A before/after gate (C4).

---

## 9. RESULTS (2026-10-02 ~21:00) — v52k rewrite SHIPPED to the live lane

### 9.1 What shipped (in-container swap, `.v52kbak` rollback at site)

`vllm/v1/worker/mamba_utils.py` (site md5 `02c0725b…`, backup `27fa93ba…` = pristine
pull). Four change sites, diff-verified byte-identical elsewhere:

1. **`batch_memcpy_kernel` — 2-D chunked grid** (the stall root-fix): axis0 = copy
   entry, axis1 = 8 KB chunk; one masked load/store per program. Replaces the
   single-program serial 1 KB loop.
2. **`batch_memcpy(src_ptrs, dst_ptrs, sizes, max_size=None)`** — host-side max size
   (from `copy_bufs.sizes.np[:n].max()`, no D2H sync) computes
   `grid = (n, cdiv(max_size, 8192))`; `max_size=None` fallback (one tiny `.max().item()`)
   keeps the dormant external `cuda_mem_ops` importer working; `batch==0` early return.
3. **Structural bias containment** in `collect_mamba_copy_meta` (single choke point —
   all three call sites funnel through it): per-group
   `group_bias = clamp(accept_token_bias, 0, len(block_ids)-1-src_block_idx)` — the
   temporal read `block_ids[src+bias]` can now only select blocks inside the request's
   own allocation → v52e NaN-poison family dead by construction; v52f detector stays
   as telemetry.
4. **`VLLM_V52K_TIMER` env-gated device timing** (FIX-0 folded in): XPU event pairs per
   copy event, p50/p99/max + effective GB/s rolled every 64 events into the log;
   fully exception-guarded (degrades to silent-off, never touches the copy path).

All v52f log lines, the `src==dest && bias==0` no-op guard, and the module API are
preserved unchanged.

### 9.2 Microbenchmark (in-container, XPU, under live serving contention)

`p77_memcpy_bench.py` — workload = GDN slot size 917,504 B × 90(+small/odd/edge) entries;
correctness: byte-exact vs source incl. odd sizes, size=1, exact-block multiples, no
past-end writes — **PASS ×3** (old sanity / new host-max / new None-fallback).

| | OLD serial 1 KB | NEW chunked 8 KB 2-D |
|---|---|---|
| p50 per copy event | **18.2 ms** | **0.49 ms** |
| max observed | **314.5 ms** | 20.9 ms |
| effective BW | 1.31 GB/s | 49.2 GB/s |
| **speedup** | | **×37.45** |

The old kernel's 314 ms max IS the user-observed stall class caught red-handed: one
copy batch could freeze the engine for a third of a second; at storm cadence (~4
events/s/rank) the pre-forward critical path was saturated by copy time.

### 9.3 Apply sequence + two NEW operational laws

Apply hit a boot crash → traced, NOT v52k (kernel ran clean 3×35 reps minutes before;
failure preceded any request traffic):

- **KILL-DIRTY LAW:** `pkill -9` on a long-running spec engine leaves the device dirty
  (`dmesg`: reset storm, `Timedout job … guc_id=51 … in no process [-1]` = orphaned
  contexts; VRAM never released) → next boot fails `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`
  (39) → cascade `UR_RESULT_ERROR_DEVICE_LOST` (20) + "shutdown drain did not finish".
  `pgrep`-empty is necessary but NOT sufficient: after killing a spec engine that ran
  deep traffic, **reboot the host before relaunching** (the standing directive, now
  understood to cover kills, not just crashes).
- **WATCHDOG-PAUSE-EVAPORATION LAW:** `systemctl stop lane-watchdog` is runtime-only —
  a host reboot re-activates the enabled unit, and its restore path `docker rm -f`s the
  lane container, **destroying in-container patches** (RIDER law surface). After any
  reboot, re-stop the watchdog before its ~10-min grace elapses. Correct revival of a
  patched container: `docker start lsv-test` (FS survives stop/start; only rm destroys),
  then `docker exec -d … bash /root/serve_user.sh`.

Post-reboot posture (all asserts green): health 200, MTP×3 spec active, KV 481,725 =
spec-pool fingerprint, template detected, `VLLM_V52K_TIMER=1` confirmed in the serving
process env (pid 53), `VLLM_XPU_SPEC_DRAFT_BARRIER=0` per directive, watchdog resumed,
dmesg clean, live traffic serving with **acceptance 2.47 / 48 tok/s** within minutes.

### 9.4 Gate status

- COPY-TIMER windows roll after 64 copy events (needs storm-scale load; light traffic
  ≈ 0.05 events/s → watch under the next CC session).
- C1 (standard ×3 parity-or-better) / C2 (depth class-kill, H-R2a vs H-R2b
  discriminator): pending lane availability — the decisive legs.
- Bake v1.2.30 + chain rebase: only after C1/C2 pass; until then the fix is
  in-container only (any watchdog restore wipes it — see 9.3 law).
- FIX-B prefill-chunk coalescing: DROPPED (running state must live at the tail before
  each forward — layout constraint; the ×37 kernel makes each carry ~0.5 ms instead).
- FIX-C: OFF PROGRAM per user directive ("issue is not on driver or firmwares").

## 10. Round-2 live measurement (2026-10-02 21:05-21:30, instrumented boot)

Engine: graceful-SIGTERM stop of the 20:14-era engine → instrumented relaunch
(`VLLM_SPEC_TIMING=1`, `VLLM_V52K_TIMER=1`, `VLLM_XPU_SPEC_DRAFT_BARRIER=0`),
watchdog paused for the surgery window. User drove live traffic (their client
10.100.8.6): deep request to 57k+ computed, 3-4 concurrent reqs, acceptance 2.3-2.9.

### 10.1 Step anatomy measured (SPECTIMING, MTP×3, TP2, steady graphed decode)

| Regime | step_wall | tforward h / d | propose h / d | notes |
|---|---|---|---|---|
| pure decode, ~5k depth, 1-3 reqs | 72-82 ms | 1.6-3.5 / 47-56 | 16.5 / 11.3 | h≈2 ⇒ graph replay |
| pure decode, 57k depth, 4 reqs (16 rows) | 89.1 ms | 2.5 / 59.4 | 18.0 / 12.0 | depth adds ~5-10 ms only |
| mixed prefill+decode (7168-chunk + verify rows) | 146.2 ms | **83.1 / 118.3** | 17.5 / 12.3 | h=83 ⇒ EAGER backbone |

Remaining segments (per step): tlogits h=0.8 d=2.7 · greedy ×3 h=2.3 d=2.2 each ·
dforward h=4.4 d=2.6 · dforward_loop h=7.2 (2 calls) d=1.2 · dmeta h=0.4.
Uncovered host glue ≈ 25-30 ms/step (scheduler/prepare/sampler/bookkeeping/RPC —
sync-scheduler serialization).

### 10.2 Round-2 laws

- **VERIFY-GRAPHED FLAT LAW:** `tforward h≈1.7-2.6 ms` is a graph *launch*, not an
  eager enqueue (eager backbone = h 83). The 47-65 ms d-span is genuine graphed
  device execution and scales only mildly with depth (5k→57k: +5-10 ms). The
  `uniform_decode_query_len = 1 + num_spec_tokens = 4` classification works: steady
  spec verify IS graph-replayed. The eager-verify hypothesis is dead.
- **MIXED-EAGER LAW:** whenever a chunked-prefill chunk (sched=7168) shares the step
  with verify rows, `_is_uniform_decode` fails → the whole 27B backbone runs EAGER
  (h=83, d=118, step 146 ms) for the prefill's duration. This is exactly when the
  user's PRE-COPY/DEF-PP fingerprint appears. Merging is still throughput-optimal vs
  splitting (split = prefill-eager step + separate graphed verify step ≈ 199 ms for
  the same tokens vs 146 merged); verdict: keep the merge, document the cost.
- **MIXED-WIDTH TAIL:** isolated `sched=2` rows observed (per-request width clamp /
  near-finish requests) also break uniformity → 1 eager step at request tails. Minor.
- **DRAFTER-HOST LAW:** propose h=17-18 ms vs d=12 ms — the MTP head is HOST-bound
  (3 eager forwards ≈ 11.6 ms dispatch + 3 greedy samples ≈ 6.9 ms + metadata ~1 ms),
  ~6 ms of device idle inside the span. This is pure MTP code, the rewrite's own turf.
- **CRASH-SURFACE LAW (from the 20:32 user crash):** the old engine's DEVICE_LOST
  surfaced at `sample_tokens → _bookkeeping_sync → RejectionSampler.parse_output →
  output_token_ids.cpu()` — the per-step sync-scheduler D2H, i.e. exactly the path
  WEDGEFIX-A forces (CONVICTION #1). The wedge forms in the deep spec forward
  (CRASHFIX-76), but the sync bookkeeping is the observation surface AND ~30 ms/step
  of glue. Watchdog auto-restore mid-drama booted a stripped engine (rider law),
  host reboot 20:42; the instrumented boot then ran clean through 57k spec (fits the
  intermittency law — NOT claimed as a fix).

### 10.3 Rewrite program (P78 surface)

- **R1 drafter host-path reduction** (`llm_base_proposer.py`, pure MTP code):
  cached/preallocated buffers + out= ops in greedy sampling, fewer per-pass metadata
  rebuilds, target propose h 18 → ≤10 ms.
- **R2 async scheduling enablement** (retire WEDGEFIX-A forced-sync): env gate
  `VLLM_XPU_ALLOW_ASYNC=1` already exists, fork scheduler has async+spec paths
  (`engine/core.py:493`), no assert blocks it. Removes the per-step parse_output D2H
  (crash surface) and overlaps the ~25-30 ms glue. Historical lever +85-110%
  (v89-era). Hazards that forced it off are dead (v32 align-drain, v25 full-vocab
  gather, v52k copy kernel); #11 rendezvous risk re-certified by live battery.
  Sequence: env-only A/B first, wedge battery = user's live deep traffic + polls.
- **R3 mixed-eager**: no structural change (merge is throughput-optimal; §10.2).

## 11. Rounds 3-5 — v78 engine variant, fleet bring-up, A5000 stock-CUDA reference

### 11.1 v78 engine variant (Track 2) — implemented, staged, UNBOOTED

Design: make the TARGET's `embed_tokens` + `lm_head` TP1-replicated inside the TP2
engine via a module-scoped spoof of `vocab_parallel_embedding`'s own
`get_tensor_model_parallel_world_size/rank` bindings, scoped to the target-load
window ONLY (construction AND checkpoint row-slicing both read those module
bindings — a class-init wrap misses weight loading). The drafter then shares them
through default MTP share logic. Kills the target embed all-reduce, the logits
gather, and the head collectives.

- 4 files: `gpu_model_runner.py` (load-site spoof, fail-loud binding assert),
  `llm_base_proposer.py` (3 build-site gates widened to `in ("1","engine")`;
  share-forcing branches stay `== "1"` so engine falls to upstream share=True),
  `interface.py` (mamba page ×TP gate), `logits_processor.py` (`_get_logits`
  gather gate mirroring the existing `get_top_tokens` law).
- Memory: embed/lm_head footprint 2.78 → 5.58 GB → available ≈ 5.48 ≥ 4.97 GiB
  needed for 262,144 len (the capacity assert that killed boot 3).
- SYNTAX_OK all 4; staged `/root/build/p77/code/` on 10.20.3.65; inert until a
  boot window. Lane is in production use (foreign gateway traffic) — restart
  windows need care.

### 11.2 Fleet bring-up round (3 headless claude instances)

- Conductor REPORT: **VERDICT BOOT_FAILED** — boots 1-3 died
  (`split_with_sizes` → unify assert → capacity 4.97 > 3.31 GiB); live lane
  reverted to v77 fallback posture (propose ~10.2 ms, greedy 2.2 ms dead-flat =
  collectives still present). Independent confirmation of the coordinator's own
  R1b bring-up: the engine variant needs the v78 spoof to clear capacity.
- Verifier Pass-1: propose_d median 10.2 ms, greedy_d 2.2 ms flat in 40/40,
  10/10 sanity; spec acceptance 2.70 while carrying foreign gateway load.
- Law: one-shot `claude -p` instances CANNOT self-schedule wake-ups — both
  watcher and verifier "scheduled" then exited; artifacts bank to that point,
  coordinator cron is the safety net.

### 11.3 A5000 stock-CUDA reference lane (10.100.8.6) — LIVE 2026-10-03

Hardware/stack: RTX A5000 24 GB (Ampere SM86), stock `vllm/vllm-openai:cu129-nightly`
(0.23.1rc1.dev515+g68ee8300a), TP1 host-net :8100, container `vllm-mtp`,
MTP×3 (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`),
model `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (GPTQ sym G128 → marlin
on Ampere; MTP head excluded from quant via `"-:.*mtp.*": {}` ignore rule, stays
BF16; 18.22 GiB). Host NEVER REBOOTED (litellm gateway :4004 lives here).

Boot laws banked:

- **IMAGE TRAPS:** cu129-nightly ENTRYPOINT interprets CMD as `vllm serve`
  JSON-args → arbitrary python needs `--entrypoint python3 -c ...`; ANY python
  in the image needs `--gpus all` (device probe at import).
- **KEY NAMING:** qwen3.8's MTP config key is `mtp_num_hidden_layers: 1` (NOT
  `num_nextn_predict_layers`) — grep `-iE 'nextn|mtp|draft'` or you
  false-negative on quant configs. Community W4A16 quants often LACK the MTP
  head entirely; this one preserves it.
- **LOAD-ORDER OOM LAW:** gmu does NOT govern load-time memory. The MTP head
  constructs its OWN embed (1.47 GiB) + `ParallelLMHead` (2.37 GiB, at
  `qwen3_5_mtp.py:389`) BEFORE upstream share logic deletes them — the OOM is
  in that transient. Fix = shrink the init workspace:
  `max-num-batched-tokens 2048 + max-num-seqs 32 + max-model-len 16384 +
  enforce-eager + expandable_segments`.
- **FINAL FIT:** gmu 0.92 → KV 1.46 GiB / 19,945 tok / 1.22× concurrency @16k
  (0.85 → 0 blocks; 0.90 → 0.99 GiB < 1.19 needed; the hybrid GDN state pool
  makes per-request need ≈ 77 KB/token effective — mamba state dominates, so
  max_num_seqs is a strong KV lever on hybrid archs).

### 11.4 Stock-CUDA MTP reference RESULTS (probe v2, 2026-10-03 00:01-00:09)

- Shorts 10/10: **2.66 tokens/step** (1493 accepted / 2703 drafted = 3.00
  width + bonus; log-window mean acceptance 2.35-2.98; per-position
  0.75 / 0.55 / 0.41).
- **DEEP SHAPE SURVIVED** — the XPU hazard shape on stock: 11.5k fill → 6
  growing turns → 11.8k → 2048-token long-gen at depth (12.2 tok/s). Drafting
  ACTIVE the whole way (metrics windows: mean acceptance 2.10-2.79, avg draft
  acceptance 31-75%), zero stalls, zero resets, health 200 after.
- Burst 8-way clean (200-cap ×8 concurrent).
- Draft width exactly 3.0 per draft at every depth (no ramp-down at short ctx).

### 11.5 STOCK PARITY LAWS (the comparison deliverables)

1. **ACCEPTANCE PARITY:** stock CUDA 2.66 tok/step vs XPU fork 2.70 — the
   fork's MTP proposal/verification QUALITY is stock-equivalent (±2%). The
   fork's deficit is pure execution-path (TP2 collectives + host glue), not
   algorithm.
2. **FITS-IN-DRAFTER GATE** (`_input_fits_in_drafter`, nightly
   gpu_model_runner.py:4419): stock drafts only while batch `max_seq_len +
   num_spec_tokens ≤` the DRAFTER's own max_model_len, which auto-syncs to the
   serve length ("Overriding draft model max model len from 262144 to 16384").
   The XPU lane serves 262,144 with spec×3 → the gate NEVER closes → drafting
   at ANY depth. A context-conditional spec gate is a stock-blessed root-fix
   lever for the deep-spec hazard — the finer-grained cousin of CRASHFIX-76's
   engine-level nospec-default.
3. **DEEP ACCEPTANCE DECAY IS MILD on stock** (per-pos 0.61-0.79 / 0.31-0.55 /
   0.17-0.45 at 11.5-13.8k): spec stays net-positive at depth; wholesale
   deep-disable leaves perf on the table — the stock-style threshold gate is
   the surgical form.
4. **EAGER + SPEC coexist on stock nightly** (this lane runs enforce-eager) —
   precedent for an XPU eager-spec posture while graphs are suspect.
5. **max_num_scheduled_tokens auto-clamped to 2048 under spec** (nightly
   warning) — same clamp family the fork observes.

| Metric | Stock CUDA A5000 (int4) | XPU fork lane (fp8) |
|---|---|---|
| Acceptance short | 2.66 tok/step | 2.70 |
| Acceptance deep | 2.10-2.79 @ 11.5-13.8k, ACTIVE | deep+spec = crash zone (≥22k) |
| Draft width | exactly 3.0 | 3 (mixed-width batches exist) |
| Deep hazard shape | SURVIVED, no stall/reset | guc ccs reset cascade class |
| Throughput | ~14 tok/s short, 12.2 deep (eager A5000) | accepted 34.8 tok/s |
| Concurrency burst | 8-way clean | 24-way standard clean |

### 11.6 Probe v2 fixes (tooling, for reuse)

- `/metrics` on nightly carries `{engine=,model_name=}` labels — snapshot regex
  must be label-aware (`([\w:]+)(\{[^}]*\})?\s+(value)`), else every counter
  silently parses as absent.
- Deep-shape sizing is tokenizer-dependent → size the fill via the server's
  `/tokenize` endpoint against max_model_len, don't hardcode token estimates.
- Delta bug law: never derive two snapshots from one expression — capture the
  pre-phase snapshot into a variable BEFORE the phase runs (short-circuit
  `x and snapshot() or y` took both snapshots post-phase → structural zeros).

### 11.7 Next actions (updated)

1. Boot-window test of v78 engine variant on the XPU lane (staged, §11.1):
   verify capacity ≥ 4.97 GiB + boot markers, then R1b SPECTIMING bar
   (propose ≤7 ms, greedy ≤1.0 ms).
2. P78 surface per §10.3, now benchmarked against §11.4/§11.5 stock numbers.
3. Consider porting the fits-in-drafter/context-conditional spec gate into the
   fork's runner as a hazard root-fix complement (stock parity + exposure law).

## 12. Round 6 — R1 sharded-view drafter head: LIVE + MEASURED (2026-10-03)

### 12.1 Design (v2, shipped)

In engine mode (`VLLM_XPU_MTP_TP1=engine`), after v78's `_v78_tp1_spoof` builds
the drafter's `lm_head` replicated, `_r1_apply_shard_view` converts it IN PLACE
to a TP2-sharded ZERO-COPY view of the target's replicated table: recompute the
real shard indices, mutate `own.shard_indices` / `own.tp_size` / per-partition
counters, then rebind
`own.weight = torch.nn.Parameter(target_lm_head.weight[start:end],
requires_grad=False)` with a `data_ptr()` aliasing assert. Every consumer then
flips to the stock sharded paths via the existing
`getattr(lm_head, "tp_size", None)` gates (get_top_tokens → (value,index) XPU
exchange; `_get_logits` → gather). Per-call GEMV read halves
(2.54 → 1.27 GB); drafter's own full table freed. Guard `_r1_shard_view_ok`
(Unquantized both heads, float target weight, padded-extents equal, no
added-vocab straddle, both replicated, dtype match) — on any failure R1 falls
back to v78 replicated-share, never worse.

### 12.2 Boot casualties → three new operational laws

- **FORK-API TRAP (boot 1, 12:07:31):** `AttributeError: type object
  'VocabParallelEmbeddingShardIndices' has no attribute '_get_indices'` — in
  THIS fork `_get_indices` is a **classmethod on `VocabParallelEmbedding`**
  (image `vocab_parallel_embedding.py` ~:320), NOT on the dataclass (that's
  newer upstream). Autopsy ONLY possible because the watchdog banked the
  killed container's full log (`WD_crash_121425/serve_full_Q22.log`,
  AttributeError at log :933-981). Fixed → `VocabParallelEmbedding.
  _get_indices(own.num_embeddings_padded, own.org_vocab_size_padded,
  own.num_embeddings, own.org_vocab_size, tp_rank, tp_size)`.
- **WATCHDOG-AUTORESTART RACE (12:14:25):** the 07:30 host reboot silently
  re-armed lane-watchdog (`enabled; preset: enabled` — a prior `systemctl
  stop` does NOT survive reboot). My 12:06 deploy's health gap hit its 420 s
  grace while the spec engine was still booting → it `rm -f`'d the container
  and restored chain nospec posture (KV 554,304 fingerprint; no spec, no
  --chat-template — ALSO the root of fleet-validator 400s from 10.100.8.6:
  jinja line-49 template errors on serves lacking the template bind). LAW:
  `touch /root/build/lane_watchdog.paused` before ANY planned restart; `rm`
  to re-arm. Memory-banked.
- **SETPROCTITLE BLINDNESS:** vLLM renames procs (`VLLM::EngineCore`,
  `VLLM::Worker_TP0/1`) rewriting comm+cmdline — `pgrep python3` / restart
  script v1 false-negatived the live tree and TERM'd a stale pid. Ground
  truth = `/proc/[0-9]*/cmdline|comm` scans with bracket patterns
  (`bin/v[l]lm`, `resource[_]tracker` to defeat grep self-match). restart_r1.sh
  v2 = TERM the serve CLI (proc-scan pid) → poll comm-scan count to 0 (8 s
  both runs) → relaunch. Clean kill law holds: SIGTERM, never -9 on a live
  spec engine.

### 12.3 Boot gates — ALL GREEN (container ed4068df, 12:23:15, health 200 @160 s)

- SHARDED-VIEW both workers: rank0 rows [0:124160), rank1 [124160:248320),
  fp16 [124160, 5120], tp_size=2, zero-copy assert PASSED.
- Target REPLICATED marker retained (v78 spoof intact).
- Draft KV block wrap present (R1b site).
- **KV 5.68 GiB — memory-NEUTRAL vs v78** (empirical, not estimated).
- 0 Tracebacks.

### 12.4 SPECTIMING (steady 200-step windows, both workers, temp 0.5 smoke)

| metric | v78 replicated-share | **R1 sharded-view** | R1b stock-fallback | §11.7 bar |
|---|---|---|---|---|
| propose d | 16.8-17.5 (regression) | **10.6-10.9** | 10.2 | ≤7 |
| greedy d | 4.3 (regression) | **2.2** | 2.2 | ≤1.0 |
| tforward d | 30-40 | **30.4-31.1 (kept)** | 47-56 | — |
| tlogits d | 4.4 | 4.3 | 4.3 | — |
| propose h | 12 | 13.1-13.7 | — | — |
| greedy h | ~1.0 | 2.2-2.4 | — | — |
| step_wall | — | 98.2 ms steady (first window 964-998 = compile warmup) | — | — |

### 12.5 Round-6 laws

1. **R1 VERDICT: device parity achieved.** The v78 drafter regression is fully
   recovered (propose d −38%, greedy d −49%) while v78's target-side win is
   retained (tforward 30-31 vs 47-56 fallback). Memory-neutral. Zero-degradation
   law met at the component level.
2. **THE §11.7 BAR (propose ≤7 / greedy ≤1.0) IS NOT REACHABLE BY THE STOCK
   SHARED PATH AT ALL** — the stock-sharded fallback measures the same
   10.2/2.2. The residual is the sharded exchange + host glue itself: cached
   exchange buffers (already shipped) absorbed ~6 ms/call of alloc storm, but
   per-call copy_ + all_gather launch remains (~+1-1.4 ms host vs v78's
   replicated path, visible in propose h 13.1-13.7 / greedy h 2.2-2.4 vs
   v78's 12/~1.0). The bar is R2 territory: async drafter path (host glue off
   the critical path), then the fits-in-drafter port.
3. **Fleet-400 environmental law:** watchdog-restored serves (nospec chain
   posture, no --chat-template) 400 on fleet payload shapes via the tokenizer
   template's jinja line-49 — present at boots 09:40 and 11:53, NOT a fleet
   bug. Any 400-wave check first serves the lane actually has (spec + template
   = R1 posture) before debugging the fleet.

### 12.6 Storm gate (leg5 shape, fresh host, 2026-10-03 12:47) — GATE FAIL, R1 exonerated

Host rebooted per hygiene directive; R1 redeployed clean on the fresh host
(all gates green, KV 5.68 GiB, health 200 @170 s). Standard storm (leg5 shape
18 convs × 4 turns): **ok=62/72, wall=1838 s — outside the spec-posture band
865-1312 s, 10 requests lost to a WEDGE.** Recovery: EngineCore's own RPC
timeout fired 13:18:10 (EngineDeadError → clean 500s to clients, APIServer
shutdown clean); 2 workers stayed deadlocked holding GPUs (kill -9 → zombies
→ host reboot per KILL-DIRTY law). Lane restored on the certified chain
(posture=ckpt, KV 554,304 nospec fingerprint, health 200) + watchdog
re-armed. All artifacts at `/root/build/lce1/WD_wedge_131924/` (serve log,
py-spy dumps, split jsonls; v134 leg5 original 74 lines preserved/restored).

Wedge anatomy (LIVE py-spy capture — first stack-level capture of the class):

- **TP0 (492)**: `all_gather → _xpu_allgather_allreduce → all_reduce →
  gdn_linear_attn.py:1413` — waiting forever in the TP collective.
- **TP1 (498)**: `_gdn_attention_core_xpu_impl` (`vllm/_xpu_ops.py:308`) one
  step earlier — **the GDN XPU kernel never retires**; the peer's collective
  can never complete. Target model, FULL_DECODE_ONLY graph replay
  (`cuda_graph.py:254`), deep context (~21.5k+ computed), batch 5.
- ZERO R1/drafter/lm_head frames in either stack — **R1 exonerated**; this is
  the P71/CRASHFIX-76 GDN deep-forward hazard class itself, manifesting as a
  pure two-rank deadlock with NO guc reset (dmesg clean throughout).

Round-6 storm laws:

1. **EXPOSURE-TIME LAW RE-PROVEN AT THE WEDGE GRADE:** the storm's tail
   entered cyclic-eviction re-prefill thrash (prefix cache hit 4.2%, KV
   54→72%, between-turn evictions forcing full 30-50k re-prefills at
   ~350 tok/s) — stretching deep-spec seconds past every prior leg until the
   kernel hazard asserted. Slower tail ⇒ more exposure ⇒ wedge; the baseline
   legs' 865-1312 s walls contained the SAME hazard with less exposure time.
2. **SLOW-TAIL = WHOLE-CONV LRU THRASH (v130 class at request level):** 5
   deep convs resident (5.68 GiB) + 5 capacity-queued; every admission evicts
   a prefix the conv needs next turn. Structural levers (size-aware
   eviction/admission) remain the capacity-side fix; the spec-side fix is the
   fits-in-drafter/context-conditional gate (§11.5 law 2), now DOUBLY
   motivated: it kills both the crash class AND this wedge's exposure.
3. **WEDGE FAILS OVER, CRASH DOESN'T:** RPC-timeout → EngineDeadError → 500s
   (no infinite client hangs) — but the workers' deadlocked kernels hold the
   GPUs until reboot. The lane's failure mode under the hazard is
   engine-death + host-reboot, never silent.
4. **Bake v1.2.30 STAYS GATED:** R1's component-level wins are real and
   exonerated, but no spec-posture ship trigger can fire while the deep-spec
   hazard asserts on the standard storm shape. Order: port the
   context-conditional spec gate → re-run this storm on gated spec → then
   revisit bake.

### 12.7 Lane state at close of round 6

Certified chain posture (ckpt, nospec v1.2.29, KV 554,304) on a fresh host;
watchdog ACTIVE (pause flag removed after restore). R1 code master stays at
`.tmp-p76/v78/code/llm_base_proposer.py` (md5 0992294b0b24b15915b10a44972296d9)
+ host `/root/build/p77/code/`, deploy via `run_v78.sh` (idempotent post-
reboot). Storm runner: `wsc_pressure_v134_leg5.py` (writes to its ABSOLUTE
jsonl path — pass a copy or accept the split-at-header dance).
