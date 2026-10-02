# GuC CCS Engine-Reset / DEVICE_LOST Crash Class — Upstream Handoff

**CRASHFIX-76 Stage B3 (P74c)** · 2026-10-02 · lane `lsv-test` @ 10.20.3.65

## Executive summary

An intermittent GPU crash class kills long-running vLLM serving engines on
Intel Battlemage (BMG G21, 2× `8086:e223` at PCI `b1:00.0` / `da:00.0`),
kernel `6.17.0-1010-intel`, GuC firmware `bmg_guc_70.bin` 70.72.1. The
GuC scheduler fires a **ccs (compute) engine reset** with reason
**"LR job cleanup"**; a kernel-submitted bcs job then times out behind the
wedged context within ~6 s and the device escalates to a full GT reset →
`DEVICE_LOST` for every process on the card.

We have banked two full mid-storm devcoredumps plus 27 earlier partial
resets (P71 census, chronic since 2025-09-15), a reliable repro (~2/2 at
the deep shape), and a forensic toolchain for the dump format. What we do
**not** have is a kernel-level handle — the dumps carry no kernel-level
telemetry (see GUC-MODE HWSP LAW below).

**We have ruled out our own code** (custom ESIMD kernels and the MTP
speculative path — see verdicts below). The remaining trigger is the
upstream compute path at deep context: a long-running batch dispatched to
ccs never retires its first command (ACTHD == RING_BBADDR = batch start).

## Signature invariants (identical across both banked dumps)

| Field | WD_crash_132400 (card2 `da`, 2026-10-02) | LIVE_crash_144353 (card1 `b1`, B1 leg) |
|---|---|---|
| Engine | ccs32 | ccs (guc_id=22) |
| Reason | `LR job cleanup` | `LR job cleanup` |
| Timeout | `9223372036854775807` ms (infinite) | same |
| ACTHD | `0x0000d5569db23a04` == RING_BBADDR | `0x0000c001ffde8344` == RING_BBADDR |
| RING_INSTDONE | `0xffdefffe` (clear bits 0/17/23/31) | `0xffdefffe` |
| ROW_INSTDONE | `0` ×16 | `0` ×16 |
| SAMPLER_INSTDONE | `0xffffffff` | `0xffffffff` |
| IPEHR | `0x61050001` | `0x61050001` |
| Ring head/tail | `0x2fd4` / `0x3028` (0x54 B unretired) | unwound, same shape |

ACTHD == RING_BBADDR on both: **the hang is at batch START — the first
dispatch of the batch never retires.** P71 census adds: every banked event
since 2025-09-15 says "LR job cleanup"; guc_ids 22/32/52/62/112 across
epochs; both cards.

## Trigger shape (measured)

- **Depth, not occupancy**: dying batches have `num_computed >= 22k`
  tokens per request in the batch; KV pool sits ~15 % full. Deep-context
  target-forward decode (attention + GDN recurrence), not memory pressure.
- Standard shape (72 reqs, shallow prompts): 5/5+ clean, walls 865–1312 s.
- Depth shape (18 convs × 6 turns = 108 reqs, prompts 29.5 k–35.5 k tok):
  **2/2 death** at turn 4–6 (T+27…46 min); one survived lone reset at
  engine-idle (survivable, no escalation).
- HTTP symptom: wedge → engine unresponsive; requests in flight hang
  until client timeout (1200 s).

## Verdicts: what is ruled out (our side)

1. **NOT the custom ESIMD kernels** (CRASHFIX-76 Stage A, B1 leg
   2026-10-02): all 15 `DISABLE_ESIMD_*` knobs set (certified 1:1
   upstream fallback path) → same-class crash at depth turn 5
   (14:43:52, card1, guc_id=22, ok=84/102). Cross-kernel-set identical
   signature.
2. **NOT firmware version alone** (FW TIMELINE LAW): manual
   70.44→70.72.1 swap happened 2025-09-24 08:06 (`.zst.bak-7044`);
   P71's 27 banked resets date ≤2025-09-15 — the class predates the swap;
   both fw versions exhibit it.
3. **MTP speculative path IS the trigger** (CRASHFIX-76 Stage B2,
   2026-10-02): nospec depth legs **2/2 clean** (108/108, walls 431 s /
   337 s, zero resets) at the identical shape where spec died 3/3. The
   class fires in the MTP draft/verify deep forward. Separation
   analysis: standard-shape legs also reach 35.7 k per-request depth
   and 10 concurrent-deep requests while clean — the hazard scales with
   deep-spec-decode EXPOSURE TIME, not with any per-step discriminator.

## Repro recipe

Host scripts (transfer with pscp; run in maintenance window):

1. Boot the certified posture: `boot_v1227_restore.sh <mode>` (chain
   restores container `lsv-test`, image `llm-scaler-exp:v1.2.28`,
   patchers, spec MTP×4, kv fp8_e4m3, async scheduling; health on :8000).
2. Arm the dumper: `setsid nohup /root/build/devdump_watch_v3.sh &`
   (read-probe + harvest + dismiss + pid maps; fingerprint dedupe).
3. Launch the depth storm: `v134_b2nospec.sh`-style runner for
   `wsc_pressure_v134_leg7_depth.py` — 18 convs × 6 turns, prompt tokens
   29.5 k–35.5 k, urllib timeout 1200 s, append-mode jsonl with start
   marker; terminal line `PRESSUREB_DONE ok=N/M wall=…`.
4. Expect death at turn 4–6 within T+27…46 min; watcher harvests the
   devcoredump ≤2 s after the sysfs slot appears (harvest window ~6 min
   before the next reset dismisses it).

## Capture laws (sysfs devcoredump, this platform)

- **SYSFS-DEVDUMP-SIZE-0 LAW**: the `devcoredump/data` node **always**
  stats 0 bytes (bin_attr size unknown). `[ -s ]`/stat-gated capture is
  structurally blind; readability exists only ON READ (`cat`).
  Valid dumps observed ~512 KB; require ≥100 KB before saving.
- `echo 1 > dismiss` frees the slot for the NEXT reset but does not
  stop re-arm — the watcher fingerprints content (md5) per card and
  harvests only on change.
- **VM-STATE-ERROR-19 LAW**: dump `[VM].error: -19` (VM destroyed at
  capture) — the dump's internal page-table walk is dead by
  construction; user VA naming needs live `/proc/<pid>/maps` (the
  watcher snapshots maps+cmdline of the `Process:` pid from the dump).
- **GUC-MODE HWSP LAW**: the hardware status page holds job-level
  seqnos only (0x200 last retired, 0x208 in-flight, 0x210 job
  timestamp) — **no kernel-level telemetry on the card**; kernel naming
  needs config A/B or live maps cross-reference.

## Forensic tooling (in-repo)

- `p73_analyze.py` — devcoredump format discovery: section map, anchor
  offsets (Process/Reason/ACTHD/RING_*), `--around PAT:N` context dump.
- `p73_decode_blobs.py` — ASCII85 decoder for `[HWSP].data` (0x1000) and
  `[HWCTX].data` (0x2000) blobs; prints nonzero dword runs and seqno
  anchors. HWCTX decodes to the LRC LRI-restore command stream.
- Format: text sections separated by `**** count` markers; blobs are
  ASCII85 with `z` zero-run shorthand; printed dwords are true register
  values.

## Environment

| Component | Value |
|---|---|
| Host | 10.20.3.65, 2× Xeon Gold 5218 |
| GPU | 2× Intel BMG G21 `8086:e223` (`b1:00.0`, `da:00.0`) |
| Kernel | `6.17.0-1010-intel` |
| GuC fw | `bmg_guc_70.bin` 70.72.1 (Sep 24); prior 70.44 backed up |
| Driver | xe (GuC-mode scheduling, ccs0-3 + bcs engines) |
| Engine | vLLM 0.21.1.dev0 (prod fork `llm-scaler-exp:v1.2.28`), TP=2 |
| Model | qwen3-next hybrid (GDN + full attention), fp8 W8A8 dynamic, fp8_e4m3 KV |
| Serving | spec MTP×4, XGrammar-2, async scheduling, hybrid allocator (page 1024 tok) |
| Workload | multi-turn conversations, deep context ≥22 k computed tokens |

## The ask

Upstream/driver-side investigation of the ccs "LR job cleanup" reset
class under long-running deep-context compute batches on BMG:

1. Why does a batch's first dispatch (ACTHD == RING_BBADDR) stop
   retiring on ccs while the ring shows only 0x54 B unretired?
2. Kernel-level context (which batch/kernel/offset) is not recoverable
   from the devcoredump in GuC mode — is there a debug path
   ( GuC logging, KMD trace ) we can enable on a repro host?
3. Expected limits: is there a documented maximum batch/compute
   duration for ccs contexts under GuC "LR job cleanup" policy, and a
   recommended mitigation (engine pinning, preemption granularity,
   heartbeat tuning) for serving workloads with deep-context decode?

Repro available on request (scripts above run end-to-end in ~50 min,
2/2 reliability at the death shape).
