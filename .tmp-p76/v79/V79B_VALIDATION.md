# v79b/v79a width-2 validation — FINAL (2026-10-04)

Posture: `DFLASH2_EMIT_K=1` + v79b scheduler trim (patch md5 9dd74c96, in-image
scheduler.py `update_draft_token_ids`) on the R1 stack (v1.2.29 image + binds,
TP1-engine variant, spec config k=4 MTP ×4 intact). v79a adds the drafter
loop-skip (host bind `llm_base_proposer.py` md5 77fd6e17, marker count 1):
at EMIT_K=1 the drafter exits after position 0's forward and zero-fills the
remaining [B,k] emission slots — the v79b trim never schedules them.

## Results matrix

| Cell | Result | Notes |
|---|---|---|
| Smoke ×2 (greedy) | coherent, deterministic | 39 drafts, 39 accepted ALL pos0, pos1-3 = 0 — width-2 exact signature; 2.0 tok/step @ 2 rows |
| Standard leg5 | **ok=72/72, wall=1569s** | CLEAN — zero errors, zero resets (R1 stack same-shape: wedge 62/72 @ 1838s) |
| Depth leg7 rep1 (v79b) | **ok=83/108, wall=3347s, WEDGE** | RPC timeout 16:32:25, 11× HTTP 500, **0 resets** — crossed BOTH crash death marks (2060s/2740s) clean; log banked `lce1/v79b_depth_wedge_r1_serve.log` |
| **Nospec leg7** (same stack, spec off) | **ok=108/108, wall=362s** | THE CELL NOBODY RAN — depth shape is harmless without spec |
| Depth leg7 rep2 (v79b) | **ok=96/108, wall=3528s, server-clean** | health 200 throughout, 0 errors, 0 resets; 12 lost rows = client BARRIER timeout (900s) on slow deep-tail convs — NOT server faults. Width-2 depth wedge rate 1/2 intermittent |
| Depth leg7 v79a (drafter 4× cut) | **ok=87/108, wall=3428s, WEDGED @ tail** | both crash marks crossed clean (0 resets); early-storm pace ~2× (43 rows @T+10min vs ~18-20 v79b); wedge at 19:09:36 T+56min: RPC `sample_tokens` timeout → 15× 500 → EngineDeadError CLEAN shutdown; workers orphaned then self-terminated |

## Wedge forensics (v79a round — first Python-frame capture)

Live py-spy dumps at wedge time (`lce1/WD_wedge_live_191306/tp{0,1}.txt`):

- **BOTH ranks parked at the identical frame**:
  `parse_output (vllm/v1/sample/rejection_sampler.py:270)` ←
  `_bookkeeping_sync (gpu_model_runner.py:4089)` ← `sample_tokens` ←
  `worker_busy_loop`. Line 270 = `output_token_ids.cpu().numpy()` — the
  **D2H sync** where an unretired device kernel becomes visible in Python.
- 0 guc/GT resets DURING the storm — pure two-rank stall, not the crash class.
  **CORRECTION (post-mortem, 19:14:36):** at worker teardown (~5 min after the
  RPC timeout) BOTH cards logged teardown-class engine resets — ccs guc_id=32
  (da:00.0) / guc_id=22 (b1:00.0) + bcs 36/26, the LR-cleanup pair pattern, no
  devcoredump (banked `lce1/v79a_postmortem_resets.txt`). Mid-storm record
  stands: zero resets while serving; the wedged kernels only reset when their
  contexts were destroyed.
- Server death path is CLEAN: EngineCore RPC timeout (60s) → EngineDeadError →
  APIServer graceful shutdown. No hang-forever; workers terminate on their own
  shortly after. Recovery = plain restart, but standing hygiene (reboot before
  next leg) applies given the teardown resets.

## SPECTIMING — mystery solved + deep-phase economics (v79a leg)

18 lines existed all along (`lce1/v79a_spectiming_full.txt`): **worker stdout
is block-buffered under the file redirect — lines flush only at worker exit**;
every mid-run grep (both rounds) read an unflushed buffer. Emission = every 200
steps per rank. TP0 series (ms per 200-step window):

```
ts        wall/200  tforward_d  propose_d  dforward_d  greedy_d
18:13        2013         289          14          11        2   (smoke+prefill burst)
18:14-18     320-433    250-357       12-19       10-14      2-5   (shallow decode)
18:20          619         532          27          21        4
18:23          876         705          93          63        4
18:28         1600        1224         201         138       27
18:44         4696        3405         709         375      121   (deep tail, 30k+ computed)
```

- **Deep step = 23.5 ms, of which target verify forward = 17 ms (72%)**;
  drafter total ≈ 6 ms (26%) — and dforward is 1.87 ms/step = exactly ONE
  forward (v79a loop-skip verified working from timing alone).
- tforward grows ~12x shallow→deep and wall is device-bound on it — the deep
  tail IS the target verify forward. The parked D2H at the wedge waits on a
  ~17 ms-class tforward kernel that never retires → wedge site = target verify
  forward (GDN attention at 30k+ computed), consistent with R1's device capture.
- Early-storm 2x pace under v79a = drafter share at shallow steps; at depth the
  4x drafter cut saves ~1.5 ms of a 23.5 ms step — explains the unchanged wall
  band (3428s ≈ v79b).

## Laws established (round-final)

1. **WEDGE IS SPEC-LINKED** — nospec leg7 = 108/108 @ 362s: the depth shape is
   harmless without spec. Spec-linked at ANY width tested (×4 3/3 death+wedge,
   width-2 2/3 wedge, v79a 1/1 wedge).
2. **Width-2 kills the CRASH class** — every width-2 leg (v79b ×3, v79a ×1)
   crossed both CRASHFIX-76 death marks with zero guc/GT resets. Exposure-time
   law holds for the ccs-reset onset; the deadlock has a separate floor.
3. **DRAFTER EXONERATED — wedge driver = TARGET VERIFY PATH** (v79a's purpose):
   with drafter forwards cut 4× the wedge still fired at the same depth band.
   Both ranks park at the verify-step sampler D2H (`rejection_sampler.py:270`),
   i.e. waiting on kernels enqueued by the TARGET forward/sampler of the verify
   step at deep context. Consistent with R1 device view (TP1 kernel never
   retires / TP0 all_reduce) and P71 fr-trace (never mid-propose). Next probe
   surface: target-verify forward (attention/GDN at 30k+ computed), not the
   drafter.
4. **Spec depth economics ~9x underwater** — 362s (nospec) vs 3347-3528s
   (width-2) on leg7; standard band 330-372 vs 1569. Production posture stays
   nospec; spec workstream value = mechanism knowledge + R2 target.
5. **Zero-width agreement law** (v79 falsification): safe only when
   scheduler-emitted width == GMR udql == graph lattice. Engine variant never
   constructs AsyncScheduler (v42c dead); live trim authority =
   `update_draft_token_ids` storage (v79b).
6. **Propose-time share is shallow-depth only** — v79a doubled early-storm row
   pace but the deep tail dominates the wall (~3428s ≈ v79b band): at 30k+
   computed the target verify forward dwarfs propose.

## Post-round lane state (2026-10-04 ~19:20)

- lsv-test container UP, serve DEAD (EngineDeadError shutdown), workers gone.
- Lane NOT restored: certified-nospec chain + watchdog re-arm pending user
  decision (watchdog paused via /root/build/lane_watchdog.paused — still there).
- Host reboot recommended before ANY next leg (standing hygiene; KILL-DIRTY
  not triggered — no kill -9 was used; server died on its own).
- v79 stack = experimental (in-image v79b scheduler patch + host-bind v79a
  drafter patch + /root/boot_v79_x4.sh). Certified chain boot remains
  boot_v1227_restore.sh (md5 882b613a…, V1227_IMAGE v1.2.29).
- SPECTIMING still emits 0 lines with env verified present — unresolved,
   non-blocking (documented both rounds).

## Artifacts

- Host: `lce1/v79a_depth_wedge_r1_serve.log` (623 KB),
  `lce1/WD_wedge_live_191306/{tp0,tp1,parse_output_context}.txt`,
  `v130_stage/wsc_pressure_v134_leg7.v79{b: r1 104, r2 106, a: r1 107}-row.jsonl`
  splits + `v79_leg7_*.out` summaries; originals restored (leg5 74, leg7 865).
  Nospec depth per-row detail NOT banked (lost in rep2 restore) — headline
  preserved in `v79_leg7_nospec_r1.out`: ok=108/108 wall=362s.
- Local: patch_v79b_sched_trim.py, patch_v79a_drafter_skip.py (post-patch
  assert corrected to `count == 0` — the shipped host copy patched fine),
  boot_v79_x4.sh, boot_v79_nospec.sh, V79_BROKEN_ANALYSIS.md, this file.
