# Speculative prefill: diagnosis and phase-race fix

2026-09-09. Patch status: **reproduced in the installed scheduler, fixed, regression-tested, and built into separate images**. Full GPU serving validation of these images is still pending. The ongoing extension benchmark was left running.

## What the live evidence establishes

Prefill is not universally broken with speculation. The archived 131072-token, single-client runs completed five repetitions without request errors for both [MTP4](evidence/cell_mtp4_131072_c1.json) and [DFlash7](evidence/cell_dflash7_131072_c1.json). Their first repetitions reached first content in approximately 86.9 and 84.4 seconds respectively.

Three different symptoms need separating:

| Symptom | Evidence and interpretation | Action |
|---|---|---|
| Prompt throughput stays at zero for many log intervals | Archived MTP logs show zero prompt throughput during growing KV occupancy, followed by a prompt-token accounting burst when output starts | Do not interpret this metric alone as stalled computation. Use per-request computed-token progress and TTFT; a zero reading is not a reason to abort |
| Near-262k DFlash/FP8 request fails immediately | [Recorded HTTP 400](evidence/cell_f8e4df7_261888_c1.json); extension driver launches this lane with maximum model length 245760, below the 261888-token prompt | Use a prompt/output budget inside that lane's supported limit. Raising the limit requires separate draft-KV/capacity validation; bypassing validation cannot fix it |
| An empty prefill result is treated as a failed decode | Reproduced against the real installed `AsyncScheduler` methods; described below | Apply `spec-prefill-phase.patch` |

The inspected MTP and DFlash compressed logs contain **zero `STRIKE-OUT` entries**. Therefore the newly reproduced race is not proven to be the cause of a particular live incident reported by the user. A request timestamp/error is still needed for that attribution. The patch fixes a concrete unsafe abort path; it does not claim to cure every prefill stall, GPU fault, or long-context slowdown.

## Root cause: mutable request state is not the phase of a returned step

`Scheduler._update_after_schedule()` advances `Request.num_computed_tokens` immediately and overwrites `Request.is_prefill_chunk`. This intentionally allows the next prefill chunk to be scheduled before an earlier result is processed. The engine's `step_with_batch_queue()` retains the corresponding `SchedulerOutput` alongside each pending future.

The fork's `AsyncScheduler.update_from_output()` implements the v52m zero-emission guard. It decides whether an empty result is expected by reading the **current** `Request.is_prefill_chunk`, instead of the phase of the result's scheduled step. An intermediate prefill normally returns an empty sampled-token list.

Reproduction at a 131072-token prompt with 8192-token chunks:

1. Request starts at 114688 computed tokens.
2. Schedule chunk A: computed becomes 122880; `is_prefill_chunk=True`. A legitimately emits no token.
3. Schedule final chunk B while A is pending: computed becomes 131072; `is_prefill_chunk=False`.
4. Resolve A's empty output. The original guard reads B's current phase and records a false decode strike for A.
5. A strike also arms the five-second watchdog. A later scheduling tick can invoke `STRIKE-OUT` even without three empty decode results if the watch remains armed. Long final-prefill execution makes a wall-clock assumption particularly unsafe here.

This is independent of the target KV dtype and model arithmetic. It can affect nonspeculative async chunked prefill too; MTP/DFlash are not the only possible triggers.

## Patch

[spec-prefill-phase.patch](spec-prefill-phase.patch) changes only two installed files:

- `vllm/v1/core/sched/output.py`: append an optional immutable `scheduled_prefill_chunk_req_ids` field. Appending preserves the positions of existing dataclass constructor arguments.
- `vllm/v1/core/sched/async_scheduler.py`: snapshot intermediate-prefill request IDs immediately after the base scheduler advances this step's counters. Consult that snapshot before recording a zero-emission strike. Preserve the existing current-prefill, finished-request, and structured-output exclusions.

Unknown phase metadata does not justify a strike. Normal patched scheduling always populates it; manually constructed or older outputs have `None` and are conservatively excluded from this guard. Deploy both files together using a fresh process/image. Genuine empty decode results with known phase still reach the existing strike logic.

The snapshot adds one small CPU set per step. It does not change scheduled widths, speculative placeholders, causal masks, model tensors, KV allocation, recurrent-state rollback, GPU kernels or collectives. It does not disable chunked prefill, speculation, async scheduling, or all failure detection. No GPU synchronization is introduced.

## Verification

[test_prefill_phase.py](test_prefill_phase.py) imports the actual installed scheduler classes. It bypasses model initialization, stubs downstream output bookkeeping, and injects a clock so the race and watchdog can be tested deterministically without loading a model or occupying the GPUs.

- Baseline initial suite: six test methods, 13 failing assertions/subcases and one missing-field error. All 12 combinations of prompt length 65536/131072/261888 and draft width 0/1/4/7 reproduce the false strike. See [baseline failure log](evidence/baseline-tests.log).
- Final patched suite: eight methods pass on **both image bases**, including the 12 race cases, mixed prefill/decode rows, snapshot serialization and independence, preserved placeholder/width accounting, valid-output recovery, unknown phase, cancellation/grammar exclusions and genuine empty-decode detection.
- Source generator checks expected source layout, parses generated Python and records SHA256 hashes. [Manifest](evidence/manifest.json).
- Unified patch dry-run passed against the live installation. It was not applied there.
- These are scheduler regressions and packaging checks, not GPU numerical, long-context throughput or full-service tests.

## Built artifacts on 10.20.3.65

| Base | New image | Image ID |
|---|---|---|
| `llm-scaler-exp:v1.2.6t2` | `llm-scaler-exp:spec-prefill-phase-v1` | `sha256:7840c648e83bfd1b04cf17d97cf58b6e94d43d6e04f06b454a0973a555957e8a` |
| `llm-scaler-exp:v1.2.5` | `llm-scaler-exp:spec-prefill-phase-v1-v125` | `sha256:563f7362bb3592120276e95b1a5864b6862953d760488defabb97c5fbb566c97` |

Build logs: [v1.2.6t2 base](evidence/build.log), [v1.2.5 base](evidence/build-v125.log). Base image IDs at inspection were `006791e5f95c2d262f962198df5929953b441a2be5fe0309eb12fde7cea72c5c` and `a522bf15be2b666e7dcaab1d59557c8cc21e3bbcdf6be0a9c6bbf686b31f0ff8`. Tags can change; check the base ID when rebuilding.

From this directory, on the Docker host:

```sh
docker build --build-arg BASE_IMAGE=llm-scaler-exp:v1.2.5 \
  -t llm-scaler-exp:spec-prefill-phase-v1-v125 .
docker run --rm --entrypoint /opt/venv/bin/python \
  llm-scaler-exp:spec-prefill-phase-v1-v125 \
  /tmp/spec-prefill-fix/test_prefill_phase.py
```

Alternatively generate patched sources into a separate directory with `build_patch.py SOURCE_SCHED_DIRECTORY OUTPUT_DIRECTORY`. The generator refuses identical source/output directories. Keep the patch and manifest for review; do not hot-replace modules in a running engine.

## Full-service qualification still required

After the existing benchmark finishes, compare base/candidate with identical MTP4 and DFlash7 flags, TQ4-NC and supported FP8 lanes, clients 1 and 2, async scheduling and chunked prefill enabled. Keep each image's actual context/capacity limit. Include 64k and 128k, then near262k only where the launch supports it, reserving output and speculative headroom.

Exercise the penultimate/final chunk transition; overlap a decoding request with a long prefill; repeat with a reused prefix and a cancelled/resumed request. Require normal completion/usage, no false zero-emission/watchdog abort, preserved token/placeholder accounting, and no TTFT regression beyond run variability. Compare output and acceptance against repeated baselines. Separately test legitimate terminal delivery for genuine invalid decode results.

Remaining bugs are not masked by this patch: the guard still infers failure from empty decode emissions rather than an explicit worker error status; recurrent-state/slot-boundary failures and device/transport hangs require their own evidence. A broad timeout increase, blanket guard disable, or proposer-only speculative-width change would not repair the demonstrated phase race.

## Upstream comparison

The current [upstream issue #54392](https://github.com/vllm-project/vllm/issues/54392), opened August 30, 2026, describes a different speculative-prefill contract failure: disaggregated KV admission plus Mamba alignment leaves a physical query window smaller than its speculative placeholders. That report concerns an Ascend/DSpark/connector deployment, not this XPU server. It is relevant as a boundary-test design reference, not evidence that its patch should be transplanted here. The local fix above follows a directly reproduced fork-specific guard bug.
