# spec-prefill-peer-barrier v23 — prefill-step peer barrier for the spec drafter (#11) — DISPROVEN

> **Outcome (2026-08-30): this fix does NOT work.** Kept as the documented
> negative result that motivated v24. Live catches on the v23 image still
> wedged (`run=1` frozen, ttft counted, success frozen), and py-spy native
> stacks showed why (below): the slow rank blocks BEFORE the barrier is
> reached. See `spec-prefill-peer-rendezvous-v24` for the follow-up (also disproven) and
> `vllm/KNOWN_ISSUES.md` #11 for the corrected verdict.

`llm-scaler-vllm-adv:v23` = validated v22 image + one env-gated overlay in
`vllm/v1/worker/gpu_model_runner.py` that ATTEMPTED to stop the silent loss
of large-prompt requests under MTP (KNOWN_ISSUES #11).

Rollback: `VLLM_SPEC_PREFILL_BARRIER=0` restores v22 behavior. Default: `1`.

## The bug (KNOWN_ISSUES #11)

With the user's flags (TP=2, `--enable-prefix-caching`, MTP k=4,
turboquant_4bit_nc, `--async-scheduling`, FULL_DECODE_ONLY graphs):

- **~7.5% of ~4.8k-token-prompt requests never complete** (streaming and
  non-streaming equally): the client times out, the engine's success counter
  never moves (the request is not slow — normal latency is a flat 2.2 s),
  and the request sits `Running` with KV allocated until the client
  disconnect triggers the abort. Serve log completely clean; dmesg clean.
- 12-token prompts never wedge (0/200) — the trigger needs a multi-block
  prompt (block size negotiates to 4096 under tq4nc on this hybrid model).
- **py-spy during the wedge** (both workers "active"):
  - TP0 spinning in `ur_command_list_manager::appendUSMMemcpy` — the
    align-mode `.cpu()` D2H in `_update_states_after_model_execute`
    (gpu_model_runner.py:1476) cannot append because its stream never drains;
  - TP1 up to a full step ahead, polling `urEventGetInfo` in
    `_calc_spec_decode_metadata`, or blocked in the Triton launcher inside
    the MTP draft forward — its submission queue is jammed behind the same
    stuck stream.
- **xpu-smi during the wedge**: both GPUs Compute+Copy engines 100%, EU
  ~11% active / ~56% stall — oneCCL spin-wait kernels burning both engines.
  At prefill there is no competing queue work, so the 640 ms GuC preempt
  timeout documented for the deep-stream #05 variant never fires: no reset,
  no DEVICE_LOST, just an eternal spin.
- The StepWatchdog (`VLLM_WORKER_STEP_TIMEOUT_S`, 600 s default on XPU)
  would eventually kill the worker on a >600 s wedge; most wedges are
  "resolved" earlier by the client disconnect abort, silently.

### Isolation matrix (v22 image, ~4.8k prompts, 80 requests per cell)

| cell | delta vs user flags | lost |
|------|---------------------|------|
| BASE | none | 6/80 |
| B | `--mamba-cache-mode all` (drops the align `.cpu()` sync) | 2/80 |
| D | no `--async-scheduling` | 6/80 |
| E2 | `--max-num-batched-tokens 4096` (chunked prefill) | 1/80 |
| C | no `--speculative-config` (MTP off) | **0/80** |

Only MTP-off is clean: the trigger is the spec drafter's collectives right
after a large prefill forward — the documented #05 oneCCL peer-late race
family, prefill variant. When the two ranks reach the drafter skewed (one
host still draining its prefill stream, the peer already submitting head
collectives), the non-preemptible spin-wait kernels miss the rendezvous.

## The fix

`GPUModelRunner.propose_draft_token_ids` now drains this rank's XPU stream
before running the drafter whenever the step scheduled >= 512 tokens:

```python
if _SPEC_PREFILL_PEER_BARRIER and num_scheduled_tokens >= 512:
    torch.xpu.synchronize()
```

Both ranks execute the same step code, so both drain: the peer-late window
collapses to host glue, and the head's collectives start from empty,
symmetric queues. Decode steps (`max_num_seqs * (k+1)` tokens — 320 at 64
seqs / k=4) skip the barrier and keep the async-scheduling fast path; the
sync cost is paid only on large prefill steps (~2 s each), where losing tail
overlap is negligible.

## Validation (v23, user flags unchanged otherwise)

**Disproven.** Two things went wrong with the first validation pass, both
caught by instrumented re-runs:

1. **The initial "0 lost" batches used the wrong prompt size.**
   `loss_rate2.py <N> <REPS>`: arg 2 is the prompt REPS (~16 tokens each),
   not the timeout (hardcoded 45 s). Calling it with `80 45` produced
   ~740-token single-block prompts — a population that (like 12-token
   prompts) almost never wedges. The valid comparison is REPS=300 (~4.8k).
2. **A wedge catcher with correct pids + py-spy + metrics polling** (hold
   the connection open, sample `num_requests_running`/success/TTFT every
   5 s) reproduced the wedge on v23 at the v22-like rate (catches at
   request 12 and 59 of 300): `run=1` frozen 90-120 s, ttft counted,
   success counter frozen, freed only by client disconnect (and even then
   no abort is counted).

**Decisive py-spy stacks during a v23 wedge (native):**

- TP0 stuck BEFORE the barrier: `_update_states_after_model_execute`
  (align-mode `.cpu()` D2H) inside `ur_command_list_manager::appendUSMMemcpy`
  — the copy cannot even be SUBMITTED because this rank's in-flight queue
  never drains.
- TP1 already PAST the barrier (its stream drained fast — it is ahead),
  inside the drafter issuing the MTP head's vocab all-gather
  (`qwen3_5_mtp.py:476 -> _gather_logits -> ProcessGroupXCCL`).

A one-sided drain is symmetric in name only: each rank drains its OWN stream
and immediately proceeds, so the fast rank still enters the collective
alone. Hence v24 (true gloo rendezvous). Prefix-cache validation on the same
boots was unaffected and correct: hits 4096/8192 exactly at the
`(floor(n/4096) - 1) * 4096` formula, warm TTFT 2.3-3.2x faster; canonical
car-game run clean.

## Files

- `gpu_model_runner.py` — overlay (2 hunks: module flag + barrier).
- `Dockerfile` — FROM v22, grep guards + py_compile + ast checks.
- `serve.sh` — v22 reference serve + `VLLM_SPEC_PREFILL_BARRIER` forwarding.
