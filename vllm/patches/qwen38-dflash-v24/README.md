# qwen38-dflash-v24 — prefill-step peer RENDEZVOUS for the spec drafter (#11) — DISPROVEN

> **Outcome (2026-08-30): this fix does NOT work either.** The wedge still
> reproduced on the v24 image (catch at request 17 of 300 with the plain
> rendezvous; at request 22 with `--mamba-cache-mode all`, which this hybrid
> model silently coerces back to "align" anyway — see
> `model_executor/models/config.py:438-446`). py-spy during the v24 wedge
> showed **both** ranks stuck at the SAME place — the align-mode `.cpu()`
> D2H submission — meaning the collective clog forms UPSTREAM of any
> barrier placed before the drafter (in the target/verify/drafter
> collectives themselves, the #05 oneCCL family). Neither barrier design
> can reach it. Kept as a documented negative result; the serving image
> remains v22. Full corrected verdict in `vllm/KNOWN_ISSUES.md` #11.

`llm-scaler-vllm-adv:v24` = validated v22 image + one env-gated overlay in
`vllm/v1/worker/gpu_model_runner.py` that ATTEMPTED to stop the silent loss
of large-prompt requests under MTP (KNOWN_ISSUES #11).

Rollback: `VLLM_SPEC_PREFILL_BARRIER=0` restores v22 behavior. Default: `1`.

## The bug (KNOWN_ISSUES #11) — see v23 README for the full matrix

With the user's flags (TP=2, `--enable-prefix-caching`, MTP k=4,
turboquant_4bit_nc, `--async-scheduling`, FULL_DECODE_ONLY graphs), ~7.5% of
~4.8k-token-prompt requests wedge inside the engine forever (client timeout,
request stuck `Running` with KV allocated until the client disconnect
triggers the abort, serve log and dmesg clean). 12-token prompts never
wedge; only MTP-off is clean (isolation matrix: BASE 6/80, mamba-all 2/80,
no-async 6/80, chunked 1/80, no-MTP 0/80).

## Why v23 (one-sided drain) failed

v23 added `torch.xpu.synchronize()` before the drafter on steps >= 512
tokens. Deployed and disproven: a live catch on v23 showed the same wedge
(run=1 frozen, ttft counted, success frozen), and py-spy native stacks gave
the definitive picture:

- **TP0 stuck BEFORE the barrier**: `_update_states_after_model_execute`
  (align-mode `.cpu()` D2H) — `ur_command_list_manager::appendUSMMemcpy`
  cannot even be *submitted*: this rank's in-flight queue never drains.
- **TP1 already PAST the barrier** (its stream drained fast — it is ahead),
  inside the drafter issuing the MTP head's vocab all-gather
  (`qwen3_5_mtp.py:476 -> _gather_logits -> tensor_model_parallel_all_gather
  -> ProcessGroupXCCL`).

A one-sided drain cannot fix this: each rank drains its **own** stream and
immediately proceeds, so the fast rank still enters the collective alone.

Deadlock cycle (#05 oneCCL peer-late family, prefill variant): fast rank
submits the head all-gather → its non-preemptible spin-wait kernels
monopolize the engines → they starve that rank's own earlier in-flight
collective sends → the slow rank's pending receives never complete → the
slow rank's stream never drains → its host blocks forever in the `.cpu()`
D2H submission before it can join the all-gather → eternal spin (no
competing queue work ⇒ no GuC preempt reset; both engines 100%, EU
stall-dominant; dmesg clean).

## The fix (v24)

True host-level rendezvous before any drafter work on steps with >= 512
scheduled tokens:

```python
if _SPEC_PREFILL_PEER_BARRIER and num_scheduled_tokens >= 512:
    torch.xpu.synchronize()
    _peer_group = _spec_peer_rendezvous_group()   # tp_group.cpu_group (gloo)
    if _peer_group is not False:
        torch.distributed.barrier(group=_peer_group)
```

The gloo `cpu_group` every vLLM `GroupCoordinator` already builds is a pure
host barrier (no GPU kernels — it cannot itself wedge). The fast rank now
**waits**; the slow rank's stream drains (its completion needs only sends
the fast rank already submitted, which are not blocked by the gloo wait);
both ranks enter the drafter with empty, symmetric queues. Decode steps
(`max_num_seqs * (k+1)` tokens — 320 at 64 seqs / k=4) skip the rendezvous
and keep the async-scheduling fast path; the cost is paid only on large
prefill steps (~2 s), where losing tail overlap is negligible.

## Validation (v24, user flags unchanged)

**Disproven.** Wedge catcher (sequential 4.8k prompts, hold-connection
sampling + py-spy):

- v24 plain: **wedge at request 17 of 300** — `run=1` frozen 120 s, ttft
  counted, success frozen. py-spy (native): TP0 AND TP1 both stuck in
  `_update_states_after_model_execute:1502` (the align-mode `.cpu()`
  `num_accepted_tokens.gpu[:num_reqs].cpu().numpy()`) inside
  `ur_command_list_manager::appendUSMMemcpy` — submission blocked, queues
  mutually clogged. Note the code path: `num_accepted_tokens` is the
  spec-VERIFY output — this is a decode/verify step, and the stall sits
  upstream of (before) the rendezvous, which was therefore never reached.
- v24 + `--mamba-cache-mode all`: **wedge at request 22 of 300**, same
  stack. `models/config.py:438-446` coerces `all` -> `align` for hybrid
  models without `supports_mamba_prefix_caching`, so this arm never really
  left align mode (this also re-opens the v22 matrix cell "B": its 2/80 vs
  BASE 6/80 was the same align mode twice — not a real delta).

What DID validate cleanly on the v24 boots (unchanged from v22/v23
behavior):

- Prefix caching + MTP: `cargame_prefix.py` — warm hits exactly
  `(floor(n/4096) - 1) * 4096` (4096 @ 8.6k prompt, 8192 @ 12.3k), warm
  TTFT 2.3-3.2x faster than cold.
- Canonical `"Write a html car game."` (temp 0.3, top_k 20, top_p 0.95,
  max_tokens 4096, streaming): full HTML-canvas game, TTFT 0.16 s, clean.
- No xe resets on any boot; dmesg clean throughout all wedges.
