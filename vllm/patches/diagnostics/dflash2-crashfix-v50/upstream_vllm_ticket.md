# Ticket (draft) — vLLM: v1 async engine usage-less silent stream end leaks the
request; unresolved num_output_placeholders diverge from real token history and
drive an out-of-bounds gather in the worker (fatal on XPU: SYCL vectorized-gather
assert)

Status: **DRAFT 2026-09-06, NOT YET POSTED** (posting gated on F8a validation
cell).

---

## Title

[Bug] v1 async engine: usage-less silent stream end leaks the request; unresolved
num_output_placeholders diverge from real token history and drive an out-of-bounds
gather in the worker (fatal on XPU: SYCL vectorized-gather assert)

## Environment

- vLLM 0.21.1.dev0+gad7125a43 (XPU build, Intel fork stack) + vllm-xpu-kernels
- torch 2.11.0+xpu
- 2x Intel Arc Pro B70 (Battlemage BMG G31), TP=2
- Model: Qwen3.8-27B fp8, `--kv-cache-dtype turboquant_4bit_nc`,
  `--block-size 512`, `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
- DFlash2 drafter, `num_speculative_tokens=7` (out-of-tree drafter; the placeholder
  accounting path cited below is stock v1 async-scheduler code — MTP k=1 uses the
  same `_update_after_schedule` / `_update_request_with_output` path)
- Prefix caching ON, async scheduling ON

## Minimal repro

No single-request deterministic repro exists. The sequence that produced 2-of-2
fatal engine deaths on a single container (plus 2 sub-fatal zombie stream-ends on
separate containers): fresh boot, full benchmark battery (prefill/decode at 2k/16k/65k
contexts), bench3 + conc8 concurrency runs, 3 coherence probes, one 8192-token
streaming chat completion (prompt `Write a html car game.`, server-default sampling,
`stream: true`, `include_usage: true`), then repeated long-decode rounds (8192-token
streaming completions). After this sustained load, the next `/v1/chat/completions`
stream with `stream: true, include_usage: true` silently ends well below
`max_tokens` (796 / 1,650 / 2,836 tokens in the three best-instrumented events)
with no usage chunk and no `finish_reason`; the request remains RUNNING
server-side and the engine eventually dies after further solo decoding.

## What we established (each point from a dedicated boot or salvage)

1. **The stream ends silently with no usage chunk and no finish_reason.** On
   `/v1/chat/completions` with `stream: true, include_usage: true`, the SSE stream
   terminates at some token count well below `max_tokens`, with no usage chunk and no
   `finish_reason` field. No APIServer error line appears in the server log for this
   event. The client-side connection just closes.

2. **The engine-side abort fires only on CancelledError/GeneratorExit.** Per
   `vllm/v1/engine/async_llm.py` (~line 591-595), the `generate()` task abort
   handler only catches `CancelledError` and `GeneratorExit`. A stream that ends
   through any other exit path sends no abort to the engine. The `check_stop`
   length guard had not tripped (real token count well below `max_tokens`), so the
   request continues decoding alone server-side.

3. **The scheduler keeps scheduling the request every step.** In the async scheduler
   (`vllm/v1/core/sched/async_scheduler.py`), `num_output_placeholders` accrues in
   `_update_after_schedule` (~line 59) at `(num_sampled_tokens_per_step + spec
   width)` per scheduled step. It is only resolved in
   `_update_request_with_output` (~line 81) by `-= len(new_token_ids)`. When outputs
   stop being applied while scheduling continues — the unresolved-placeholder
   signature; why the emission leg stalls is open question 1 below — placeholders accumulate
   monotonically.

4. **Cached num_output_tokens sent to workers = emitted + placeholders.** Per
   `vllm/v1/core/sched/scheduler.py` (~line 1060-1063), the
   `CachedRequestData.num_output_tokens` is `req.num_output_tokens +
   req.num_output_placeholders`. After ~2,400 scheduled steps the cached count
   reached 22,688 (8 x 2,836) while the real emitted token history was ~2,836
   tokens. The `max_tokens` was 8,192 — 22,688 real tokens would be physically
   impossible.

5. **The worker persistent-batch alignment indexes past the real token-history
   arrays.** The worker computes `end_idx = num_prompt_tokens[i] + num_output_tokens[i]`
   and gathers token IDs from arrays sized by the real token history. The
   placeholder-inflated `num_output_tokens` overruns the array, producing an out-of-
   bounds gather.

6. **On XPU this is a fatal SYCL assert, 1,024 repetitions.** The first fatal event
   (salvage_010017.log, 1,885 lines) shows 1,024 log lines of
   `ind >= 0 && ind < ind_dim_size_ && "vectorized gather kernel index out of bounds"`
   (torch-xpu-ops IndexKernelUtils.h:63), starting at log line 790 and constituting
   the first anomaly in the entire log. Everything between boot and that point is
   clean health GETs (zero warnings). The assert kills Worker-0, then the
   EngineCore receives `RuntimeError: cancelled` via shm_broadcast, and the server
   exits.

7. **On CUDA this class would likely be silent corruption rather than a loud assert.**
   The XPU SYCL runtime asserts on index OOB; CUDA would index the same out-of-bounds
   memory silently. The defect is not XPU-specific — only the assert presentation is.

8. **The death dump shows exactly one zombie request.** Scheduler state at death:
   request `chatcmpl-...-8f028749`, 57-token prompt (the loopscan cargame request),
   `num_computed_tokens=22,745`, `num_output_tokens=22,688` in
   `scheduled_cached_reqs`, `new_block_ids=[None]`,
   `scheduled_spec_decode_tokens=[-1]x7` (empty-draft padding),
   `finished_req_ids=[]`. Real emitted ~2,836 tokens; ~19,852 unresolved
   placeholders. The battery cargame (6-token raw prompt, 512 tokens with usage
   chunk) is exonerated.

9. **Incidence: 4 events across approximately 17 cells.** Always after sustained
   battery-scale load on one container. Never in short cells. Two were fatal (full
   engine death); two were sub-fatal (zombie stream-end + request-scoped HTTP 500).
   The v50 overlays are innocent (no finish/placeholder code touched by the overlays).

10. **No watchdog or timeout exists anywhere in the v1 finish path.** There is no
    mechanism to detect or abort a request that keeps scheduling after its client-side
    stream has ended. The `check_stop` length guard is the only eventual termination,
    but under async scheduling with speculative placeholders the engine can die long
    before that point.

## Mechanism

The full chain:

```
Client stream ends silently (no usage chunk, no finish_reason, no error)
  --> No CancelledError/GeneratorExit reaches async_llm.generate()
  --> Request remains in RUNNING status, kept scheduled every step
  --> _update_after_schedule: num_output_placeholders += (1 + spec_width) each step
  --> _update_request_with_output never called (no consumer, no new_token_ids)
  --> Placeholders accumulate monotonically (~19k after ~2,400 steps)
  --> CachedRequestData.num_output_tokens = real + placeholders sent to workers
  --> Worker persistent-batch alignment indexes end_idx past token-history arrays
  --> Vectorized gather OOB (SYCL assert on XPU; silent corruption on CUDA)
  --> Worker process death, engine death
```

The placeholder accrual rate and the absence of any watchdog are both in the stock
v1 async-scheduler code. The speculative drafter (DFlash2 or MTP) is not the cause —
it is an ingredient that sets the per-step accrual rate. The root enablers are:

1. A stream exit path that does not trigger `CancelledError`/`GeneratorExit` in
   the engine's `generate()` task.
2. No scheduler-side bound or watchdog on `num_output_placeholders` divergence.

## Suggested fix (two independent layers)

**Layer 1 — Serving layer:** detect a usage-less, non-cancel stream end and abort
the request. Currently `async_llm.py` only aborts on `CancelledError`/`GeneratorExit`.
A stream that ends through any other path (network drop, ASGI disconnect, etc.)
should be treated as a client disconnect and the request should be force-finished
(ABORTED or FINISHED_ABORTED).

**Layer 2 — Engine-side guard:** bound the `num_output_placeholders` divergence.
Either assert early with a clear message when `num_output_placeholders` exceeds a
small threshold (e.g., 64 — a monotonic climb of >64 placeholders is never correct),
or clamp/force-finish the request. Additionally, if `num_output_tokens` exceeds
`max_tokens + num_speculative_tokens + margin`, force-finish immediately. This
prevents any single request whose outputs stop resolving from driving a worker-side
OOB gather, regardless of the serving-layer fix.

A separate improvement would be to add a watchdog or timeout in the v1 finish path,
so that a request that stops producing output for N steps is force-finished even if
the placeholder counters happen not to diverge.

## Files and line references

- `vllm/v1/engine/async_llm.py` ~line 591-595: `generate()` abort handler — only
  catches `CancelledError`/`GeneratorExit`
- `vllm/v1/core/sched/async_scheduler.py` ~line 59: `_update_after_schedule` —
  placeholder accrual `num_output_placeholders += (num_sampled_tokens_per_step +
  cur_num_spec_tokens)`
- `vllm/v1/core/sched/async_scheduler.py` ~line 81: `_update_request_with_output` —
  placeholder resolution `num_output_placeholders -= len(new_token_ids)`
- `vllm/v1/core/sched/scheduler.py` ~line 1060-1063: `CachedRequestData` construction
  — `num_output_tokens = req.num_output_tokens + req.num_output_placeholders`
- Worker persistent-batch alignment (gpu_model_runner.py): `end_idx = num_prompt_
  tokens[i] + num_output_tokens[i]` — the gather that indexes past the array

## Workaround (applied locally, not proposed as upstream fix)

A scheduler-side runaway guard in `async_scheduler._update_after_schedule` that
force-finishes (`FINISHED_ABORTED`) any request whose `num_output_placeholders`
exceeds 64 or whose `num_output_tokens` exceeds `max_tokens + num_speculative_
tokens + 8`. Fires approximately 9 scheduled steps after outputs stop
resolving. Configurable via `VLLM_V50_PLACEHOLDER_LIMIT=0` to disable.

Validated live (2 independent firings in one replay cell, 2026-09-06): the class
recurred deterministically in two consecutive long-decode rounds; the guard
caught both at `placeholder-overflow 72 > 64`, reaped each request in <5 s, and
the engine survived both events (health 200, zero SYCL asserts — the unguarded
engine died with 1,024 asserts in the same sequence), and served fresh requests
correctly while a zombie's stream was hung. Limitation confirmed: force-finishing
is scheduler-side only — the zombie request's own client-facing SSE stream never
terminates (the finish marker cannot traverse the stalled output leg), so
clients without read-timeouts hang on that one request until they time out.
This is why the serving-layer fix (Layer 1) is still required.

## Open questions

1. What produces the usage-less silent stream end at the ASGI/serving layer? Is it a
   client disconnect, an HTTP/2 RST, a FastAPI streaming-generator exit path, or a
   uvicorn transport behavior? The server log shows no error line for the event.

2. Does stock non-fork vLLM reproduce the zombie stream-end if a plain network cut is
   made mid-stream (e.g., kill the client process abruptly, or `curl` with a SIGKILL)?
   This would confirm that the missing abort path is the root trigger, independent of
   our specific serving configuration.

3. Is the placeholder divergence observable via existing vLLM metrics? If not, adding
   a `num_output_placeholders` metric would make this class of defect detectable in
   production before it kills the engine.

4. Is there any non-async path (e.g., the standard synchronous scheduler) that also
   lacks a watchdog for requests that stop producing output, or is this particular
   to the async-scheduling placeholder accounting?