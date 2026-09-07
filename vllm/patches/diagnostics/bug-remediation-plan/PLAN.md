# Bug remediation plan: serving, speculation, cache/state and failure handling

Date: 2026-09-07. Status: researched plan; no production fixes deployed. Scope: this llm-scaler XPU codebase and its v1.2.5/v1.2.6t2 overlay lineage, nonspec/MTP/DFlash2, supported KV modes and128k/262k contexts. Performance-only work remains in the [latency/throughput plan](../long-context-plan/LATENCY_THROUGHPUT_PLAN.md).

## 1. Decision and evidence standard

**Fix lifecycle and configuration-contract defects first; isolate token/state corruption next; change kernels and adaptive speculation only behind verified invariants.** A guard that prevents an engine crash is containment, not proof that the originating corruption is fixed. A periodic output tail is a real observed symptom, but its cause must be established before changing quantization or sampling.

This pass used the prior audits, freshly captured installed scheduler/async helper source, current host logs, relevant Python/vLLM primary documentation and one CPU-only reproduction. The external benchmark was still running on v1.2.5 in the MTP/FP8 deep phase at collection. No GPU inference, server restart or kernel modification was initiated by this review. [Host snapshot](evidence/host-snapshot.txt).

Confidence labels:

- **Reproduced helper defect:** demonstrated with the captured function in isolation; production impact still needs an integration test.
- **Code-confirmed defect/risk:** an inspected path violates an intended contract or lacks a needed constraint; occurrence may require a particular configuration.
- **Observed failure:** raw/historical evidence establishes the symptom, not necessarily the root cause.
- **Open hypothesis:** needs a discriminating experiment before a fix is justified.

The planning API is unavailable in this session; scope, discovery, focused reproduction, synthesis and verification were tracked in the conversation. This document is the new remediation artifact, not a claim of exhaustive verification of every repository line or public issue.

## 2. Prioritized issue register

| ID | Priority / evidence | Problem | Required action |
|---|---|---|---|
| BUG-01 | P1, reproduced helper defect | Multi-iterator merge returns before child cleanup completes | Cancel, await completion, then close generators; retain cancellation semantics |
| BUG-02 | P1, code-confirmed | DFlash emission env affects other scheduler modes | Resolve method-specific effective width once; reject irrelevant settings before worker launch |
| BUG-03 | P0, observed failure + contract evidence | Proposer-only adaptive width can wedge async full-graph execution | Keep feature disabled; scheduler-owned width protocol and transition tests |
| BUG-04 | P0, observed symptoms | Invalid/NaN token-state path, zero emissions and runaway placeholders | Trace first corrupt state; explicit invalid-result handling and exactly-once terminal delivery |
| BUG-05 | P0 for affected workloads, observed | DFlash periodic tails and no final answers | Paired target-only/spec replay, raw reasoning/token evidence, verifier/state diagnosis |
| BUG-06 | P1, code-confirmed missing integration | DSpark confidence knob does not populate DFlash2 selector confidence | Reject unsupported combination or implement calibrated selector confidence with safe scheduling |
| BUG-07 | P1, latent code risk | Ragged FP8 fallback catches every Exception | Catch only recoverable pre-dispatch conditions; propagate device/OOM failures |
| BUG-08 | P1, latent code risk | TQ fallback scratch key omits dimension/ownership | Shape/dtype/stream-safe reusable workspace and lifecycle tests |
| BUG-09 | P1, observed support mismatch | Dtype enum, backend declarations and actual kernels disagree | Executable capability validation before allocations and collectives |
| BUG-10 | P0 reliability, observed | DEVICE_LOST and silent worker timeout conflated | Preserve first error, rank timelines and bounded recovery; independent failure classes |
| BUG-11 | P1, evidence integrity | Benchmark counts/identity can misattribute speed or success | Mandatory usage, protocol completion, fixed identity, correct denominators |
| BUG-12 | P2 performance, open causal gaps | GMU cliff, TQ prefill overhead, deep anomaly/fairness | Profile and discriminate; do not promote speculative causal stories as fixes |

P0 means a release blocker when the affected feature/workload is enabled. It does not mean the default deployment has reproduced every listed failure.

## 3. Concrete fixes and reproductions

### BUG-01: merged stream cleanup does not wait for children

Evidence: installed [async_utils.py](evidence/async_utils.py), `merge_async_iterators`, approximately302–320. The multi-input finally block calls `f.cancel()` then immediately `await it.aclose()` under `suppress(BaseException)`. Cancellation schedules work; it does not wait for the outstanding `anext` task to stop. Closing a still-running async generator can fail, and that failure is suppressed.

The [CPU-only reproduction](evidence/repro_iterator_cleanup.py) extracts that exact function with AST, gives one generator delayed finalization, closes the merge and checks cleanup. [Result](evidence/repro-result.json): cleanup was false when close returned, true later. This proves the helper lacks a synchronous cleanup-completion guarantee; it does not prove leaked GPU memory or a permanent engine request.

Proposed patch:

1. Snapshot outstanding `anext` tasks and child generators; cancel all outstanding tasks first.
2. Await their completion with collected outcomes before calling `aclose` on children. Close all relevant children, including one whose task raised and was already removed from the mapping.
3. Preserve the original cancellation/error; do not let one cleanup failure skip the rest. Log unexpected cleanup errors with iterator/request identity.
4. If serving imposes a cleanup timeout, retain ownership of a supervised cleanup task after timeout and report incomplete cleanup. Do not claim success or discard the task. Repeated cancellation must not orphan it.
5. Verify the single-input fast path and serving-layer finalizers follow the same ownership policy; avoid indiscriminate suppression of all BaseException.

Regression cases: zero/one/multiple iterators; one child raises; child finalizer awaits; early consumer break; cancellation during `anext` and during cleanup; repeated cancellation; one already-finished child. Assert cleanup before successful close returns, or explicit supervised-timeout status. Integration: disconnect chat/completion clients and verify engine request/block release and healthy next request.

Python's [asyncio task documentation](https://docs.python.org/3/library/asyncio-task.html) explains cancellation and retaining task references. That supports task ownership design, not a blanket instruction to shield every coroutine. The reproduction uses the host's Python runtime; deployed integration must also run in the image's Python3.12 environment.

### BUG-02: emission-width environment is not scoped to DFlash2

Evidence: captured [async_scheduler.py](evidence/async_scheduler.py), lines27–36. Scheduler always reads `DFLASH2_EMIT_K`, checks against `self.num_spec_tokens` and uses it for placeholders, without a method check. For nonspec with zero spec tokens, a leftover positive value fails validation. For MTP4 with value2, the scheduler constructs two placeholders even though this variable belongs to the DFlash2 proposer policy. End-to-end MTP failure is not reproduced here, but the cross-mode configuration effect is code-confirmed.

Fix: a central validated config resolves `(method, native draft width, emitted width, scheduled width)` before any worker startup. An irrelevant DFlash setting should produce a clear configuration error, or an explicitly documented ignored-setting warning; never silently change MTP. The scheduler and proposer consume the resolved config rather than independently parsing environment variables. Validate checkpoint block width separately from emitted width.

Tests: nonspec/MTP/DFlash1/DFlash2 × unset/empty/0/1/2/7/negative/noninteger/out-of-range. Verify no worker/collective starts on invalid config. Ensure every consumer sees the same effective width and existing default behavior is unchanged. Test stale env during sequential benchmark lane switches.

### BUG-03: dynamic-width changes violate async/state contracts

The [v53 post-mortem](../../v125-fixes/PLAN.md) records a silent300-second timeout after proposer narrowing; the current scheduler initializes placeholders at startup. Precise blocked device instruction remains unisolated. Containment: leave `VLLM_V53_L2_UNSTABLE=0` and context-width knob unset.

Fix: scheduler publishes width with step/request generation identity; runner, graph selection, proposer, verifier, placeholder accounting and GDN commit consume it consistently. Start with one width per batch/step. Drain old-width in-flight steps before transition. K0 is a real nonspec path; re-enabling spec needs explicit draft catch-up, otherwise use a one-way per-request transition. Preserve DFlash2's native anchor+seven-mask layout even when emitted verification width differs.

Tests: all supported width transitions, accept0…K, EOS within proposed block, output budget clipping, prefill tails, preemption, abort while old output is pending, mixed request lengths, graph/eager comparison and both TP ranks. Assert outstanding placeholders equal submitted-but-unresolved work, never merely clamp discrepancies away. Current [upstream Dynamic SD limitations](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/) require a validated port for this MRv1/full-graph deployment.

### BUG-04: zero emissions and sentinel clamps contain rather than explain corruption

Captured scheduler comments and v52 history describe invalid sampler rows, empty parsed emissions and placeholder growth. Its detector skips structured-output requests because empty emissions may be legitimate; detection also runs only when the sampled-token/index containers are truthy. A globally empty result therefore needs an explicit protocol classification rather than silently bypassing observation.

Fix design:

- Worker output carries an explicit status: valid decode, valid prefill/no-emission, grammar-deferred, cancelled/stale step, or invalid result. Do not infer NaNs solely from an empty list or a missing request index.
- At the first invalid row, record bounded diagnostic tensors/metadata: step ID, valid vocabulary range, finite-logit flags, accepted length, positions, GDN block/copy state and KV mapping. Correlate both ranks.
- Invalid token IDs must never become committed output, prefix hashes or state. If a clamp is necessary to make speculative gather memory-safe, carry a validity mask and prevent the substituted token from being committed. Fail the request explicitly if recovery cannot preserve semantics.
- Define exactly-once finish/abort delivery even when engine becomes idle or request was not scheduled. Clear per-request counters/maps on every terminal path. Never expose fabricated valid output to hide an internal error.
- Validate finite/nonnegative guard settings and document what a timeout/strike proves. A grace period alone cannot prove no valid slow request will be aborted.

Tests: synthetic all-NaN/Inf logits, invalid/sentinel IDs, missing index, empty batch, grammar-deferred row, valid prefill, cancelled stale output, delayed output beyond grace, idle finish and duplicate finish. Long tests hit512/1024/2048-aligned boundaries±1, with prefix caching and accept/reject patterns. Check terminal reason, counters and memory recovery, not just engine survival.

### BUG-05: periodic output must be separated from budget exhaustion

The prior [FP8 result](../latest-image-audit/evidence/loop8-partial.out) detected one periodic tail in four selected prompts. The later [TQ4 result](../long-context-plan/evidence/latency-throughput-host-snapshot.txt) detected three periodic tails in four; neither establishes a quantization-specific cause. Two kinds of failure must not share the label LOOP: repetitive output versus coherent reasoning cut by max_tokens.

Fix workflow: preserve exact prompt/rendered template, sampler settings, full reasoning/content, raw token IDs where possible and finish status. Replay the same teacher-forced prefix through nonspec/MTP/DFlash at identical target weights/KV/SSM. Locate first layer/logit/state divergence, then free-run controlled continuations. Repeat with auto KV and FP32 SSM as separate controls; these still use FP8 target weights.

If spec-only, inspect rejected-token rollback, anchor/position/selector mapping and target sampler. If compressed-KV-only, inspect scale/layout/rotation and saturation. If FP16-SSM-only, test recurrence precision and copy timing. If target-only reference also fails, investigate model/weight precision or prompt behavior. Repetition penalties and arbitrary stop strings alter generation semantics and are not correctness repairs.

Tests: temperatures0/0.7/1.0 with recorded sampling parameters;4k/8k/16k output budgets; code/JSON/tool calls and original loop prompts; short and128k/near262k inputs. Distributional sampler tests use controlled logits and many samples; individual seeded generation equality is insufficient to establish losslessness. [vLLM losslessness discussion](https://docs.vllm.ai/en/latest/features/speculative_decoding/) distinguishes algorithmic guarantees from floating-point/batching variation; it does not certify this patched port.

### BUG-06: advertised adaptive-confidence integration is missing for DFlash2

The [captured base proposer](../long-context-plan/evidence/dflash.py) clears confidence before proposal and truncates only when confidence exists. [DFlash2](../long-context-plan/evidence/dflash2.py) overrides sampling with a selector walk that does not populate that field. Thus the inspected ordinary selector path does not implement the proposed DSpark confidence policy.

Fix first: reject/warn for this unsupported combination and expose the resolved policy in startup diagnostics. Longer-term: calibrated selector-confidence output, explicit instrumentation proving it fires, and BUG-03's safe width contract. Never add arbitrary post-propose dynamic truncation as a shortcut. Tests cover zero/low/high confidence, no-confidence fallback, request-local widths and no unwanted D2H synchronization.

### BUG-07 and BUG-08: fallback paths need explicit error and workspace contracts

[FP8 ragged path](../../v125-fixes/flash_attn_v53.py), approximately1424–1472, broadly catches Exception around allocations/kernel work and falls through to another route. A device/OOM exception is not a harmless shape surprise; continuing may mask the first error or worsen failure. Asynchronous device faults may surface after the try block, so adding this catch does not make the operation safe.

Fix BUG-07: prevalidate supported shape/layout; narrow catches to documented recoverable cases before submission. Propagate device/OOM failures with original context and request/step identity. Ensure no partial result is committed. Test injected allocation/kernel failures and valid ragged shapes; do not insert a synchronizing check in every hot-path call just to catch faults locally.

[TQ continuation fallback](../../v125-fixes/turboquant_attn_v53.py), approximately1559–1570, keys `_DEQUANT_BUFS` by device and checks capacity/head count, but not complete head dimension/dtype/stream ownership. Current serialized target use may be safe; multiple model/draft shapes or future concurrent execution create a latent reuse risk.

Fix BUG-08: workspace key/spec includes full shape capacity, dtype, device and execution owner; guard reuse by stream events when concurrency exists. Release ownership on model lifecycle transitions; bound retained capacity. Test alternatingD128/D256, different head counts, growth/shrink, model unload/reload, capture and concurrent streams. Distinguish reusable reserved memory from a live allocation leak. Fused continuation attention is a separate optimization, not necessary to repair workspace identity.

### BUG-09: unify dtype capability validation

The [latest-image audit](../latest-image-audit/REPORT.md) lists15 dtype enum values; enum membership and tensor mapping do not imply correct XPU kernel support. Historical e5m2 guard bypass exposed further backend failures. FP8 paths are exercised despite a backend advertised list containing only auto/FP16/BF16.

Fix: one capability resolver checks target/draft architecture, head dimensions, packed layout/scales, model quantization, graph mode, hybrid block requirements and actual installed kernels. Produce an actionable unsupported error before GPU initialization. Do not bypass only a guard. Test all enum values across nonspec/MTP/DFlash: expected rejection is a passing rejection test, not a serving certification. Every supported mode must pass prefill, continuation, q1, verify, graph replay and semantic smoke.

### BUG-10 and BUG-11: preserve failures and trustworthy measurements

DEVICE_LOST during v125 restore and silent v53 timeouts are different signatures with similar downstream RPC errors. Preserve first rank error, device/kernel logs, outstanding step IDs, collective sequence and resource state. Keep bounded diagnostics; do not repeatedly relaunch into an unhealthy device or silently retry failed lanes. Any driver/firmware remedy needs its own pinned-version experiment; do not change watchdog timing merely to hide a symptom.

Benchmark fixes: final usage required; DONE/error/finish accounted for; zero-token result is not success; count tokens rather than SSE events/characters; compare image/container ID before/after each batch. In the existing [master script](../latest-image-audit/evidence/master127.sh), post-phase code accumulates characters into ctoks and uses a constant256 numerator. Replace it with actual completion usage and label true token timing versus event timing. Reject attribution if the server changed or fallback flags differed. Store actual prompt tokens rather than128k/262k labels alone.

Tests use protocol fixtures for missing usage, HTTP200 with error event, partial stream, multi-token speculative chunks, zero output, explicit cancellation and container replacement. Device-loss tests begin with injected error paths; real GPU-fault reproduction needs an isolated recovery-capable lane, not the concurrent benchmark.

### BUG-12: keep unresolved performance mechanisms open

Separate fixes from experiments: tiled TQ prefill has a code-established conversion/materialization opportunity; the FP8 GMU cliff, DFlash depth anomaly and per-client fairness attribution need timelines. A smaller pool also frees memory, so it cannot alone discriminate allocator pressure. Lane-total stalls cannot locate decode stalls. Consult the detailed [long-context plan](../long-context-plan/PLAN.md) for controlled experiments. Do not merge a performance patch on an unmeasured causal assumption.

## 4. Implementation sequence and review units

| Change set | Contents | Merge gate |
|---|---|---|
| R1 | BUG-01 iterator cleanup; supervised task lifecycle | CPU async unit suite and client-disconnect integration |
| R2 | BUG-02 config scoping, BUG-06 unsupported policy detection, BUG-09 capabilities | Config cross-product; clean pre-worker failures; default baseline unchanged |
| R3 | BUG-11 harness identity/protocol fixes | Fixtures and raw-result reconciliation; no invented throughput |
| R4 | BUG-04 explicit output-status/terminal protocol and diagnostic validity masks | State-machine/property tests, invalid-output injection, exactly-once finish |
| R5 | BUG-05 isolated verifier/state/numerical repair based on reproduced divergence | Reference replay plus quality/loop corpus; retain original failing fixture |
| R6 | BUG-07 exception narrowing, BUG-08 workspace ownership | Error injection, alternate shapes, capture and concurrency |
| R7 | BUG-03 scheduler-owned width; enabled only after R4/R5 | Full async/state transition suite; rollback to static behavior |
| R8 | BUG-10 lifecycle observability and BUG-12 isolated performance changes | Repeated healthy boot/teardown, stage profiles and full release matrix |

These are separate reviewable patches/images, not one large overlay. Existing guard fixes stay enabled until replacements demonstrate equivalent containment and correct behavior. Keep old kernel routes and static widths available for rollback. No production hot-swap during comparative runs.

## 5. Regression and release matrix

Primary lanes: nonspec/MTP4/DFlash7 × FP8_e4m3/TQ4nc, same user baseline flags; auto-KV and FP32-SSM controls where capacity permits. Add other servable dtypes only after capability gates. Pin image digest, source manifest, model/tokenizer/template and environment.

Required scenarios:

- Short structural tests, natural EOS, streamed/nonstreamed chat/completion, tool JSON and grammar-deferred output.
- Exact131072 prompt+256 output and261888+256=262144 total. Long generation258048+4096 and253952+8192. Test limit−1/exact/+1 and fully rendered chat budgets.
- C1/C2/C4 at long contexts and mixed C8/C16; distinguish queued clients from resident requests. Admission success does not imply all requests fit simultaneously.
- Prefix reuse/eviction, unique per-client synthetic facts, repeated cancellation, abort during prefill/verify/decode, slow consumer, engine-idle final delivery and graceful restart.
- Page/hybrid boundaries, accept/reject extremes, graph/eager validated reference, corrupted/sentinel outputs, stale/duplicate worker messages.
- Natural long-context retrieval, multi-hop joins, code tests and periodic-tail corpus. Throughput filler cannot establish correctness.

Release gates: zero silent corruption, invalid committed IDs, cross-request leakage, orphan requests, duplicate terminal outputs or device resets in the required suite. Clean expected rejection for unsupported configurations. No new loop/quality regression; disclose numerical tolerance and sampling uncertainty. Stable live allocation/block counts after warm-pool stabilization and repeated abort cycles. At least2h screening and24h mixed release soak. Retain all failures in the denominator.

Performance is a secondary bug-fix gate: no unexplained regression above5% in protected workloads, measured with repeated matched runs; throughput optimizations follow their separate promotion plan. Do not demand byte-identical outputs across all precision/batching changes as the sole correctness criterion.

## 6. Verified work and remaining limits

Completed in this research pass: captured deployed scheduler/helper source, inspected cross-mode env handling and failure guards, reproduced delayed child cleanup with the exact captured merge helper, reconciled loop evidence, and wrote this plan. The reproduction succeeded once and is archived with output; no fixes have been implemented or load-tested.

Remaining: CPU regression of patched cleanup, end-to-end stale-env reproduction, explicit-state protocol implementation, raw loop replay, all-dtype/latest-image coverage, long concurrent semantic tests and device-fault causality. Broad public issue enumeration and every code path are not exhausted. The highest-value next action is R1–R3, followed by state/sampler diagnosis; another broad audit does not substitute for those reproductions.
