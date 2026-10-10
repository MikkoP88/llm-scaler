# P83 — Upstream Official vLLM MTP Research (fork root-issue mapping)

**Source of truth:** vLLM `0.23.1rc1.dev515+g68ee8300a` — read directly from the
A5000 reference host's `vllm-mtp` container (`vllm/vllm-openai:cu129-nightly`),
the lane that SURVIVES the deep shape drafting-active at 2.66 tok/step (fork
parity ±2%). This is the "most popular official MTP vLLM codebase"
(vllm-project/vllm) in the exact build our reference evidence comes from.
(Web quota exhausted → read from the installed package, better than the repo:
it is the code that actually ran.)

## 1. Upstream architecture — five layers

### Layer 1 — Scheduler (`v1/core/sched/scheduler.py`)
- **Per-STEP dynamic K**: `num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]`
  (line ~1078) — spec width chosen by *runtime batch size* each step, default
  `self.num_spec_tokens` when no schedule configured.
- **Dynamic SD schedule** (`spec_decode/dynamic/utils.py`, 148 lines): config
  `num_speculative_tokens_per_batch_size: [(range_start, range_end, K), ...]`
  → dense 1-indexed lookup `dense_schedule[batch_size]`, gaps/tail carry the
  previous K forward, every K clamped by `num_speculative_tokens`, **K=0
  legal** (validated `>= 0`).
- **Spec tokens are budget-accounted tokens** (lines 586-601):
  `num_scheduled_spec_tokens = num_new_tokens + num_computed_tokens −
  num_tokens − num_output_placeholders`; if they don't all fit,
  `spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]` — **partial
  spec scheduling under budget pressure**; a request with no spec tokens this
  step is simply absent from `scheduled_spec_decode_tokens` (dict[str, list]).
- `vllm_config.num_speculative_tokens` is a top-level config field.

### Layer 2 — Runner routing (`v1/worker/gpu_model_runner.py`)
- `use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0`
  (line 2161): if false → `spec_decode_metadata = None` → **plain decode step,
  zero special-casing**. Spec on/off per step is metadata-None, not a gate.
- `_calc_spec_decode_metadata` (line 2191) builds the ragged metadata from the
  per-request dict.
- **`use_async_spec_decode`** exists natively (line 635) with
  `deferred_spec_decode_corrections` (1192-1493) — the fork's R2 track
  (async drafter, §11.7 bars) is already engineered upstream.
- Proposer classes selected at init (lines 179-195): eagle, draft_model,
  medusa, ngram(+gpu), suffix, dflash, gemma4, step3p5, extract_hidden_states,
  custom_class.

### Layer 3 — Ragged metadata (`spec_decode/metadata.py`, 66 lines)
```python
draft_token_ids: torch.Tensor        # [num_tokens]
num_draft_tokens: list[int]          # [batch_size]  ← PER-REQUEST widths
cu_num_draft_tokens: torch.Tensor    # cumsum ragged layout
cu_num_sampled_tokens: torch.Tensor
target_logits_indices / bonus_logits_indices / logits_indices
```
Mixed widths in one batch (incl. 0-draft members) are the native
representation — cumsum ragged, no padding, no static batch-wide width.

### Layer 4 — Proposer engine (`spec_decode/llm_base_proposer.py`, 1,765 lines)
`SpecDecodeBaseProposer`; MTP = `EagleProposer` (22 lines,
`pass_hidden_states_to_model=True`) — Qwen3-Next/GLM-style MTP heads ride the
EAGLE interface. `propose()` (459-705) properties:
- **K=0 native**: "No draft tokens requested (e.g. Dynamic SD decided K=0).
  The prefill forward pass above already ran to keep the drafter KV cache in
  sync" → returns `torch.empty(batch, 0)`.
- **K=1 early exit** fast path.
- **GPU-resident loop**: preallocated device buffers
  (`self.input_ids[:batch_size] = input_ids`), `draft_token_ids.int()` stays
  on device, positions from buffers; CPU shadows explicitly invalidated
  (`common_attn_metadata._seq_lens_cpu = None`) to avoid H↔D syncs.
- **`num_rejected_tokens_gpu`**: rejection counts stay on GPU;
  `seq_lens -= num_rejected_tokens_gpu` on device.
- **`constant_draft_positions`** (Gemma4 MTP): attn metadata built once,
  reused across loop iterations.
- **`_share_mtp_indices`**: MTP indexer top-k computed at step 0
  (`set_skip_topk(False)`), steps 1+ reuse via shared buffer
  (`set_skip_topk(True)`) — MTP-specific indexer dedup.
- Padded-vs-ragged execution decided per call:
  `_determine_batch_execution_and_padding()` (line 1676), cudagraph-aware.

### Layer 5 — Draft model lifecycle (`spec_decode/draft_model.py`, 88 lines)
- `_raise_if_draft_tp_mismatch`: draft TP must equal target TP (compile-cache
  corruption hazard documented).
- Draft loads via `_create_draft_vllm_config` (`model_config=spec.draft_model_config`,
  `quant_config=None`) + `get_model(prefix="draft_model")` under
  `set_model_tag("draft_model")`.
- `validate_same_kv_cache_group` (proposer line 1591): draft KV pool must map
  to the same cache group as target.

## 2. Fork root-issue mapping (conviction laws → upstream answer)

| # | Fork defect (measured law) | Upstream mechanism that solves it |
|---|---|---|
| 1 | Batch-wide STATIC `num_spec_tokens` gate @ `gpu_model_runner.py:945`; mixed 0-draft batches FATAL (v52j/t3g/t3i) | Per-request `num_draft_tokens: list[int]` ragged metadata; 0-draft members legal; spec absent = plain step |
| 2 | 0-row/1-row spec widths FATAL | `propose()` K=0 returns empty after KV-sync forward; K=1 early-exit path |
| 3 | Plain-decode routing requires the static gate (all-or-nothing) | `use_spec_decode = len(dict) > 0` per step; metadata=None = plain decode |
| 4 | D2H parks: `rejection_sampler.py:270 .cpu()` bookkeeping sync | `num_rejected_tokens_gpu`; CPU shadows nulled instead of synced |
| 5 | Drafter 6 ms/step deep (R1); loop on host | GPU-resident propose loop, buffers preallocated, no per-step H2D/D2H |
| 6 | No FITS-IN-DRAFTER context gate (A5000 lever) | Spec tokens budget-sliced per request; scheduler can schedule a request without spec (absent from dict) |
| 7 | Spec × TP>1 livelock (KNOWN_ISSUES #11, forced eager) | `_raise_if_draft_tp_mismatch` + TP-equal draft; parallel drafting params (`_init_parallel_drafting_params` line 331) |
| 8 | Fixed K=4 regardless of load | Dynamic SD: `dense_schedule[batch_size]` per-step K, small-batch→high-K policy |
| 9 | Async drafter = R2 roadmap item | `use_async_spec_decode` native + `async_scheduler.py` + deferred corrections |
| 10 | MTP indexer top-k cost per draft step | `_share_mtp_indices` step-0 compute + reuse |

## 3. Root-issue fix program (zero-degradation law applies)

**Phase A — port the metadata contract (the keystone).** Replace the fork's
static gate with per-request ragged `SpecDecodeMetadata` (cumsum layout).
This alone converts laws 1/2/3 from FATAL to legal. Touches: GMR spec branch,
scheduler output plumbing, verify path. Gate: nospec parity legs + spec leg
108/108 clean + KV fingerprint stable.

**Phase B — GPU-resident verify/propose.** Move rejection bookkeeping to
device tensors (`num_rejected_tokens_gpu` pattern), null CPU shadows instead
of syncing. Kills the D2H park class (law 4) and cuts drafter step cost
(law 5). Gate: SPECTIMING propose p50 ≤7 ms / greedy ≤1.0 ms (§11.7 = R2 bar,
absorbed here).

**Phase C — scheduler-side width policy.** `scheduled_spec_decode_tokens`
budget-slicing + optional dynamic-SD lookup (laws 6/8). Opt-in knob, default
off; measure accept-rate curve before enabling K-schedule.

**Phase D — TP>1 spec unlock** (law 7) — only after A-C certified; the v124
eager-forcing gate stays until a spec×TP leg survives N≥3.

Order A→B→C→D; each phase independently gated and revertible (nospec posture
never at risk — spec stays opt-in).

## 4. Relation to P83 headroom guard (parallel track)

The launch-headroom guard is posture-general and independent of the MTP
program (P82 law: spec only raises exposure frequency). Ship the guard first
(small surface: init_device cap + pool shave knob), then phase A. Both ride
the same gate harness (storm legs, ok=108/108, wall parity band, f-floor ≥
headroom, zero resets, zero OOM Tracebacks).

## 5. Open items for design phase
- Read proposer `__init__` (63-308): draft KV pool sizing + workspace.
- Read `prepare_next_token_ids_cpu` vs `_padded` (946/979): async-scheduling
  adjacency (R2).
- Read upstream rejection/verify kernel (GPU verify path).
- Qwen3-Next MTP head specifics upstream: `qwen3_next_mtp` model class +
  `mtp_num_hidden_layers` handling (A5000 lane evidence).
