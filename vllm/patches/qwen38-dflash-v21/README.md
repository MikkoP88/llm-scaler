# llm-scaler-vllm-adv:v21 — spec decode honors `--kv-cache-dtype` VRAM saving

`FROM llm-scaler-vllm-adv:v20`. v20's headline k2c3 config (tq4nc, SPEC_K=2)
could not run at the user's `--max-model-len 262144`: boot failed with
"13.09 GiB KV cache is needed ... available 6.67 GiB ... estimated maximum
model length is 126976", while the SAME config without spec booted fine.
Root cause was our own guard, not a hardware limit: DFlash's
`_create_draft_vllm_config` rewrote the draft engine's turboquant KV dtype
to "auto", forcing an uncompressed (bf16) drafter KV pool (~4x the target's
per-token cost) on top of the target's compressed pool.

v21 = guard fix + three follow-ups found by validation:

1. **Guard fix** — the draft engine keeps the turboquant dtype (VRAM parity).
2. **Non-causal draft attention in TQ** — the guard's "TurboQuant kernels are
   causal-only" reason was half-right: DFlash draft steps build metadata with
   `causal=False` (dflash.py asserts it), i.e. every draft query row attends
   the FULL stored context (the chunk's K/V are pre-stored by the drafter's
   precompute, so there is no causal ramp). First v21 boot crashed at
   `dflash.py:894` ("does not have non-causal support") until TQ carried the
   causal flag on its metadata and served that shape; v21b added a
   `NON_CAUSAL` mode to the multi-query kernel so draft steps run one shared
   KV pass instead of q_len synthetic-decode full-context rescans.
3. **Two-regime draft-KV policy (v21c)** — measured: a compressed draft pool
   next to a tq4nc target costs acceptance (E[len] ~1.78 vs ~2.0-2.4) and
   the TQ draft backend adds ~5-10 ms/step vs flash. Neither cost is
   justified below the VRAM parity boundary, so the policy only compresses
   where parity actually needs it:
   - `max_model_len <= 131072`: draft KV "auto" (v20 behavior, bf16 flash
     pool — measured equal to v20's gates).
   - `max_model_len > 131072`: draft KV `turboquant_k8v4` (parity: the
     user's tq4nc k2 @262144 config boots like nospec and out-runs it).
   - fp8/bf16/k8v4 targets: plain inherit — v20-identical by construction
     (the pre-v21 guard only ever rewrote turboquant_* dtypes, so e.g. v20's
     fp8 cells already ran an fp8 draft pool).
   131072 is derived: 6.67 GiB available / (13.09 GiB / 262144 tokens)
   ~= 133.6k tokens, so 131072 is the largest standard maxlen where the
   bf16 draft pool still fits.

## Root-cause detail (measured/verified 2026-08-29)

1. The dtype rewrite: `_create_draft_vllm_config` mapped turboquant_* ->
   "auto" for the draft engine only, so `xpu.get_attn_backend_cls` routed the
   drafter to FlashAttention with a bf16 pool (~11.3 GiB at 262144, vs ~2.8
   GiB tq4nc / ~4.2 GiB k8v4). The target pool was already compressed —
   hence spec "not honoring" the dtype while nospec did.
2. The non-causal contract: dflash's `build_per_group_and_layer_attn_metadata`
   builds full-attention draft groups with `causal=False` and asserts
   `metadata.causal is False`. The drafter's store path calls
   `do_kv_cache_update` per layer (TQ's split-store contract: store BEFORE
   forward; forward never writes KV), and its query-only propose forward is a
   "continuation chunk" (q_len = accepted tokens <= k+1 <= 8) attending the
   whole stored context.
3. The one real ordering hazard — TQ's mixed decode/prefill split assumes the
   target runner's decodes-first `reorder_batch`, which the drafter does not
   do (per-request q_len = accepted+1, mixed 1..k+1) — is fixed by the new
   `TurboQuantMetadataBuilder.build_for_drafting`: a genuinely mixed batch is
   forced onto the pure-prefill per-request continuation loop (correct for
   any q_len mix). Pure batches keep their natural fast paths.
4. fp8 draft KV already flowed through unchanged in v20 (cells c2/k2c2) —
   the Flash backend natively serves the non-causal draft shape; no TQ work
   needed there.
5. Two mixed-dtype pool bugs surfaced once the draft dtype could differ from
   the runner's (i.e. tq4nc target + k8v4 draft, or an fp8 target with a TQ
   draft via override). The drafter registers its draft-dtype TQ groups into
   the TARGET runner's group list, and two runner-side sites keyed off
   `self.cache_config.cache_dtype` (the target's dtype) instead of the
   group's own spec:
   - `_get_attention_kv_cache_shape` computed the TQ slot size via
     `TurboQuantConfig.from_cache_dtype(runner_dtype)` -> pool view
     `shape '[.., 134]'` against a 196-byte/page allocation (k8v4 draft in a
     tq4nc runner). Fixed: derive the slot from the group's own
     `TQFullAttentionSpec` (`tq_slot_size`, fallback
     `page_size_bytes / (block_size * num_kv_heads)`).
   - `_update_hybrid_attention_mamba_layout` -> base
     `get_kv_cache_block_dim` builds its sentinel shape through
     `get_kv_cache_shape(runner_dtype)` -> "Unknown TurboQuant cache dtype:
     'fp8_e4m3'" (TQ draft groups in an fp8 runner). Fixed: TQ overrides
     `get_kv_cache_block_dim` to probe with a valid preset (the block dim is
     preset-independent: num_blocks is always dim 0).
   Flash backends ignore `cache_dtype_str` for shape purposes — which is why
   same-dtype pairings and fp8/bf16 drafts never tripped either site.

## Changes vs v20

| Change | File | What | Env knob (default) |
|---|---|---|---|
| Keep turboquant dtype for draft KV | dflash.py | `_create_draft_vllm_config` no longer rewrites turboquant_* -> "auto"; `use_non_causal=False` for TQ drafts | `VLLM_DFLASH_TQ_DRAFT_KV` (1) |
| Two-regime draft-KV policy | dflash.py | tq4nc target: draft KV "auto" at `max_model_len <= 131072`, `turboquant_k8v4` above (parity); other dtypes inherit. Rationale in-code, all measured | `VLLM_DFLASH_DRAFT_KV_DTYPE` (policy) |
| Causal flag on TQ metadata | turboquant_attn.py | `TurboQuantMetadata.causal` from CommonAttentionMetadata; absent attr defaults True (target paths unchanged) | — |
| Draft-step metadata | turboquant_attn.py | `build_for_drafting` override; mixed batch -> pure-prefill per-request loop | — |
| Non-causal draft attention | turboquant_attn.py | DFlash draft steps (causal=False) route to the multi-query kernel: q0 = FULL seq_len, every row attends the stored context; synthetic-decode fallback keeps a non-causal variant | `VLLM_TQ_MQ_VERIFY` (1; 0 = synthetic-decode) |
| Non-causal MQ kernel | triton_turboquant_decode.py | `NON_CAUSAL` constexpr in `_tq_mq_decode_stage1`; `triton_turboquant_mq_decode_attention(..., non_causal=False)` — default False is bit-identical v19/v20 verify | — |
| TQ slot from group spec | gpu_model_runner.py | `_get_attention_kv_cache_shape`: TQ slot size from the group's own `TQFullAttentionSpec`, never the runner dtype (mixed-dtype pool crash) | — |
| TQ block-dim probe | turboquant_attn.py | `get_kv_cache_block_dim` override probes with a valid TQ preset (mixed runner dtype crash) | — |
| Env pass-through | serve.sh | forwards the new knobs | — |

Non-TQ dtypes and all target-side TQ paths are byte-identical to v20.
`VLLM_DFLASH_TQ_DRAFT_KV=0` restores v20 behavior exactly.

## Build (host, only while NOT serving)

```bash
scp -r vllm/patches/qwen38-dflash-v21 root@<host>:/root/build/
ssh root@<host> 'cd /root/build/qwen38-dflash-v21 && docker build -t llm-scaler-vllm-adv:v21 .'
```

The Dockerfile verifies the overlay bases are pristine vs the v20 image
(md5), greps for every new knob, and py_compile + import-checks the overlays.

## Run

```bash
# the config that failed on v20 (now boots; draft pool auto-pairs k8v4
# because 262144 > 131072):
SPEC_K=2 KV_DTYPE=turboquant_4bit_nc TARGET_DIR=/models/qwen3.8-27b-fp8 \
  DRAFTER_DIR=/models/drafter-fp8-v5 MAXLEN=262144 \
  EXTRA_ARGS='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  ./serve.sh
# at max_model_len <= 131072 the same command keeps the v20 (auto/bf16)
# draft pool automatically; force either regime explicitly:
VLLM_DFLASH_DRAFT_KV_DTYPE=turboquant_k8v4 ... ./serve.sh
```

<!-- V21_MEMORY_RESULTS -->
### Memory validation @262144 (first pass, 2026-08-29; one boot per cell)

All three quantized `--kv-cache-dtype` values now boot with spec at
`--max-model-len 262144` — VRAM parity with nospec. Zero xe resets, health
200, full canonical car-game bench served (sane text) on every cell.

| cell | target KV | draft KV | k | boot | GPU KV tokens | conc. | steady tok/s | E[len] | resets |
|---|---|---|---|---|---|---|---|---|---|
| t21k2c3 | turboquant_4bit_nc | turboquant_4bit_nc | 2 | UP | 510,202 | 1.95x | 31.18 | 1.77 | 0 |
| t21k8v4 | turboquant_k8v4 | turboquant_k8v4 | 4 | UP | 338,507 | 1.29x | 31.33 | ~2.07 | 0 |
| t21fp8 | fp8_e4m3 | fp8_e4m3 | 4 | UP | 269,042 | 1.03x | 33.15 | ~1.90 | 0 |

Reading: acceptance loss on t21k2c3 (1.77) is the compressed DRAFT pool,
not the guard fix; a bf16 draft next to the same target holds ~2.0-2.4
(v21d d1/d2). Step cost was also up ~8 ms (synthetic-decode draft
attention), recovered by the v21b non-causal MQ kernel. v21c made the
compressed draft conditional on actually needing it for parity.
<!-- /V21_MEMORY_RESULTS -->

<!-- V21_GATE_RESULTS -->
### Validation gates (steady_true last-50%, canonical car-game bench)

v20 gates re-derived per-cell from `/root/telemetry/cargame_*.out` (the v20
matrix ran the spec gates with `/models/drafter`; the v21d A/B showed the
`drafter-fp8-v5` swap is performance-neutral with an auto draft pool, so
the gates transfer):

| cell | config | v20 gate | v21 final | delta | E[len] |
|---|---|---|---|---|---|
| bar | tq4nc, no spec, 262144 | 32.78 | **32.79** | +0.03% (exact) | — |
| k2c3 | tq4nc k2, 98304 | 39.97 | **39.78** | −0.5% | 2.18 |
| c3 | tq4nc k4, 98304 | 33.39 | **34.08** | +2.1% | 1.87 |
| c2 | fp8 k4, 98304 | 33.34 | **34.34** | +3.0% | 2.12 |
| t21buser | tq4nc k2, 262144 (user cfg) | n/a (v20: cannot boot) | **32.93** | ≥ bar twin (+0.4%) | 1.66 |
| c2mix | fp8 k4 + k8v4 draft override, 98304 | n/a (override path) | **31.42** (boots) | crash-fix proof | 2.30 |

ALL 6 CELLS PASS, zero xe resets, zero crashes, sane text on every cell
(steady_true last-50%; boot lines confirm the policy regime per cell: draft
pool "auto" at 98304 tq4nc, turboquant_k8v4 at 262144 tq4nc, inherit on fp8,
override on c2mix). The user's failing config now boots AND edges out its
nospec twin at 262144 (32.93 vs 32.79) — v20 could not boot it at all.

Diagnostics that shaped the policy (v21c image, 98304 k2 tq4nc):
d1 `/models/drafter` + auto draft ~= v20 buckets; d2 `drafter-fp8-v5` +
auto draft = 41.29/39.76 tok/s buckets, E[len] 2.05-2.42 — i.e. the
earlier 31.29 miss was entirely the compressed draft pool, and the final
policy's default at 98304 reproduces v20-level throughput.
<!-- /V21_GATE_RESULTS -->

## Known limits

- Draft-KV quantization can perturb draft proposals (not target logits);
  the two-regime policy keeps the draft pool uncompressed unless parity
  requires it, and acceptance gates bound the effect where it does. Output
  text remains governed by the target's distribution.
- At >131072 with a tq4nc target, spec trades E[len] (~1.66 vs ~2.2 with a
  bf16 draft at 98304; +~5-10 ms/step TQ draft backend) for the ability to
  boot at all — measured net on the final image: 32.93 vs the 32.78-32.79
  nospec twin at 262144.
- Mixed multi-request drafting batches take the per-request continuation
  loop (correct, marginally more host work than the split fast path);
  single-request and pure batches are unaffected.
- fp8 + spec @262144 boots at 1.03x concurrency (one full-length request);
  the draft pool is the fp8 flash path, unchanged from v20.
- #03/#05(a) driver protocol unchanged.

## Files

- `Dockerfile` — FROM v20; overlays dflash.py + turboquant_attn.py +
  triton_turboquant_decode.py + gpu_model_runner.py; grep guards,
  py_compile + import checks (md5-pristine base check in the build script)
- `dflash.py` — guard fix + two-regime draft-dtype policy (env-gated)
- `turboquant_attn.py` — causal flag, `build_for_drafting`, non-causal MQ
  routing + synthetic-decode fallback, block-dim probe override
- `triton_turboquant_decode.py` — NON_CAUSAL mode of the MQ kernel
- `gpu_model_runner.py` — TQ slot size from the group's own spec
- `serve.sh` — v20 + the new env pass-throughs
- everything else inherited unchanged from v20 (bench client/matrix,
  spec_timing instrumentation set, monitor/battery scripts)
