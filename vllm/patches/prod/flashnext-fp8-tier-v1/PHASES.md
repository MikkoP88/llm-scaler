# PHASES.md — flashnext-fp8-tier-v1 (Strata tiering ideas → vLLM fork, Qwen3.8-Flash-Next FP8 on 2× Arc Pro B70)

Round opened 2026-10-11. Directive: deep-research Strata's expert-offload
pipelines/ideas/logic and port them into this vLLM XPU fork so
**Qwen3.8-Flash-Next 125B MoE runs in FP8 at Strata-class speed on limited
VRAM (2× B70, 64 GB)**. **All changes live in patches** (patch_*.py under
this dir, applied in-container to /opt/venv/.../vllm, docker-commit bake at
the end). No subagents; md5+grep verify every edit.

---

## 1. Research base (what Strata actually does — closed reading 2026-10-11)

Sources: C:\Projects\Strata @61b3fb5 — docs/HOW_IT_WORKS.md, MODELS.md,
DETAILS.md, DISJOINT_EXPERT_CACHE.md, INTEL.md, INTEL_ARC.md + sycl/ port.

**Model shape (Flash-Next):** 48 layers = 36 gated-delta-net (linear attn) +
12 sparse full attn; **512 experts/layer → 24,576 total**, **10 active per
token** (~0.04% of expert bytes touched per token — this is why tiering
works at all); shared expert + router + dense head are small. MTP draft
layer ~6 GB. FP8 weights ≈ **103–112 GB experts + ~13 GB dense** → cannot
be VRAM-resident on 64 GB → tiering is mandatory (same conclusion as
Strata; Strata has NO FP8 weights at all — GGUF GSQ-RCO only — so this is
strictly new capability for the fork).

**Strata's tiering stack (the ideas to port):**
1. **VRAM hot-expert cache** — slot pool sized from free VRAM (`--expert-cache
   auto`), seeded from a shipped **expert profile** (offline routing
   histogram), then **adapted at runtime** (`--adapt-every` windowed swaps,
   `--adapt-swaps`, `--adapt-decay`, ADAPT_LAG=2 windows; `--adapt-async 1`
   does swap copies on a helper thread between windows). Profile persists
   across restarts (0.1.36 #477). `tools/replay_cache_policy.py` replays
   `--dump-routing` traces offline to tune policy.
2. **All experts mirrored in pinned host RAM** (`--stream-experts`); low-RAM
   fallbacks: mmap from SSD / resident-subset copy — NOT needed for us
   (376 GB host RAM ≥ 110 GB FP8 mirror + OS).
3. **Miss handling**: CUDA engine computes cold experts **in place on the
   CPU** (AVX-512 ggml kernels) concurrently with GPU computing hits —
   the two sides never wait for each other; SYCL port instead **streams
   misses in over PCIe** and prefetches **the next layer's experts during
   the current layer's attention** (`--pcie-frac` splits the two modes).
   Our v1 = the SYCL-port approach (stream + prefetch); CPU-side compute is
   a stretch phase.
4. **Prefill regime is easy**: a chunk touches most of a layer's 512 experts
   once → stream next layer's set during current layer (Strata does exactly
   this; 2× B70 prompt 394–1,002 tok/s).
5. **MTP speculative decode** `--spec 4` (1.6–1.8× on CUDA; 2.4–3.2
   tokens/verify window typical) + prompt-lookup drafter for repeats.
6. **KV tiering**: `--kv-resident N` keeps only hot KV in VRAM (frees
   expert slots; Q2_0 262K: 50.9→62.6 tok/s). Fork already has the stronger
   version: LMCache CPU tier (certified on this lane family).
7. **Reproducibility laws** (#152/#410): GPU-vs-CPU expert split and PCIe
   fraction change rounding → greedy answers shift. Port rule: streaming a
   miss into VRAM then computing on GPU = same math as resident → greedy
   parity vs reference must hold; make it a gate, and make the reference a
   UVA-offload full-resident run.
8. **xe hazards** (their INTEL.md, matches our fleet law): never
   host-writable flags on device memory (`STRATA_VERIFY_NO_HOST=1` class —
   NaN logits), leave VRAM reserve, silent sysmem-eviction livelock when
   over-allocated.

**Calibration points (measured, their hardware):** 2× B70 layer-split
IQ3_XXS (43 GB experts, 100% resident) = 66 tok/s decode / 394 prompt
(older engine); 1× B70 IQ3_S **47% resident** = 30–41 tok/s at 262K ctx /
980–1,002 prompt @4K. Disjoint-cache guard (helper/primary no duplicate
slots): +15.5% gen speed.

## 2. Fork surface inventory (verified in-container 2026-10-11, lsv-test)

- `model_executor/models/qwen3_next.py` (1980 ln) — **Flash-Next arch class
  present** (GDN + MoE + shared-expert), already imports fork ESIMD fp8
  MoE kernels: `esimd_gemv_fp8_blockscale_fp16_fused2`, `esimd_gemv_fp8_pert`,
  `esimd_resadd_norm_gemv_fp8_pert`, `torch.ops.moe_ops.moe_forward_full_fp8_block`
  (block-fp8 and per-tensor fp8 paths both wired, INT4 too).
- `model_executor/models/qwen3_next_mtp.py` (310 ln) — **MTP drafter for the
  same arch exists** (num_nextn_predict_layers), plus the lane's certified
  MTP×4/async runtime.
- `layers/fused_moe/` — full modern stack; **`routed_experts_capturer.py`
  provides `vllm::capture_routing` custom op (torch.compile-traceable)** =
  our profiler hook; `cpu_fused_moe.py` (CPU MoE custom ops, for the
  stretch phase); `oracle/` (fp8/int8/mxfp4 reference impls = parity
  oracles).
- `config/offload.py` — upstream offload backends **uva** (zero-copy pinned
  CPU) and **prefetch** (grouped-layer async H2D, `offload_prefetch_step`)
  = our P0 reference-run mechanism, no patch needed to use.
- `custom-esimd-kernels-vllm/` (repo) — setup_moe_only / setup_moecore_only /
  setup_moe_int4_only build targets; kernel .so is baked per image
  (KERNELS_MAX_JOBS=52 law if we ever rebuild).
- Lane posture (lsv-test, v1.2.35 content): TP2, fp8_e4m3 KV, mamba fp16,
  MTP×4, async, LMCache, MNBT 8192, CHUNK 1024 — all carried into the new
  lane's serve shape; content gates of the OLD lane untouched until bake.

## 3. Capacity math (the design constraint set)

Per card (32 GB, TP2, FP8): dense+head+embed+MTP ≈ 9–10 GB, activations +
FULL_DECODE_ONLY graphs ≈ 3 GB, **xe reserve ≥1.5 GB (law)** → expert cache
≈ **17–18 GB/rank ≈ 4,000 FP8 experts ≈ 33% of all experts resident**.
Host: 110 GB pinned FP8 mirror + OS ≤ 376 GB (fits; keep LMCache
MAX_LOCAL_CPU_SIZE small on this lane — pools are per-lane anyway).

Decode traffic: all-cold per-token expert reads ≈ 48×10×4.4 MB ≈ 2.1 GB
(1.05 GB/rank). At effective aggregate H2D ~40–50 GB/s (pinned, two cards),
55 tok/s needs miss-fraction ≲ 20–25% → **hit-rate ≥ ~75–80% is the whole
game**; Strata's adaptive+profiled caches operate in that band on
code/repetitive traffic (their 47%-resident leg still did 30–41 tok/s at
262K ctx on ONE card — our regime is better: 2 cards, 4K–32K ctx, profile
seed, MTP×4, next-layer prefetch). Prefill: one 8192-chunk touches ≤512
experts/layer once ≈ 54 GB/chunk streamed overlapped → ≳6,000 tok/s
ceiling → prefill target is comfortably reachable.

**Targets (certification, single-stream, warm-engine law):**
| metric | GATE | STRETCH |
|---|---|---|
| decode tok/s (code2k, 4K ctx) | ≥ 40 | ≥ 55 |
| prompt tok/s (4K chunked) | ≥ 600 | ≥ 800 |
| MTP acceptance | ≥ 2.2 tok/window | ≥ 2.6 |
| correctness | fib(50)/gcd/exec 100% | — |
| stability | 0 NaN, 0 xe oops, 0 stalls, ≥1.5 GB VRAM free/card | — |
| greedy parity | byte-identical vs UVA reference (same seed, 2 repeats) | — |

## 4. What we port / what we deliberately do NOT (v1)

PORT: profiled+adaptive VRAM expert cache; pinned host mirror; PCIe
stream-in with next-layer prefetch (prefill) and window-ahead staging
(decode); disjoint-slot ownership across the two TP ranks (each rank owns
its shard — disjoint by construction, note in plan); MTP×4; greedy-parity
gate; profile persistence.
NOT (v1, revisit after P7): CPU-side cold-expert compute (stretch P6);
PLE/prompt-lookup SSD n-gram table (fork has MTP; LMCache prefix reuse is
stronger than their prompt cache); draft-vocab subsets; coupled/Gumbel
draft sampling; layer-split/EP topology (keep certified TP2 posture;
record as study leg); mmap-SSD experts (RAM fits).

## 5. Phases (each = patcher(s) + banked evidence + gate)

**P0 — assets + FP8 bring-up (no fork edits).** Obtain Flash-Next HF
checkpoint; quantize to **block-fp8** (path the ESIMD blockscale kernels
want; per-tensor fp8 is fallback leg) with fp8 gate/router handling
(`_dequantize_fp8_gate_weight` exists). Smoke on 2× B70 lane-safe window:
load qwen3_next TP2 at small ctx with `--cpu-offload-gb` (UVA backend,
upstream infra) → logits sane, greedy fib(50) correct; qwen3_next_mtp loads.
*Gate: correctness on UVA-offload reference run; bank reference outputs for
the parity gate.* patch_00 only if registry/config shims are needed.

**P1 — routing profiler.** patch_01_routing_dump.py (anchor
routed_experts_capturer.py; env `V1X_ROUTING_DUMP=<file>` gates buffer
flush, off by default) + install `tools/fn_expert_profile.py` (aggregate
per-layer expert histograms → ranked list + coverage curve JSON; Strata
profile analog). Run over a corpus (code+prose+agentic traces). *Gate:
coverage curve banked (top-k experts vs cumulative hit); expects ≥75%
coverage within ≤8k experts on code-ish traffic.*

**P2 — tiered expert store (the core).** patch_02_tiered_experts.py: new
`fused_moe/tiered_experts.py` + anchors in FusedMoE
`process_weights_after_loading` (mirror→pinned, top-P set→device slot pool,
sized `V1X_TIER_CACHE_GB` or auto-from-free-VRAM with xe reserve law) and a
forward pre-step reading the capturer's routing buffer → miss list → async
H2D into LRU slots (rank+recency eviction); sync-fallback wait if a slot
isn't staged (correctness before speed). Knobs `V1X_TIER_*`: CACHE_GB,
SEED_PROFILE, ADAPT_EVERY, ADAPT_SWAPS, PREFETCH_LAYERS, LAG. Design rules:
no host-writable flags on device buffers (xe NaN law); all slot state
device-side or copied. *Gate: greedy byte-parity vs P0 reference; decode
runs with hit-rate counter logged; zero NaN over 30-min soak.*

**P3 — prefill streaming.** patch_03_prefill_prefetch.py (anchor chunked
prefill path in gpu_model_runner-class code, NEW md5 content gate on the
NEW image tag only): during layer L compute, stage layer L+PREFETCH_LAYERS'
chunk-touched experts. MNBT stays 8192, CHUNK_SIZE 1024 (standing REJECTED
law on raising). *Gate: prompt tok/s ≥600 @4K; no TTFT regression vs P2.*

**P4 — MTP.** Serve config: speculative mtp ×4 on qwen3_next_mtp; reuse
certified async runtime (V60Z=3, draft barriers 0, FULL_DECODE_ONLY
graphs). patch_04 only if drafter/verify gaps surface. *Gate: acceptance
≥2.2 tok/window; storms not required here but run one 108-req soak for the
crash class.*

**P5 — adaptive tier + persistence.** patch_05_adapt_persist.py: windowed
swaps from live routing stats (async helper thread between windows —
Strata `--adapt-async` analog; mind P10 lesson: helper threads must
queue-kick, never park on events), decay, profile file persisted across
restarts. *Gate: hit-rate improves ≥5 points over seed on a second
corpus leg; restart reproduces cache state.*

**P6 (stretch) — CPU cold-expert compute.** Only if P2/P5 measurements
show PCIe-bound decode: route a miss-fraction to CPU fp8→f16 dequant GEMM
via cpu_fused_moe-class ops. Adds the Strata CUDA-engine concurrency win
and the `V1X_TIER_PCIE_FRAC` knob. *Gate: decode +≥15% on the bound leg;
greedy parity re-certified (rounding law #410: CPU-split changes answers —
if enabled, parity gate becomes distributional, documented).*

**P7 — certification + bake.** Bench matrix (short/code2k/prose2k ×
4K/32K ctx × cache-size sweep × MTP on/off), correctness exec-gates,
NaN/xe/stall gates, VRAM-free law, warm-engine law, greedy parity. Then
docker-commit bake **v1.3.x**, patch md5s recorded, chain rebase
(V1227_IMAGE→new tag, assert block extended with V1X_TIER flags), drift
verdict `certified-fn-fp8-tier-v1`, watchdog pause/unpause law throughout.
Every GPU window uses the LANE-SAFE procedure (pause → graceful TERM →
`pgrep -af "VLLM::|EngineCore"` empty → work → restore chain → health 200
→ drift → unpause); host reboot after any xe event.

## 6. Standing laws carried into this series

oneCCL pip law (libccl 2021.15 symlinks + numpy 2.3.5 re-verify after ANY
pip) · xe reserve ≥1.5 GB/card + silent sysmem-eviction watch · patch
discipline (anchors/markers/bak/idempotent/restore + md5+grep verify, no
subagents) · lane lsv-test untouched until bake; its content gates
(dcb23fbe/bb3c905e/1a6ae0ec/6520f916) must still match at every restore ·
orphaned-engine restart law · MNBT 8192 / CHUNK 1024 standing ·
KERNELS_MAX_JOBS=52 if kernels rebuild · commit with Co-Authored-By line ·
bank every leg (config + results + stderr tail + xpu-smi dump).

## 7. P0 execution log (2026-10-11, throwaway container fn8-t1)

Assets: checkpoint `/models/flash-next-fp8` on .65 — 185.5 GB
(173 GiB), 131 shards, 152,089 tensors; arch
`Qwen4ExpForConditionalGeneration`, model_type `qwen4_exp` (vendored trio
Qwen4ExpConfig/Text/Vision; stock transformers 5.8.0 has none of it).
Throwaway `fn8-t1` from llm-scaler-exp:v1.2.35; image parity vs live
lsv-test proven by md5 on the three patched-file pristine backups.

**Round 1 — t1_config_test.py PASS** (config/registry/import/ckpt cross-facts):
PLE per-tensor scale EXISTS — `…ple.ple_embedding.ngram_embedding.weight_scale`
BF16 [1], shard 00005 (corrects the earlier "NO scale tensors" —
fn8_inspect.py had a filtering gap) · ple_layer_ids [2] is ONE-indexed →
model layer index 1; 49 layers = 36 GDN + 12 full-attn + 1 PLE ·
hash buffers torch.equal vs checkpoint · vocab math exact
(total 320,001,446 ≤ 128×2,500,012 = padded 320,001,536) ·
`--runner generate` protocol law: registered class must expose
`__init__(vllm_config=…)` + `embed_input_ids` + `forward(input_ids=,
positions=)` + `compute_logits` — ForCondGen needed an explicit
`embed_input_ids` delegator · fused_moe_make_expert_params_mapping imports
from `vllm.model_executor.layers.fused_moe` (NOT models.moe.utils) ·
GatedDeltaNet from `layers/mamba/gdn_linear_attn.py` (prefix-based
attn_metadata_raw[self.prefix] plumbing confirmed) · SKIP_TABLE mode =
0-row fp8 table (never allocate 51 GB on a dry-run).

**Round 2 — t2_weightmap_test.py PASS** (real load path, meta model,
stub tensors): 148,487 loader calls == 148,487 non-skipped ckpt names
EXACT, 1,079/1,079 distinct params loaded, zero "Parameter not found"
warnings, non-scale untouched params = 0, PLE staging shard-index parse
0..127 + 1 scale + 3 buffers.  Checkpoint GDN projections ship SPLIT
(`in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`) — matches the
stacked mapping.  **MIXED-QUANT LAW (the round's discovery):** the
checkpoint quantizes ONLY the routed experts — 73,728 weight_scale_inv
tensors = 48 MoE layers × 512 experts × 3 (gate/up/down); GDN
projections, self-attn, router gates and shared experts ship BF16 with
NO scale tensors.  → GPU-smoke watch item #1: Fp8LinearMethod on
non-expert linears has no checkpoint scale (expect the fork's
dynamic-requant missing-scale path; if it asserts instead, we need a
per-linear quant override patch).

GPU-less dry-run idioms (banked for future CPU rounds):
`DeviceConfig(device="cpu")` (no inference) · `set_current_vllm_config(vc)`
must wrap dist-init AND model construction (Fp8LinearMethod reads it) ·
force `vllm.platforms.current_platform = XPUPlatform()` BEFORE any vllm
import that binds it (kernels/linear/__init__.py binds at module import;
unpatched ⇒ KeyError PlatformEnum.UNSPECIFIED) + `XPUPlatform.current_device
= lambda: "cpu"` (GDN ctor allocates RMSNormGated on
current_platform.current_device()) · stacked-mapping loops call
`param.weight_loader(param, w, shard_id, …)` POSITIONALLY — instrument by
param identity, not name · AutoWeightsLoader re-partitions weight
streams (a name-current variable coupled to the input generator lies;
tee at the child-module boundary instead).

Remaining pre-GPU unknowns (P0 smoke verifies): PLE runtime metadata
fields, F.conv1d depthwise-dilated fp16 on XPU, real weight load +
outputs, the mixed-quant missing-scale path above.
