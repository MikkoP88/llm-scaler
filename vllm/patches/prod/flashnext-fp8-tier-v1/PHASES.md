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

**Round 3 — offload-path verdicts (2026-10-11, throwaway fn8-p + fn8-t1):**
checkpoint is the official Qwen/Qwen3.8-Flash-Next-FP8 release (README:
base Qwen/Qwen3.8-Flash-Next, "fine-grained fp8 quantization with block
size of 128") · expert scales are BLOCK-scaled, not per-tensor:
`experts.N.{down,gate,up}_proj.weight` F8_E4M3 [2560,640]/[640,2560] with
`weight_scale_inv` BF16 [20,5]/[5,20] = ceil(dim/128) grid, 512/512 scales
in shard-3 census — exactly the ESIMD blockscale path (construct-time
selection `XPUFp8BlockScaledMMKernel` already banked); PLE scale stays
per-tensor [1] · **PIN CEILING LAW (torch path):** `pin_memory()`
escalation probe (fn8-p, GPU-attached, host-side only, lane up) died
silently between 16 and 20 GiB cumulative/process — same band as the
LMCache USM pinned law → UVA weight offload (hard-asserts pinned on XPU,
needs 48 GiB/rank) is DEAD on this stack · **G1 reference shape =
pageable offload**: `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` +
`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` + `--cpu-offload-gb 48` → params
pageable on CPU, functional_call copy-per-forward (correct,
PCIe-bound ~5 s/token; fine for a correctness reference).  Prefetch
backend unusable as-is (torch.cuda.Stream/Event flavored).

## 8. Round 4 — GPU bring-up, P82 repack/forward hoard (2026-10-11, fn8-g1)

**TP2-BLOCK-128 LAW (user directive, standing):** TP2 must be WHOLE-EXPERT
partitioning, never tensor-slicing — moe_intermediate 640 = 5×128 blocks;
320 = 2.5 blocks is not a legal slice. Whole-expert keeps both cards
computing every layer (the structural win over Strata layer-split; see
§10). **gmu 0.8 for fn8-g1 testing** (user directive; lane keeps its own
posture). 4/6-bit quant fallback is BANKED, not mainline (FP8 first).

**Leg-4/5 wedges → patch_00c + patch_00d (P82 family on the offloader):**
weight REPACK loop hoarded freed XPU allocator blocks across modules
(expandable_segments never returns them) → segment growth stalled in
`xe_vm_bind_ioctl` (D-state, "Range is already in VRAM", xpu-smi + py-spy
both hung against the process). patch_00c_offload_repack_cache.py =
empty_cache per repack module boundary (load path); patch_00d (anchor
uva.py functional_call wrapper, `module.forward = forward / return
output`) = same medicine where the ~2.9 GiB/layer H2D copies actually
happen at serve time. Both idempotent (FN8_00C/FN8_00D), .q4dbak +
--restore, md5-verified.

**PLE offloader law:** plain-attribute staging buffers (not nn.Parameters)
are invisible to the offloader's module scan → keep the 47.7 GiB PLE
n-gram table OUT of offload accounting; staged per-shard host→(optional)
device assembly with explicit empty_cache between shards (assembly law).

## 9. Round 5 — serve legs 6/7/8 root causes + P0 reference economics

- **Leg-6 PROFILE LAW:** `profile_run → _dummy_run(is_profile=True)` runs
  the whole model with `attn_metadata=None` and zero wired mamba pools;
  native GDN core survives via the None-guard; PLE must mirror that →
  `_ple_forward_stateless` (synthetic single sequence, EOS-seeded history,
  zero conv state, no slot reads/writes) when attn_metadata is None.
- **Leg-7 M-ROPE KEY-PRESENCE LAW:** transformers v5 backfills
  `mrope_section`/`mrope_interleaved` into rope_parameters for this
  VL-heritage checkpoint (rope_type "default"); vllm uses_mrope() trips on
  key PRESENCE → strip when rope_type == default (config-class fix,
  verified by fn8_mrope_probe: uses_mrope False).
- **Leg-8 INT32 INPUT_IDS LAW:** XPU runner feeds int32; PLE history
  buffers int64 → `.to(torch.long)` at `_ple_forward_flat` entry.
- **Leg-9 serve (fn8-g1 :8319) HEALTHY:** weights 17.45 GiB / 161.8 s,
  host PLE table assembled, KV 4.95/5.04 GiB, "2 + 2 =" → " 4" correct.
- **P0 decode economics (measured):** ~10–17 s/token solo first-pass
  (~29.8 s/token with the 104 GiB cold page-in), ~45 s/token 2-way —
  the pageable functional_call copies ~104.4 GiB/step by design. This is
  the baseline P2/P3 must beat (target ≥40 tok/s gate, Strata's own
  single-B70 Flash-Next IQ3_S reference: 30.5–40.8 tok/s decode,
  730–1117 tok/s prefill).
- **Smoke battery (fn8_smoke_g1.py, :8319):** fib50 leg = SEMANTIC PASS
  ("12,586,269,025" — correct F50, comma-grouped; client substring check
  fixed to compare comma-stripped). Battery banks
  /stage/bank/g1_smoke.json as the P2 parity reference.

## 10. Round 5b — Strata memory deep-analysis (fn8_strata_memory_analysis.md)

All 7 quoted memory-table figures VERIFIED (experts 112.5 GiB fp8
weights-only / ~116 GiB with block scales / 104.4 GiB measured pageable;
dense ≈13 GiB; PLE 320,001,536×160 B fp8 = 47.687 GiB = 51.2 GB host;
MTP ~6 GB; 64 GiB cards; 31–32% ≈ "~33%"). Strata's tok/s story IS the
memory tiering (free-check-sized slot arena + per-layer quotas +
profile-ranked fill + adaptive exchange swaps + pinned host mirror as a
COMPUTE tier + route tail skip + layer-ahead streaming + KV resident
window); their dual-GPU is LAYER SPLIT ONLY — decode ≈ faster card alone
(5080+3090: 84–110). **Their run of Flash-Next IQ3_S on our exact card
(B70 SYCL port): decode 30.5/40.8 tok/s prose/code, prefill 730–1117,
PCIe 13.3 GB/s, 11.5k experts resident + pinned mirror** — the number to
beat. Superiority = whole-expert TP2 (2× aggregate PCIe + 2× hot cache +
both cards every layer) at fp8 fidelity; ceiling math 38–55 tok/s before
overlap, 60–90 with layer-ahead streaming, 150–300 effective with MTP×4.
Adopt list + avoid list (FS slots, helper-GPU, batch-mtp-on-split) +
xe hazards (arena aliasing tag-readback, VERIFY_NO_HOST handshake,
bounded spins) in the doc.

## 11. Round 6 — P1 routing profiler IMPLEMENTED + validated (2026-10-11)

**MONOLITHIC-BYPASS LAW:** the ESIMD fused MoE fast paths
(`moe_forward_full[_v2|_fp8_block]`, ≤12 tokens for fp8-block) do topk
INSIDE the kernel and never route through `moe_runner._apply_quant_method`
→ the capture buffer is NOT written on the decode fast path.
`DISABLE_ESIMD_MOE=1` (esimd_utils; documented as restoring the upstream
modular path 1:1) forces the path that writes it. XPUExpertsFp8Block is
`FusedMoEExpertsModular` → the non-monolithic branch (select_experts →
`_routing_replay_out.copy_`) is our capture point.

**patch_01_routing_dump.py** (FN8_01, anchors×5, .q4dbak, --restore):
(a) XPU D2H fallback — `torch.cuda.Stream/Event` creation wrapped in
try/except → None → synchronous D2H into pinned staging + immediate
scatter; (b) env-gated `V1X_ROUTING_DUMP=<file>` histogram tap in
`_scatter_to_host` (per-(layer, expert) counts, flush/500 scatters +
atexit, atomic tmp+rename, never raises into the engine). Enable path
needs NO patch: `--enable-return-routed-experts` (EngineArgs field;
gpu_worker + capture_model init + bind, moe_layer_id auto-assigns 0..47).
**fn8_expert_profile.py**: ranked (layer, expert) STRP analog + global
coverage marks + per-layer-quota (R4.2g) coverage + TP2 parity-split view
+ the ≥75%-within-8192 gate.

**Validation (fn8-t2, GPU-less, pristine v1.2.35):** patch dry-run
apply 36d913c3→58ac98e2, py_compile OK, 4 markers, restore roundtrip
exact; histogram chain test — tap×2 (64×48×10 synthetic zipf) → sum
61440 exact, hot-mass 0.7–0.9/layer as designed; profiler on the
synthetic hist: top-1024 = 0.81 (matches the 960-hot design), per-layer
q=32 = 0.843, gate logic PASS. P1 profiling RUN = next engine window
(fn8-g1 after smoke): serve + `--enable-return-routed-experts` +
`DISABLE_ESIMD_MOE=1` + `V1X_ROUTING_DUMP=/stage/bank/routing_hist.json`
over a code+prose corpus → fn8_expert_profile.py → bank the curve.
