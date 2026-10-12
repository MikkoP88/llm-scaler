# DEC40 — flash-next decode ≥40 tok/s solo (dual B70 vs Strata single B70)

Open 2026-10-12.  Directive: deep-analyze the decode gap, plan, implement, and
**do not stop before ≥40 tok/s solo decode is measured on the flash-next lane**.
Alternative model paths allowed (Q4 expert quant — variant of our choosing)
alongside FP8.  The checkpoint ships its own MTP head (user-confirmed: no
external drafter).

## D0 — analysis verdict (COMPLETE 2026-10-12)

**Current state (P3 certified stack, boot-12 banks)**: prefill 690.6 tok/s
median; decode **4.975 / 5.243 tok/s solo** ≈ 200 ms/step ≈ 4.2 ms/layer × 48.

**The wall is step latency, NOT PCIe** (P6 law): leg B absorbed 112,510 misses
(14.6% of routed dispatches) + 115,966 h2d copies on the side stream with no
wall-clock penalty (37.0 s vs 36.8 s control).  The certified P3 serve shape
runs **`--no-async-scheduling --enforce-eager` and NO speculative config** —
the three biggest decode levers this fork has certified on the swift lane
(async 2.3×, P84; graphs; MTP×4) are all switched OFF on fn8.

**Strata reference (their run, our card class, INTEL.md 739–825)**: IQ3_S
single B70, `--spec 4 --mtp`: decode 30.5 tok/s prose (66% accept, 1.98
tok/round) / 40.8 code (90% accept, 3.65 tok/round).  Their decode IS the
PCIe bound at their hit rate (~71 ms/round ÷ accept).  Their 30–40 **requires
MTP**; ours must too.

**Our ceiling math (fp8, dual B70, whole-expert TP2+EP2)**: per token per rank
~5 experts of top-10 on-rank; misses ~1–1.5/layer/rank × 4.9 MB / 13.3 GB/s ⇒
0.37–0.55 ms/layer ⇒ 18–26 ms/token ⇒ **38–55 tok/s before overlap; 60–90
with layer-ahead streaming; 150–300 effective with MTP×4** (2.0–3.6 accepted
tok/round).  Dual-card edge: 2× aggregate PCIe + 2× expert VRAM + both cards
compute every layer.  ≥40 is reachable on fp8 alone if the latency chain is
broken; Q4 (~2.6 MiB/expert) doubles slots per GiB and halves miss bytes as a
fallback/booster.

**Blog/config ground truth (qwen.ai qwen3.8-flash-next via jina proxy
2026-10-12; /models/flash-next-fp8/config.json)**: 125B main / 51B PLE
n-gram / **6B active**; 48 layers, 3:1 GDN:full (`full_attention_interval
4`); hidden 2560, hc_count 4 (Gated Residual 4 branches), hc_lowrank 320;
moe_intermediate 640 (block-128 law); QSA indexer budget 2048 / compress 4
(4.9× decode at 1M ctx, relative only — no absolute tok/s published); MTP
`{hybrid: true, layer_types: [full_attention], num_hidden_layers: 1}`,
"trained with multiple steps"; `mamba_ssm_dtype: float32` in config — serve
overrides to float16 (dtype law).  No Q4 variant published → Strata IQ3_S is
the only Q-class datapoint.

**Proposer anatomy (dumped from lsv-test v1.3.0, llm_base_proposer.py)**:
call site ~1298–1305: `supports_multimodal(target_model)` False for our FCG ⇒
`target_language_model = target_model` (the FCG, no `.model`) ⇒
`_maybe_share_embeddings` raises "Target model does not have 'model'
attribute" (boot-14b death, line 1333).  The MTP branches ALREADY exist and
are correct: `share_embeddings = True` / `share_lm_head = True` for MTP
(draft embed/lm_head swapped to the target's — Strata stem law; saves the
draft's own copies).  Draft payload exposes `.model.embed_tokens` and
`.lm_head` (files/qwen4_exp_mtp.py) — both proposer accesses satisfied.
`embed_input_ids(dummy, multimodal_embeddings=None)` TypeError is already
caught (text-only fallback).  **One edit class fixes P4**: unwrap
`language_model` on the FCG in the `else` branch.

**Host recon (2026-10-12 ~01:3x)**: lsv-test v1.3.0 Up, drift
certified-fn-fp8-tier-v1, watchdog ARMED (no pause file), both B70 `normal`;
1.8T disk free; /root/fn8_stage banks intact (p3_run12.json, tier_p3.json,
routing_hist.json …); v1.3.0 image present (121ddcf82b1e); flash-next-fp8 =
173 G, 131 shards.

## Phase ladder

Leg serve scripts descend from fn8_serve_p4.sh (P3 shape + spec).  Every leg:
warm-engine law (completion before recording), ≥3 legs, median; FLUSH law
(≥200 dispatches/store before tier-dump reads); solo decode = 1 request,
256+ new tokens, tok/s = completion_tokens / decode_time (ttft excluded).

### D1 — MTP-ON (enabling fix + spec leg)
- patch_04b: proposer `else`-branch unwrap (FCG `.language_model`), applied
  in the work container on top of v1.3.0 (patch_04 already baked in).
- Boot: fn8_serve_d1.sh = fn8_serve_p4.sh shape (P3 + `--speculative-config
  '{"method":"mtp","num_speculative_tokens":4}'`), still `--no-async-scheduling
  --enforce-eager` (single delta law: only spec changes vs P3).
- Boot asserts: spec init lines; EP local experts 256; 48× store prewarm; no
  prefetch-DISABLED; draft weight load 0 not-found; NO AttributeError.
- Gate D1: spec boots clean + acceptance lines present + decode measured
  (expect ~5 → 10–14 tok/s from accept ~2 alone).

### D2 — ASYNC-ON
- Drop `--no-async-scheduling` (→ async scheduling on XPU fork: needs
  VLLM_XPU_ALLOW_ASYNC=1); couple VLLM_V60_ZOMBIE_STRIKES=3 (P84 wide-wave
  abort law — strikes couple to async, never alone).
- Gate D2: decode leg vs D1 same-boot-law A/B.  Expect ~2× (swift law).

### D3 — GRAPHS-ON (eager off)
- Drop `--enforce-eager` → XPU graph capture (FULL_DECODE_ONLY with spec —
  swift-certified combo; capture cost at boot only).
- Gate D3: decode leg vs D2.  Watch VRAM headroom (graphs + 56 GB pinned
  offload + 0.85 util) and wedge law.

### D4 — micro-levers (only if <40 after D3; each measured A/B)
- Route tail skip: missed expert ranked ≥8-of-top-10 → skip/zero (Strata
  +11–24% decode, KL 0.003–0.006).
- FN8_TIER_SLOTS 72→88 (Round-12 ceiling note: ~0.33 GiB/store at 72).
- Side-stream demand batches: event-wait instead of device-wide sync
  (Round-9 untried lever).
- Prefill-PF keyed on draft lookahead at decode (RouterLookahead analogue).

### D5 — Q4 expert path (fallback/booster if fp8 <40; user-allowed)
- First: fork int4-path research (vllm_int4_for_multi_arc.so exports, MoE
  routing) → variant decision (fork-native int4 vs dequant-in-kernel W4A16
  vs GGUF — GGUF unlikely: fork loader is safetensors).
- ~2.6 MiB/expert ⇒ ~2× slots/GiB + ½ miss bytes; parity gate mandatory
  (fixed-seed greedy vs fp8 outputs sanity).

### D6 — certify + bake (only on gate ≥40.0 median solo decode)
- Battery: decode ≥3 legs median ≥40; prefill regression policy — P3 median
  690.6; ship posture may trade prefill for decode but prefill must stay
  ≥600 tok/s median (production prefill floor) unless user re-scopes;
  parity/adj battery re-run; 0 NaN/stall watch.
- Bake v1.3.x from winning container; PHASES.md round record; commit; memory.
- Lane restore: lsv-test untouched content-wise (work containers are
  siblings); window discipline = pause watchdog → stop lane → work →
  start lane → drift V1227_DRIFT_NONE → unpause.

## Laws carried into this series
Whole-expert TP2 only (block-128); mamba float16 + kv fp8_e4m3 standing;
PYTHONHASHSEED=0; no oneCCL pip; wedge law; pkill bracket patterns; plink
outer single quotes; backgrounded launches via nohup/docker exec -d; Bash
tool timeout 600 s; md5-verify every edit; no subagents.
