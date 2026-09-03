# v38 — #18 SOLVED: fp8-KV decode rerouted off the ESIMD fast path

`llm-scaler-vllm-adv:v38` = certified v37 + `v38_esimd_reroute.py`
(exactly one tree change). The fp8-e4m3-nospec greedy "bimodality" (#18,
filed 2026-09-02, misattributed in v34 to FA2 split counts) is a
**run-to-run race in `eagle_ops.page_attn_decode`** — a compiled ESIMD
kernel from the original image's patch stack — and it never involved the
FA2 kernel at all. Fixed by keeping fp8* KV lanes out of the
`PAGED_ATTN_ESIMD_INSERTED_v1` gate so decode routes to the vxk FA2
varlen kernel, which is deterministic, tail-invariant, fp32-reference
correct to 1.4e-5 — and FASTER on this stack.

## Root cause (isolated by rigs, every step reproducible)

1. **Engine probe** (v38probe.py): same request x6, batch=1, prefix-cache
   hit — **4 distinct outputs** with the ESIMD gate active (true
   run-to-run nondeterminism, not batch-state). Divergence clustered at
   thin-margin tokens (char ~485-512, seqlen ≈ 117). Control lanes: 4bit
   TQ 10/10 stable (`0ce080630035` = prod ref), fp16/auto 10/10 stable —
   defect fp8-lane-exclusive.
2. **vxk FA2 exonerated** (v38krig.py): exact engine call shape, 50/50
   bit-identical across dtype x kv_len x split-config x det-flag; fp32
   ref correct to 1.4e-5; invalid-tail poison invariant (v38ref.py).
   Irrelevant anyway — see 3.
3. **The bypass discovered**: `flash_attn.py` ~1085
   `PAGED_ATTN_ESIMD_INSERTED_v1` routes fp16-Q + XPU-graph + head-256 +
   GQA>=2 decoder decode to
   `custom_esimd_kernels_vllm.eagle_ops.page_attn_decode` (compiled .so,
   no source). **fp8 decode never reached the FA2 kernel.**
4. **ESIMD kernel convicted** (v38erig.py, v38quant.py; fixed inputs,
   999 calls, bit-compare vs call-0):

   | case | eager | graph replay |
   |---|---|---|
   | fp8@511 | **969/998** differ | — |
   | fp8@512 | 132/998 | **998/998** (toggles 2 modes) |
   | fp8@1024 | 988/998 | — |
   | fp8@2048 | 150/998 | — |
   | fp8@117 | ~1/50 | — |
   | fp16@512 | 5/998 (1 elem) | — |

   Multistable discrete outputs, 1-2 fp16 ULP, ~100-478 elements — the
   same 235-element signature recurs (discrete modes, not analog noise).
   Invisible on fat-margin tokens; flips knife-edge argmax tokens ->
   coherent alternate generations = the historical "bimodality".
   **fp16 latent risk**: 5/998 single-element flips on the fp16 lane
   (same kernel, much rarer) — the gate stays for fp16/auto, documented.
5. **Why the v34/v38 exact-splits pin was doomed** (v38_exact_kv_splits.py,
   falsified): f8ref prompts ~12 tokens -> wkb=2 < 16 -> heuristic
   num_splits=1 — the split machinery is INACTIVE at the lengths where
   bimodality was observed. Split counts were never the mechanism.

## The fix

`v38_esimd_reroute.py` — extends the ESIMD gate with
`and not (kv_cache_dtype.startswith("fp8") and
VLLM_XPU_ALLOW_ESIMD_F8 != "1")`. Idempotent (contiguous marker),
fail-loud anchor count, compile-before-write. A/B force-back:
`VLLM_XPU_ALLOW_ESIMD_F8=1`. 4bit TQ lanes untouched (TQ backend
regardless); fp16/auto lanes untouched.

## Build & boot

Context `/root/build/v38ctx/` (Dockerfile + 1 script; FROM v37; grep
gates incl. v37 markers + contiguous v38 marker; py_compile; pycache
purge; `/root/.v38_baked`). Post-build cmp: baked-image `flash_attn.py`
md5 **MATCH** vs the live verified container (`1000effd…`).

```bash
setsid nohup bash /root/build/bootp.sh nospec '' <log> \
  '--kv-cache-dtype fp8_e4m3' llm-scaler-vllm-adv:v38 \
  > /root/<log>_boot.log 2>&1 < /dev/null &
```

## Verification (2026-09-03)

**Env-free patch verify on live v37 tree** (v38e: boot → docker cp →
apply → restart; NO env): pre-patch control 4 distinct P1 hashes on that
exact tree (defect reproduced) → post-patch 10/10 probe + f8ref exact +
2k 33.71 + 65k warm 28.24. Idempotency re-run prints "already patched".

**Baked v38 image battery** (v38f, fp8_e4m3 nospec, no env):
- probe 10/10 = `91d489262cb5`; f8ref `[91d489262cb5, 3705c4621a59,
  95e24129958b]` exact; 700-token gens x4 through the 512/1024 boundary
  hot zone (worst ESIMD rates) all `2cd43757adfc`.
- 2k warm 33.78 tok/s (ESIMD path: 23.99, +40%; 4bit parity 33.2-33.5).
- **65k warm 28.27 tok/s** (ESIMD 23.99 +18%; prod 4bit 22.28-22.36,
  **+27%**).
- conc16 16/16 coherent.

**4bit prod lane on v38 image — patch INERT** (prodv38i): probe 10/10 =
`0ce080630035` = prod reference bit-exact; f8ref stable; 2k 33.20
(repeat; first post-boot run 30.49 is a warmup artifact); 65k warm
22.36 (ref 22.28).

## Posture

- **Prod restored on the v38 image** (4bit TQ nospec, `prodv38i`),
  bit-exact vs the v31.1/v37 references. v38 is a strict superset of
  v37; v37 remains available.
- **fp8-e4m3 nospec is now a bit-stable lane and the fastest at long ctx
  (65k: 28.27 vs 22.36, +27%)** — prod-promotion candidate, pending
  user decision (fp8 KV = coarser quant than 4bit TQ at HALF the KV
  footprint of fp16; quality gates like coh-probe were parity-class in
  v38-era spot checks).
- #18 reclassified: local patch-stack kernel race, NOT intrinsic fp8,
  NOT split-count. KNOWN_ISSUES #18 rewritten accordingly; v34 closure
  annotated (its "cap-not-exact" finding about the C++ op stands, but
  the "bimodality persists under pin" observation was the ESIMD race —
  FA2 was never reached).
- fp16/auto lane keeps the ESIMD kernel (5/998 latent single-element
  nondeterminism at @512-class lengths; bit-stability demands on that
  lane would need the same reroute — left as-is, documented).
