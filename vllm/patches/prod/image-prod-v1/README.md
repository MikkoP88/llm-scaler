# image-prod-v1 — `llm-scaler-prod:v1`: first production image, new naming

## What

Content = **EXACTLY the certified `adv:v38` tree** — all keepers, nothing
failed, nothing diagnostic:

```
FROM llm-scaler-vllm-adv:v37        # all era-3 keeper bakes (v32..v37)
  + v38_esimd_reroute.py            # #18 fix (fp8 decode off the ESIMD path)
  = byte-identical to adv:v38 on all 19 divergent files
```

New bake marker `/root/.llm-scaler-prod_v1_baked`; `bootp.sh` marker glob
generalized to `/root/.*_baked` (host, 2026-09-03).

## Build gates (all passed 2026-09-03)

- grep gates: v38 reroute markers + v37/v33/v34 keeper markers in-tree
- py_compile + ast.parse on flash_attn.py, pycache purge
- **Tree-equality vs `adv:v38`: 19/19 md5 IDENTICAL** (harmonized/MANIFEST.md5)
- harm-vaudit shape: 16 modified + 2 added vs wheel RECORD (14/16 markers),
  same as every prod image since v37

## Validation battery ×5 (2026-09-03, host 10.20.3.65)

Lane = 4bit turboquant_4bit_nc nospec, block 512, graphs on (prod posture).
Baked images booted via `bootp.sh`; v31.1/v19/v14 booted RAW
(`serve_user_nospec.sh` directly — authentic trees, no boot-time patches).
probe10 = P1×10 temp0 mt160 sha256[:12]; 2k = bench_completions
steady_true (2nd run = warm); 65k = 900-tok long decode (2nd = warm);
conc16 = 16 concurrent identical temp-0 requests. Scripts:
`../harmonized/_work/harm-batt.sh` (+ `harm_probe10.py`, `harm_conc16.py`).

| image | boot | probe10 | 2k warm tok/s | 65k warm tok/s | conc16 |
|---|---|---|---|---|---|
| **llm-scaler-prod:v1** | bootp | 10/10 `0ce080630035` | 33.20 | 22.18 | 2 distinct (14+2) |
| adv:v37 (prev prod) | bootp | 10/10 `0ce080630035` | 33.21 | 22.19 | 2 distinct (15+1) |
| adv:v31.1 (raw tree) | raw | 10/10 `0ce080630035` | 33.23 | 22.20 | 2 distinct (12+4) |
| adv:v19 (raw tree) | raw | 10/10 `cb8c3851b897` | 33.34 | 22.18 | 3 distinct (13+2+1) |
| adv:v14 (pristine wheel) | raw | 10/10 `0ce080630035` | 33.18 | 22.18 | 2 distinct (15+1) |

(65k cold ≈ 11.5-11.6 on every image — warmup, not regression.)

## Findings

1. **prod:v1 ≡ v38 behavior**: probe hash, 2k, 65k all inside the certified
   envelope (v38 4bit lane: 2k 33.20, 65k 22.36 on the same battery).
2. **4bit nospec numerics frozen since the pristine wheel**: v14, v31.1,
   v37, prod:v1 all produce `0ce080630035` — every era-3 keeper is
   nospec-inert as designed. v19 differs (`cb8c3851b897`, era-1/2 overlay
   state, pre-v26 wheel rebuild) — the only numerics excursion in the
   whole lineage, self-consistent within the image.
3. **Perf flat v19→v37**: 2k 33.18-33.34, 65k 22.18-22.20 across five
   images — the nospec 4bit lane never regressed through 24 versions.
4. **conc16 temp-0 divergence is WHEEL-NATIVE**: present in pristine v14
   (15+1). Batch-composition-dependent outputs under concurrency
   (TQ 4bit decode padding path), unchanged-or-better through the lineage
   (v19 worst at 3-way). Not a regression from any patch; candidate for a
   future KNOWN_ISSUES entry if user-visible.

## Posture

Prod restored on `llm-scaler-prod:v1` (bootp nospec, single-cycle boot via
new marker; P1 = `0ce080630035` verified post-restore). Historical
`adv:v38` retained as provenance; future production = `llm-scaler-prod:vN`,
experimental = `llm-scaler-exp:<purpose>`.
