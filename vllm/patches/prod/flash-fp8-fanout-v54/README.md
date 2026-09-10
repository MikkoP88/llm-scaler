# flash-fp8-fanout-v54 — position-sliced flash fan-out for fp8 paged verify

The era-4 production change: root-cause fix for the fp8_e4m3+MTP
long-context collapse. Campaign record and full evidence:
`../../diagnostics/fp8-mtp4-v1/REPORT.md` (§4.3 root cause, §5 fix, §6
gates, §7 case axes, §8 soak/determinism, §9 v6 bake).

## What it does

Decomposes the UNIFORM q_len fp8 paged speculative-verify attention
into q_len flash q=1 calls — one per position, sharing the paged KV
pool, block table and per-tensor descales; per-position causal limits
computed device-side (`seqused_k - (q_len - 1 - pos)`, graph-capture
safe); output rows written back position-sliced (row-layout contract
preserved).

Why: the only measured-healthy fp8 paged shape on this stack is flash
q=1 (0.075 µs/KVtok, nospec decode lane). The verify alternatives were
the v51 Triton vector kernel (1.98) and stock flash q5 varlen (3.35).
Fan-out lands the mtp4 verify step at 0.414 µs/KVtok (e4m3) / 0.283
(e5m2) — turning fp8+spec from net-negative at ≥128k to superior at
every length and client count.

- Knob: `VLLM_XPU_FP8_FANOUT` (default set by `--fanout-default` at
  bake; **1 in all prod images**). Rollback per boot: `=0`.
- Target: `vllm/v1/attention/backends/flash_attn.py`; route gate
  requires `kv_cache_dtype` str startswith `fp8` + paged + uniform
  q_len — tq4nc lanes are untouched.
- Patcher protocol: anchors must match exactly once, `py_compile`
  gates the write.

## Bake lineage (production images)

| image | Dockerfile | delta | sha256 |
|---|---|---|---|
| `llm-scaler-exp:fp8-mtp4-v5` | `diagnostics/fp8-mtp4-v1/Dockerfile.v4` (`--build-arg FANOUT=1`, FROM v1) | fan-out default ON | `e536666de558…` |
| `llm-scaler-exp:fp8-mtp4-v6` == `llm-scaler-exp:v1.2.7` | `diagnostics/fp8-mtp4-v1/Dockerfile.v6` (FROM v5) | + `ENV VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1` (e5m2 lane selectable with no env; inert on e4m3 — single read site, harmonized patch 03) | `7cf3d51cfc60…` |

Host bake context: `/root/build/fp8m4bake/`. The canonical campaign
copy of the patcher also stays in `diagnostics/fp8-mtp4-v1/` (bake
context reference); this dir is the production ledger entry.

## Certification

- v5: dV5-cert knee (7 lengths × 3 reps, no env — baked default
  proven), diag9/9b case axes (conc C2/C8, e5m2), diag10 soak (10
  cycles, 0 faults) + 3-boot determinism. REPORT §6-§8.
- v6: `master_diag11.sh` dual-lane byte-parity — dW6-e4m3 full knee
  vs dV5-cert refs (baked ENV inert) + dW6-e5m2 vs diag9b dX2 refs
  (env-passed == env-baked). REPORT §9.
