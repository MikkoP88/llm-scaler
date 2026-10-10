# P8c — output-quality + stall forensics (2026-10-10, post-P8 matrix)

## Chain of evidence (user directive: "do deep analyze i think there is lot stalling and AI stop always middle of run")

| # | Test | Config | Result |
|---|------|--------|--------|
| 1 | qual_probe 6 diverse prompts | standing best (fp16 mamba + LMC ratio 4) | 1 DEGENERATE (reasoning → `!!!` mid-answer), determinism DIVERGE |
| 2 | rerun_corrupt 5 prompts × 2 | same | 1/10 DEGENERATE (math3 r0, mid-answer at `'Subtract 7 from' + '!!!!!!!'`) |
| 3 | nonce_probe 20 unique nonces + `/tmp/lmc_off` + `/tmp/lmc_store_off` (LMCache fully neutralized in-process) | same engine, LMC dead | 1/20 DEGENERATE ([5], from FIRST token) — unpin count flat during test |
| 4 | **leg f16off** (quickleg: mamba float16, connector DELETED from boot, fresh engine) | pure engine | **0/20 clean** — but [11] **wall=137.66s** vs 2-4s typical (solo traffic!) |

## Verdicts

1. **`!!!` degeneration is ENGINE-COMPUTE class** — reproduces with LMCache fully disabled (test 3); absent in test 4 (0/20). Rate noisy 0-10%; NOT the v138c fp16→fp8 NaN-carry class (LMCNANDETECT=0 all legs). Bench legs were blind to it (max_tokens 32). Double-unpin (1232+ under mixed traffic, 0 in single-tenant) = separate real bug in `_lmc_flush_pending` lag-flush path; not the corruption driver.
2. **STALL CLASS CAPTURED** — 137s wall on pure-engine solo leg = the "AI stops middle of run" symptom. No abort, self-recovered. No xe/ccs reset (dmesg clean). Signature matches P85 async parked-context hang family (hang M1 class). **stallwatch.sh armed**: 10×20 nonce rounds + /metrics wedge detector (running>0 AND drafts zero-advance 30s → py-spy dump ALL pids ×2 + metrics + dmesg to /root/build/WD_stall_<ts>/).

## Artifacts
- /root/build/nonce_q_float16_off.out (0/20 + 137s stall row [11])
- /root/build/quickleg_f16off.out (leg transcript: EDIT connector=False, HEALTH tick=17, mech counts all 0)
- /root/build/stallwatch_probe.log + stallwatch_ticks.log + WD_stall_* (captures)
- /root/build/double_unpin_best.txt (banked earlier)
- Host: .tmp-p76/lmc/{quickleg.sh,stallwatch.sh,nonce_probe.py,rerun_corrupt.py,qual_probe.py}

## Next ladder
- f8off leg (mamba fp8_e4m3, no connector = pre-LMCache standing shape) → dtype axis for BOTH defects
- If stall reproduces → py-spy capture analysis (P85-style parked ctx vs scheduler pause)
- Bake BLOCKED until both classes root-caused/accepted.
