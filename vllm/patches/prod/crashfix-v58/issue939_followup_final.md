**Two updates from the Battlemage / vLLM side: (1) a source bisect pinning what NEO ≥ 26.18 actually changes relative to this race — it's one copy-path commit, and it doubles as a fast reproducer; (2) kernel 6.17 vs 7.2/7.3 behavior on this workload.**

## 1. The hang-avoidance in NEO ≥ 26.18 is a single copy-path commit — accidentally shipped

We bisected 26.14 → 26.18 at source granularity (10 steps, 66 min, each step = full NEO build + single-variable boot + our fp8 numerics/throughput reference; decode-throughput verdict). Result is a single-commit cliff:

- First bad: **`76e8bd47f0` "feature: enable copy via lock pointer for non-compressed resources on xe2+"** (Related-To: NEO-14823)
- Parent `392de6fe7a`: 51.2 tok/s (good) → culprit `76e8bd47f0`: 34.6 tok/s (**−32 %**) on our steady-state decode reference; at tag level this shows as the −24 % of 26.18
- Mechanism per the commit itself: copies of **non-compressed** xe2+ allocations stop going through GPU copy/blit submissions and are routed to a CPU locked-pointer path instead (GPU copy retained only for compressed resources)

Then we built the decoupled variant: **26.18 tag + `git revert 76e8bd47f0`** (clean revert, 7 files, −306 lines). Numerics bit-identical, and decode throughput fully restored to the 26.14 class (50.5–50.8 tok/s solo, conc4 148.7 — same band as stock). And the wedge came back **immediately**: first sustained-load round, TTF **5 m 51 s** — below the fastest of our seven stock-26.14 events (14 min). Devcoredump: same `Reason: LR job cleanup, guc_id=22` on the same card — now **7/7 captured events with the identical reason** across GuC 70.44.1 and 70.72.1, NEO 26.14 and this hybrid build.

So on this hardware the commit is simultaneously:
- **sufficient** for hang-avoidance (26.18 with it: 18-round / ~3 h sustain battery, 18/18 clean), and
- **necessary** (26.18 minus it: wedge in < 6 min), and
- **exactly** the −24 %..−32 % throughput cost.

Our reading: routing non-compressed copies CPU-side removes the GPU copy/blit submissions that interleave with our max-rate small-kernel dispatch during graph-replay bursts — which is what starves the `LR job cleanup` race window. In other words, the only thing currently standing between this workload and GSD-12919 on 26.18+ is a copy-path change that Intel's own CI reverted twice.

Upstream history of that feature, for triangulation:

```
master:  7874534f31 (Apr 23, land)
         52900009d1 (Apr 24, REVERT by Compute-Runtime-Validation bot)
         2a028be930 (Apr 24, re-land)
         dc3a4a946d (Apr 28, REVERTED AGAIN)
release: 76e8bd47f0 (cherry-pick into the 26.18 branch — never reverted, active in the shipped tag)
```

Two asks:
- Was either master revert (52900009d1 / dc3a4a946d) driven by failures this issue would recognize (hangs, validation timeouts)? If yes, that's independent confirmation of the same race from your CI.
- Reverting `76e8bd47f0` gives you a **sub-6-minute reproducer** of the exact `LR job cleanup` / exec-queue-reset wedge on BMG with sustained small-kernel submission — vs 14 min–2 h 52 m on stock. We can share the bisect logs, boot/battery scripts, and all 7 devcoredumps.

## 2. Mainline 7.2/7.3: deterministic TDR at boot, and clean recovery doesn't save Level Zero apps

We also tested whether the upstream timeout-recovery rework (`b1107d085e`, `a889e9b06b`, in v7.2+) changes the picture. Single-variable matrix:

| Kernel | NEO | GuC fw | Outcome |
|---|---|---|---|
| 6.17.0-1010-intel | 26.14 / 18 / 27 / 31 | 70.44.1 / 70.72.1 | boots; `LR job cleanup` wedge under sustained load (14 min–2 h 52 m; < 6 min without `76e8bd47f0`) |
| 7.2.6 | 26.14 | 70.72.1 | TDR at first profiling/warmup pass → `UR_RESULT_ERROR_DEVICE_LOST` |
| 7.2.6 | 26.31 | 70.72.1 | graph capture passes; TDR at `_xpu_C.topk_topp_sampler` with `UR_RESULT_ERROR_OUT_OF_RESOURCES` |
| 7.2.6 | 26.31 | 70.72.1 | same with native sampler (forced) |
| 7.3-rc3 | 26.14 | 70.44.1 | TDR, same stage |

Reproduced 6/6 boots; independent of NEO version, GuC firmware, and sampler implementation. The failing op (`topk_topp_sampler`, a tiny reduction-style kernel launched at max dispatch rate through torch/IPEX) is the same workload shape that races on 6.17 for hours — the 7.x kernel trips it immediately instead.

And the 7.2 recovery machinery works as designed — it just doesn't help this stack: the TDR resolves in ~13 ms (full GT reset, clean resubmit, zero `LR job cleanup` wedges — that devcoredump reason is gone on 7.2), but NEO immediately returns `UR_RESULT_ERROR_DEVICE_LOST` to the process. Level Zero apps here do not survive even one clean GT reset, so for this workload class a userspace-visible crash is guaranteed on any TDR regardless of recovery quality. Prevention (the race itself) remains the only real fix — which loops back to §1.

Happy to attach any subset of: the 7 devcoredumps + frozen GuC ring dumps, bisect/build logs, the 7.2/7.3 dmesg+devcoredump pairs, or to test any NEO/xe tag you point us at on this hardware.
