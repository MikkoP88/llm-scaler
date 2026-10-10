

---

## FINAL VERDICT — ROOT CAUSE CONVICTED 2026-10-10 ~09:30 UTC

### Complete 12-leg bisect table

| # | leg | corrupt | verdict |
|---|---|---|---|
| 1a | lsv-test f16, connector off, 200 req | 8/200 | reproduces |
| 1b | lsv-test fp8_e4m3, connector off | 2/40 | dtype-independent |
| 1c | prefix-caching off | 1/40 | pfx-independent |
| 1d | debug-env stripped | 10/100 | dbg-independent |
| 2 | lmc_off + store_off sentinels | (reproduces) | cache-independent |
| 3 | lsv-pristine container from image | **0/40** | image clean |
| 6 | pristine worker files in lsv-test | 8/100 | worker patches innocent |
| 7 | pristine core files in lsv-test | 8/100 | core patches innocent |
| 8 | lsv-fresh: image + 8 patched files + lsv-test env | **0/100** | patch set innocent |
| 8R | redo w/ runtime verification (`v138 zeroers` markers logged, md5 match) | **0/100** | leg 8 valid |
| 9 | all compile caches purged in lsv-test | 8/100 | caches innocent |
| 10a | pristine adapter trio | 7/100 | adapter innocent |
| 10b | lmcache package moved away | 7/100 | lmcache pkg innocent |
| 11 | **numpy 2.3.5 restored** (pip had 2.2.6) | 2/100 | **numpy exonerated** |
| 12 | **oneCCL 2021.15 restored** (pip had 2021.17.2) | **0/100** | **ROOT CAUSE** |

### Mechanism

`pip install` during LMCache bring-up (Oct 8 21:07) pulled the standalone **oneccl 2021.17.2 wheel**, whose RECORD owns `/opt/venv/lib/libccl.so*` — **outside site-packages, invisible to top-level listing diffs**. It replaced the image's symlinks:

- image: `libccl.so → libccl.so.1 → /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0` (toolkit)
- after pip: real 331 MB wheel files + new `libccl.so.2` + `/opt/venv/lib/ccl/kernels/kernels.spv`

`ldd libtorch_xpu.so` resolves `libccl.so.1` from `/opt/venv/lib` → every TP=2 all-reduce since Oct 8 ran a CCL mismatched to the torch/XPU stack. Symptom signature: intermittent 4-10% garbage tokens, engine-wide 137-198 s self-recovering stalls (collective retries), zero dmesg/xe/ccs signature. Perfectly matches "AI stops in the middle of a run".

Legs 11/12 are the discriminative pair — single variable = CCL version:
numpy-restored + pip CCL = corrupt; numpy-restored + image CCL = clean.

### Fix (executed)

1. `rm` pip libccl files + `/opt/venv/lib/ccl/` + oneccl/oneccl_devel dist-infos; recreate image symlinks (inert leftover: `/opt/venv/lib/libccl.a`).
2. numpy restored to image 2.3.5 (pip downgrade 2.2.6; exonerated as corruption cause).
3. Post-pip verification law: `ls -la /opt/venv/lib/libccl.so*` must be symlinks; `import numpy` == 2.3.5.

### Consequences

- P8 perf ratios remain valid (baseline + LMCache legs shared the broken CCL).
- Absolute perf + the v138-era "in-compute fp8→fp8 NaN residual ~1/96" must be re-validated on clean CCL before bake.
- lsv-test restored: adapter v139 + patched connector + lmcache 0.5.5 back, connector ON leg = `lmcon` (this doc's final validation).
