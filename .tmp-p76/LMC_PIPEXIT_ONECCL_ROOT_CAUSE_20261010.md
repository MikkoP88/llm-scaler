# ROOT CAUSE — "AI stops in the middle of a run": pip-overwritten oneCCL silently corrupting TP all-reduce

**Status: CLOSED 2026-10-10 (~09:30 UTC). Convicted by 12-leg empirical bisect; fix deployed and verified (0/100 clean).**
**Scope: lsv-test on 10.20.3.65 (all LMCache-program legs since Oct 8 21:07). Image llm-scaler-exp:v1.2.32 was NEVER affected.**

---

## 1. Symptom class (what the user saw)

Two co-occurring defects on the lsv-test lane, starting after the LMCache bring-up installs (evening of Oct 8):

1. **Output degeneration** — 4-10% of requests return garbage (`'!!!!'x24`-style) instead of the answer. Intermitting, no crash, engine stays HEALTHY.
2. **Mid-run stalls** — engine-wide freezes of 137.66 s and 198.30 s measured on pure-engine legs (no connector), self-recovering, then normal throughput resumes. This is the "AI stop always middle of run" experience.

No signature anywhere else: dmesg clean (zero xe/ccs/reset events across all legs), no guc devcoredumps, no worker deaths, /metrics drafts keep advancing during probes.

## 2. Root cause (mechanism)

During LMCache bring-up, a `pip install` on Oct 8 21:07 resolved the standalone **oneCCL 2021.17.2 wheel** as a dependency. That wheel's RECORD owns files **outside site-packages**, directly in the venv lib dir:

```
/opt/venv/lib/libccl.so, libccl.so.1, libccl.so.1.0     (331 MB each, real files)
/opt/venv/lib/libccl.so.2, libccl.so.2.0               (141 KB)
/opt/venv/lib/libccl_legacy*, libccl_openmp*
/opt/venv/lib/ccl/kernels/kernels.spv                  (5.2 MB SPIR-V GPU kernels)
```

The production image ships those paths as **symlinks into the Intel oneAPI toolkit**:

```
/opt/venv/lib/libccl.so   -> libccl.so.1                                  (image, Aug 26)
/opt/venv/lib/libccl.so.1 -> /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0
```

pip deleted the symlinks and wrote the 2021.17.2 wheel files in their place. `ldd libtorch_xpu.so` resolves `libccl.so.1` from `/opt/venv/lib` first — so **every vLLM TP=2 all-reduce from Oct 8 onward ran on a CCL that does not match the torch/IPEX stack the image was built against** (2021.15). A collective-library mismatch produces exactly this signature: intermittent silently-wrong partial sums (garbage tokens, no error), occasional collective retries/timeouts (multi-minute engine-wide stalls that then recover), and nothing at the kernel/driver level.

Because the damage lives at `/opt/venv/lib` (not site-packages), a top-level site-packages listing diff — the first bisect tool used — was structurally blind to it.

The same pip session also **downgraded numpy 2.3.5 → 2.2.6** and protobuf/prometheus_client/opentelemetry/grpcio (see §4); those were separately tested and exonerated as corruption causes, but numpy was restored to the image version anyway.

## 3. Evidence chain (the 12-leg bisect)

Every leg below ran the same nonce quality probe (degenerate-output detector). The pattern: **everything code-side was innocent; the container's binary state was guilty.**

| # | leg | corrupt | verdict |
|---|---|---|---|
| 1a | lsv-test, mamba float16, connector OFF | 8/200 | reproduces without LMCache |
| 1b | lsv-test, mamba fp8_e4m3, connector OFF | 2/40 | dtype-independent |
| 1c | prefix-caching OFF | 1/40 | prefix-cache-independent |
| 1d | debug env stripped | 10/100 | debug-env-independent |
| 2 | lmc_off + store_off sentinels (LMCache fully inert) | reproduces | cache-independent |
| 3 | lsv-pristine: fresh container from image | **0/40** | image clean |
| 6 | pristine worker files restored into lsv-test | 8/100 | worker patches innocent |
| 7 | pristine core files restored into lsv-test | 8/100 | core patches innocent |
| 8 | lsv-fresh: fresh container + the 8 patched engine files + lsv-test env | **0/100** | the whole patch set innocent |
| 8R | leg-8 redo with runtime verification (md5 match + `llm-scaler v138 mamba zeroers` markers in boot log) | **0/100** | leg 8 valid, patched code provably active |
| 9 | all compile caches purged (.triton/.cache/torchinductor/pycache/tmp-uuid) | 8/100 | caches innocent |
| 10a | pristine adapter trio restored | 7/100 | adapter innocent |
| 10b | lmcache package moved away entirely | 7/100 | lmcache package innocent |
| 11 | numpy restored to image 2.3.5 (pip had 2.2.6) | 2/100 | **numpy exonerated** |
| 12 | oneCCL restored to image 2021.15 (symlinks recreated, wheel files removed) | **0/100** | **ROOT CAUSE** |

Legs 11 and 12 are the discriminative pair: identical in every respect except the CCL version (both had numpy 2.3.5). Leg 11 corrupt, leg 12 clean ⇒ the CCL overwrite is both necessary and sufficient within the tested space. Statistical weight: legs 8+8R+3 = **0/240 clean** on image-CCL containers vs 37 degenerates in ~380 requests on pip-CCL lsv-test.

## 4. Full pip delta (for the record)

- **Replaced (version changes):** numpy 2.3.5→2.2.6, protobuf 7.36.0→6.33.6, prometheus_client 0.26.0→0.24.1, opentelemetry 1.44.0→1.40.0 (9 pkgs), grpcio 1.83.0→1.84.0.
- **Added (lmcache dep tree):** lmcache 0.5.5 (+ its compiled .so), aiofile/aiofiles, awscrt, caio, redis, google_* (bigtable/auth/crc32c/iam), cupy 14.2.0 + cupyx, cuda_python/cuda_bindings/cuda_core/cuda_pathfinder/cufile, nvtx, importlib_metadata+zipp, oneccl+oneccl_devel 2021.17.2, pytest 9.1.1.
- **The killer entry:** oneccl 2021.17.2 — the only one that writes outside site-packages into the venv lib dir where the dynamic linker resolves CCL for libtorch_xpu.

## 5. Fix (executed on lsv-test, 2026-10-10 09:23 UTC)

1. Removed every file pip's oneccl wheel owned: `/opt/venv/lib/libccl.so*`, `/opt/venv/lib/ccl/`, `oneccl{,_devel}-2021.17.2.dist-info`. (Inert leftover kept: `/opt/venv/lib/libccl.a`.)
2. Recreated the image symlink set: `libccl.so → libccl.so.1 → /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0`.
3. Verified: `ls -la` shows symlinks; `ldd libtorch_xpu.so | grep ccl` resolves to `/opt/venv/lib/.../libccl.so.1`; boot log clean; **0/100 nonce probe**.
4. numpy restored to image 2.3.5 (tar from image: `numpy/`, `numpy-2.3.5.dist-info`, `numpy.libs/`).
5. lmcache 0.5.5 restored from `lmcache.off`; patched v139 adapter + patched connector re-installed; connector-ON serve shape restored (`--kv-transfer-config ... LMCacheConnectorV1 ... mamba_pool_ratio 2` + `--disable-hybrid-kv-cache-manager`).

## 6. Standing laws (prevention)

1. **NEVER `pip install oneccl`/`oneccl-bindings*` into these containers.** If a package resolves it, block it (install with `--no-deps` + manually satisfy, or vendor into a separate venv).
2. **Post-pip verification ritual for lsv-test:** `ls -la /opt/venv/lib/libccl.so*` must show SYMLINKS (targets under `/opt/intel/oneapi/ccl/2021.15/`); `docker exec lsv-test /opt/venv/bin/python -c "import numpy; print(numpy.__version__)"` must print 2.3.5. Any deviation = restore before serving.
3. **Quality probe after any venv mutation:** `/root/build/nonce_probe.py 100` — require 0 degenerates.
4. Site-packages listing diffs are blind to venv-lib-level overwrites — when hunting binary-state issues, diff `/opt/venv/lib` directly and read RECORD files of suspicious dist-infos.
5. For the bake image: bake from a verified-clean venv state (this law chain as the checklist).

## 7. Consequences for prior results

- **P8 4-leg matrix ratios stand** (rehit 4.8-5.6×, churn 4.7-5.4×, zero regressions): baseline and LMCache legs both ran on the same (broken) CCL — the comparison is internally valid.
- **Absolute numbers were handicapped** on both sides; re-validate absolute perf on clean CCL before the bake table is finalized.
- **v138-era "in-compute fp16→fp8 NaN origination residual ~1/96"** was measured under corrupted all-reduce and must be re-qualified (may vanish entirely).
- **Stall forensics re-scope:** the 137-198 s stall class is presumed CCL-retry in origin; confirm it does not reproduce on clean CCL during soak legs (py-spy stallwatch stays available at /root/build/stallwatch.sh).

## 8. Artifacts

- Bisect scripts: `.tmp-p76/lmc/{quickleg,pristine,workerleg,freshleg,freshleg2,purgeleg,legboot,numpyleg,cclrestore,restorelmc}.sh` (banked on host `/root/build/`)
- Probe outputs: `/root/build/nonce_*.out` (q_float16_off, q_fp8_e4m3_off, pfxoff/dbgoff variants, w_*, fresh8, fresh2, purged, adapter_pristine, nolmcache, numpyfix, cclfix)
- Evidence: `/root/build/sp_image.txt` vs `sp_test.txt` (site-packages diff), oneccl RECORD inspection, `ls -la` symlink states before/after
- Master forensics doc: `/root/build/P8C_CORRUPTION_STALL_FORENSICS_20261010.md` (legs 1-12 + this verdict)
- Memory law: `pip-oneccl-tp-corruption-law` (auto-memory)
