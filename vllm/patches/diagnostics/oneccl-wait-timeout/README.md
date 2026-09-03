# oneCCL `ccl_executor::wait()` timeout - source-build recipe (UNBUILT)

Lane: convert the #11 permanent device livelock into a recoverable error at
the oneCCL executor level. This is the remaining upstream-independent fix
surface after the version axis (2021.15 / 2021.17.2 / 2022.x-per-#215) and
every env workaround were exhausted (KNOWN_ISSUES #11 v29b-v29e).

NOT BUILT YET. Recipe below replicates the proven vxk wheel-builder pattern
(patches/prod/gdn-spec-oob-fix-v26/build_vxk_wheel.sh) and the proven v27-ccl1717
library-overlay pattern (patches/diagnostics/oneccl-1717-upgrade-v27c/Dockerfile).

## Status: mitigation, not cure

Even with a working timeout, recovery is only HALF-validated: on
2021.17.2+pidfd the wedge self-heals and the first post-recovery request
returned degenerate text ("!!!!!!!!", w073435). A timeout must therefore be
paired with request-retry discipline, and post-recovery output must be
probe-checked before trusting the engine again.

## Patch sketch (against oneCCL 2021.17.2 source)

Target: `src/coll/exec/exec_worker.cpp` (`ccl_executor::wait()` /
`ccl_executor::wait_progress()`) - the host-side spin that never returns
while the device storms. Add a wall-clock deadline:

```cpp
// CCL_WAIT_TIMEOUT_SEC (0 = infinite, oneCCL-default behavior).
// When the executor spin exceeds the deadline, return ccl::error(status)
// instead of spinning forever, so the caller (ProcessGroupXCCL) surfaces a
// RuntimeError and vLLM's StepWatchdog / frw5 restart path can recover.
```

- read `getenv("CCL_WAIT_TIMEOUT_SEC")` once (parse like other env knobs,
  env/... parser style);
- in the wait loop: on each iteration where no progress is observed for
  > deadline seconds, log `CCL_ERROR ... executor wait timeout after Ns
  (no progress)` and return `ccl::error(ccl::status_runtime_error)`;
- default OFF (0) to stay byte-compatible with upstream behavior.

## Build

```bash
git clone https://github.com/uxlfoundation/oneCCL -b 2021.17.2 /root/build/oneccl-src
# apply the wait-timeout patch (to be authored when this lane is funded)
docker run --rm -v /root/build/oneccl-src:/src -v /root/build/ccl-out:/out \
  intel/omix:0.1.0-devel-ubuntu24.04 bash -c '
    set -exo pipefail
    source /opt/intel/oneapi/setvars.sh --force
    cd /src && mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF
    cmake --build . -j 16 && cmake --install . --prefix /out
  '
```

## Overlay into an image

Follow oneccl-1717-upgrade-v27c exactly: copy the built
`lib/{libccl.so.1.0,libccl.so.2.0,libccl_openmp.so.0.1}` over a
`ccl/2021.17` tree, repoint `latest` + `.bashrc` + venv symlinks, and keep
the `/etc/ld.so.conf.d/ccl-2021.17.conf` ldconfig entry (libccl.so.2
NEEDED-chain, no RPATH). SHA-pin whatever is produced.

## Validation

Boot with `CCL_ZE_IPC_EXCHANGE=pidfd CCL_WAIT_TIMEOUT_SEC=60`, provoke the
standard battery, and require: (1) the timeout error surfaces in the worker
log at the wedge instead of an eternal spin, (2) the engine reaches the
StepWatchdog/frw5 restart, (3) post-recovery coherent output on coh_probe
(the w075336-class degenerate-output risk), (4) zero false-positive
timeouts across a clean canonical battery (the spin must be
distinguishable from a merely slow collective - hence progress-gating, not
a hard wall-clock kill).
