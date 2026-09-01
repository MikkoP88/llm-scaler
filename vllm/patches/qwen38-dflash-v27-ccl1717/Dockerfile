FROM llm-scaler-vllm-adv:v27
# llm-scaler-vllm-adv:v27-ccl1717 — oneCCL 2021.15 -> 2021.17.2 upgrade arm
# (Tier-3, KNOWN_ISSUES #11: piecewise graph replay x spec x >=32k ctx
# collective livelock on 2x Arc Pro B70)
#
# Rationale: upstream oneCCL #215's version matrix marks oneCCL 2021.17.2 +
# SYCL RT 2025.3.2 (our exact RT, unchanged here) as PASS for the hang class;
# our 2021.15 hangs. 2022.x = HANG per the same matrix, so 2021.17.2 is the
# terminal upgrade target. oneAPI 2025.3 toolkits pin exactly ccl 2021.17
# (hpc-kit Depends line), i.e. the validated pairing.
#
# Artifact: intel-oneapi-ccl-2021.17-2021.17.2-5_amd64.deb from
# https://apt.repos.intel.com/oneapi/pool/main/ (SHA256 verified in-build;
# classic component package, layout-compatible with the base image's 2021.15
# GitHub-offline install: opt/intel/oneapi/ccl/2021.17/{env,etc,lib,share}).
#
# Deltas vs the 2021.15 install (image-wide sweep found exactly 3 ref sites):
#   1. /opt/intel/oneapi/ccl/2021.17 tree added (2021.15 kept for rollback)
#   2. ccl/latest symlink + /root/.bashrc vars.sh source line repointed
#   3. /opt/venv/lib bindings repointed; NEW libccl.so.2 link added
#      (2021.17's libccl.so.1 NEEDEDs libccl.so.2; no RPATH on any ccl lib,
#      so every search dir serving libccl.so.1 must also serve libccl.so.2)
#   4. /etc/ld.so.conf.d/ccl-2021.17.conf + ldconfig: the serving process
#      runs with NO LD_LIBRARY_PATH/CCL_ROOT (non-interactive bash -c; the
#      .bashrc vars.sh line only fires in interactive shells). Proven on the
#      running prod container: the vllm process maps the ccl lib via an
#      explicit path, and its 2021.15-era deps resolve only because torch
#      preloads libsycl.so.8/libsvml/libimf from torch/lib. libccl.so.2 is
#      NEW (not preloaded by torch, no RPATH, absent from the ld cache) ->
#      register the 2021.17 lib dir in the ldconfig cache. Dir contains only
#      ccl libs, so cache blast radius is limited to ccl sonames.
# SYCL RT untouched: torch 2.11.0+xpu's bundled libsycl.so.8 satisfies the
# same NEEDED set as 2021.15 (svml/irng/imf/intlc/sycl all unchanged).
#
# RESULT (2026-09-01, GPU window, k4 + FULL_DECODE_ONLY graphs, TP=2): #11
# wedge NOT FIXED — the livelock is oneCCL-version-independent on this stack.
# Behavioral deltas of 2021.17.2: drmfd IPC exchange REMOVED (hard boot error;
# must set CCL_ZE_IPC_EXCHANGE=pidfd), pidfd now works on kernel 6.17 (#213
# fix effective), and #212's CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 boots (2021.15
# died at init_device) but makes wedge onset ~40x FASTER — the open IPC handle
# cache is load-bearing for survival. Arms: pidfd wedge w073435 (long_exp iter
# 1 @39 chunks, self-heals after ~minutes but first post-recovery request
# returns degenerate text; #12 k4 corruption persists); pidfd+cache-off wedge
# w075336 (fox iter 1 @ONE chunk). #215's 2021.17.2+SYCL RT 2025.3.2 PASS row
# does not cover our trigger class. Full data: KNOWN_ISSUES #11 v29e,
# qwen38-dflash-v29/upstream_oneccl_ticket.md post-log 2026-09-01.

COPY intel-oneapi-ccl-2021.17-2021.17.2-5_amd64.deb /tmp/ccl1717/intel-oneapi-ccl-2021.17-2021.17.2-5_amd64.deb

RUN CCLDIR=/opt/intel/oneapi/ccl && \
    DEB=/tmp/ccl1717/intel-oneapi-ccl-2021.17-2021.17.2-5_amd64.deb && \
    echo "744cedaf2cd244309f6d1b0bb3236939a8f6bed7406b0da57dd4dda9f3296b67  $DEB" | sha256sum -c - && \
    dpkg-deb -f "$DEB" Package Version && \
    dpkg-deb -x "$DEB" /tmp/ccl1717/x && \
    cp -a /tmp/ccl1717/x/opt/intel/oneapi/ccl/2021.17 "$CCLDIR/" && \
    ln -sfn /opt/intel/oneapi/ccl/2021.17 "$CCLDIR/latest" && \
    sed -i 's|ccl/2021\.15/env/vars\.sh|ccl/2021.17/env/vars.sh|' /root/.bashrc && \
    ln -sfn /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1.0       /opt/venv/lib/libccl.so.1 && \
    ln -sfn /opt/intel/oneapi/ccl/2021.17/lib/libccl_openmp.so.0.1 /opt/venv/lib/libccl_openmp.so.0 && \
    ln -sfn /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.2.0       /opt/venv/lib/libccl.so.2 && \
    echo "/opt/intel/oneapi/ccl/2021.17/lib" > /etc/ld.so.conf.d/ccl-2021.17.conf && \
    ldconfig && \
    test "$(ldconfig -p | grep -cE 'libccl\.so\.[12] ')" -ge 2 && \
    ldconfig -p | grep -E 'libccl\.so\.[12] |libccl_openmp' && \
    test -f "$CCLDIR/2021.17/env/vars.sh" && \
    test -f "$CCLDIR/2021.17/lib/libccl.so.2.0" && \
    readelf -d "$CCLDIR/2021.17/lib/libccl.so.1.0" | grep -q 'libccl.so.2' && \
    readlink "$CCLDIR/latest" | grep -q 2021.17 && \
    grep -q 'ccl/2021.17/env/vars.sh' /root/.bashrc && \
    ! grep -q 'ccl/2021\.15' /root/.bashrc && \
    readlink /opt/venv/lib/libccl.so.1     | grep -q 2021.17 && \
    readlink /opt/venv/lib/libccl.so.2     | grep -q 2021.17 && \
    readlink /opt/venv/lib/libccl_openmp.so.0 | grep -q 2021.17 && \
    rm -rf /tmp/ccl1717 && \
    echo "v27-ccl1717 overlay done (oneCCL 2021.17.2, SYCL RT 2025.3.2 unchanged)"
