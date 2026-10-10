#!/usr/bin/env python3
"""llm-scaler P83 v2: launch-headroom guard with split knobs (P82 fix lever).

P82 verdict: deep-storm allocator growth (reserved 19->35.5 GB/card WITHIN one
storm) drives driver free to a 38-65 MB floor; a big GDN prefill-chunk launch
whose driver submit transient cannot be satisfied silently fails and the CS
wedges. Fix = keep the driver permanently solvent with TWO knob-gated edits:

  Site 1 (v1/worker/xpu_worker.py, init_device tail): cap the per-process
  caching allocator via torch.xpu.set_per_process_memory_fraction so reserved
  can never eat the headroom. Probe-certified (torch 2.11.0+xpu): cap is
  enforced with CATCHABLE OOMs, driver free floors at exactly total-cap, and
  freed cached blocks are reused without touching the driver (pressure
  resolves INSIDE the allocator -- the anti-wedge property).

  Site 2 (v1/worker/gpu_worker.py, determine_num_available_blocks tail): shave
  the KV pool sizing by VLLM_XPU_POOL_SHAVE_MB so the deep-storm live set
  (weights + pool + ~8 GiB chunk activations, measured 30.31 GiB/card at leg-1
  death) fits under the cap with ~2.3 GiB of allocator-internal cache margin.

v2 split-knob law (leg-1 post-mortem): a single HEADROOM knob forced cap and
pool to move together. Cap = total - HEADROOM must stay ABOVE the healthy
storm live set (P82 healthy legs ran reserved up to 31.6 GB/card), while pool
must shrink to make room INSIDE the cap -> the two need independent dials.

  VLLM_XPU_HEADROOM_MB   allocator cap headroom (cap = total - knob); 0 = off
  VLLM_XPU_POOL_SHAVE_MB extra KV pool shave; UNSET = fall back to
                         VLLM_XPU_HEADROOM_MB (v1/leg-1 behaviour); 0 = off

Leg 2 posture: HEADROOM=1024 (cap 30.89 GiB) + POOL_SHAVE=2048 (pool
8.29 -> 6.29 GiB; live peak ~29.3 GiB; driver floor >= ~1.0 GiB guaranteed).
Strict int parse on purpose -- a typo'd knob fails the boot loudly.

Apply AFTER patch_p80_event_ladder.py + patch_p82_argsnap.py (disjoint files;
order irrelevant in practice, kept for discipline). Idempotent: SKIP(already)
on re-run. Fresh containers only -- v1-patched files must be restored from
the image before re-applying (md5 will differ from v1 law).

Usage (in container):
  /opt/venv/bin/python /tmp/patch_p83_headroom.py image
"""
import hashlib
import os
import py_compile
import sys

MARK = "llm-scaler P83"

CAP_BLOCK = '''
        # llm-scaler P83 launch-headroom guard: cap the per-process caching
        # allocator so the driver always keeps VLLM_XPU_HEADROOM_MB of device
        # memory free for launch/submit transients (P82 allocation-ceiling
        # law). Cap forces the allocator to reclaim cached blocks internally
        # instead of starving the driver. Default 0 = inert.
        _p83_hr = int(os.environ.get("VLLM_XPU_HEADROOM_MB", "0") or 0)
        if _p83_hr > 0:
            _p83_free, _p83_total = torch.xpu.mem_get_info()
            _p83_cap = max(_p83_total - (_p83_hr << 20), 1 << 30)
            torch.xpu.set_per_process_memory_fraction(
                _p83_cap / _p83_total, self.device
            )
            logger.info(
                "P83 headroom guard: cap=%.2f GiB of total=%.2f GiB "
                "(headroom %d MB, free at init %.2f GiB)",
                _p83_cap / (1 << 30), _p83_total / (1 << 30),
                _p83_hr, _p83_free / (1 << 30),
            )  # llm-scaler P83
'''

SHAVE_BLOCK = '''
        # llm-scaler P83 v2 (split knobs): shave the KV pool by
        # VLLM_XPU_POOL_SHAVE_MB (fallback: VLLM_XPU_HEADROOM_MB = v1/leg-1
        # behaviour) so the deep-storm live set -- weights + pool + ~8 GiB
        # chunk activations, 30.31 GiB/card measured at leg-1 death -- fits
        # under the allocator cap with cache margin. P82 measured ~9.5 GB/card
        # of post-plan allocator growth within one storm; the boot profiler
        # cannot predict it. Default 0 = inert.
        _p83_shave = os.environ.get("VLLM_XPU_POOL_SHAVE_MB")
        if _p83_shave is None:
            _p83_shave = os.environ.get("VLLM_XPU_HEADROOM_MB", "0")
        _p83_shave = int(_p83_shave or 0)
        if _p83_shave > 0 and self.device.type == "xpu":
            self.available_kv_cache_memory_bytes = max(
                self.available_kv_cache_memory_bytes - (_p83_shave << 20), 0
            )
            logger.info(
                "P83: KV pool shaved by %d MB pool-shave -> %.2f GiB",
                _p83_shave, self.available_kv_cache_memory_bytes / (1 << 30),
            )  # llm-scaler P83
'''

# unique anchors, verified count==1 at patch time
XPU_ANCHOR = (
    "        # Initialize workspace manager\n"
    "        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1\n"
    "        init_workspace_manager(self.device, num_ubatches)\n"
)
GPU_ANCHOR = (
    "        self.available_kv_cache_memory_bytes = (\n"
    "            self.requested_memory\n"
    "            - profile_result.non_kv_cache_memory\n"
    "            - cudagraph_memory_estimate_applied\n"
    "        )\n"
)


def _ins_after(src, anchor, block, tag):
    n = src.count(anchor)
    assert n == 1, "%s: anchor count %d != 1" % (tag, n)
    return src.replace(anchor, anchor + block, 1)


def _patch(path, anchor, block, tag):
    src = open(path).read()
    if MARK in src:
        print("SKIP(already) %s" % path)
        return
    src = _ins_after(src, anchor, block, tag)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True, cfile="/tmp/p83c_%s.pyc" % tag)
    n = src.count(MARK)
    print("PATCHED %s md5 %s markers=%d"
          % (path, hashlib.md5(open(path, "rb").read()).hexdigest(), n))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode != "image":
        print("usage: patch_p83_headroom.py image")
        sys.exit(2)
    base = "/opt/venv/lib/python3.12/site-packages/vllm"
    _patch(os.path.join(base, "v1/worker/xpu_worker.py"),
           XPU_ANCHOR, CAP_BLOCK, "xpu_cap")
    _patch(os.path.join(base, "v1/worker/gpu_worker.py"),
           GPU_ANCHOR, SHAVE_BLOCK, "gpu_shave")


if __name__ == "__main__":
    main()
