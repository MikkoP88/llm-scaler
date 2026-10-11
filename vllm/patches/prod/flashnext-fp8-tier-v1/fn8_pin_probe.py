#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P0 — torch pin_memory ceiling probe (fn8-p).

The UVA weight-offload path on XPU hard-asserts pinned tensors
(utils/torch_utils.py: `assert cpu_tensor.is_pinned()`), and the reference
run needs ~48-63 GiB of pinned host memory PER WORKER PROCESS.  The standing
LMCache law says SYCL USM pinned host allocs segfault above ~16-32 GiB per
process (probed 2026-10-08) — but that was the LMCache pool allocator path.
This probe measures the ceiling for plain torch .pin_memory(), which is what
UVAOffloader actually calls (`cpu_data.pin_memory()` in offloader/uva.py).

Host-side only: zero VRAM allocation, safe to run beside the live lane.
Escalates 4 GiB steps, keeping every buffer alive (UVA keeps all offloaded
params alive simultaneously), and reports the largest cumulative pinned
total reached.  A segfault (exit 139/-11) IS the measurement.
"""
import gc

import torch

GIB = 1024 ** 3


def main():
    assert torch.xpu.is_available(), "no XPU context — /dev/dri missing?"
    torch.xpu.init()
    print("XPU_INIT_OK devices", torch.xpu.device_count(), flush=True)

    live = []
    step_gib = 4
    max_gib = 68
    reached = 0
    for n in range(step_gib, max_gib + 1, step_gib):
        t = torch.empty(n * GIB // 1, dtype=torch.uint8)
        p = t.pin_memory()
        del t
        live.append(p)
        # touch one page per 2 GiB so the alloc is real, not lazily mapped
        for off in range(0, p.numel(), 2 * GIB):
            p[off] = 1
        reached = n
        print(f"PIN_OK {n} GiB cumulative, pinned_flag={p.is_pinned()}", flush=True)
        gc.collect()

    print(f"PIN_PROBE_COMPLETE survived {reached} GiB cumulative pinned", flush=True)
    print("PIN_PROBE_OK", reached, flush=True)


if __name__ == "__main__":
    main()
