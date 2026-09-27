#!/usr/bin/env python3
"""t_jit_probe.py — cross-process triton disk-cache semantics probe.

Run twice in SEPARATE processes with identical args:
  proc1: first-ever launch of this kernel -> compile (~0.2-1s incl. store)
  proc2: if the triton disk cache serves cross-process hits -> <0.05s
Timing + the /root/.triton/cache dir delta decide load/store semantics.
"""
import os
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _probe_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)


def main() -> None:
    n = 1024
    x = torch.arange(n, dtype=torch.float32, device="xpu")
    y = torch.arange(n, dtype=torch.float32, device="xpu")
    o = torch.empty(n, dtype=torch.float32, device="xpu")
    torch.xpu.synchronize()
    t0 = time.time()
    _probe_add[(1,)](x, y, o, n, BLOCK=1024)
    torch.xpu.synchronize()
    t1 = time.time()
    print(f"JITPROBE pid={os.getpid()} launch_to_sync={t1 - t0:.3f}s sum={o.sum().item():.0f}")


if __name__ == "__main__":
    main()
