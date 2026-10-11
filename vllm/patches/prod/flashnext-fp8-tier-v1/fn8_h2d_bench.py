#!/usr/bin/env python3
"""pageable vs pinned H2D/D2H bandwidth microbench (256 MiB chunks)."""
import time

import torch

MB = 1024 * 1024
N = 256 * MB
reps = 4

torch.xpu.init()
dev = torch.device("xpu")


def bench(tag, t, to_dev=True):
    # warmup
    d = t.to(dev, non_blocking=True) if to_dev else t.to("cpu", non_blocking=True)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        d = t.to(dev, non_blocking=True) if to_dev else t.to("cpu", non_blocking=True)
        torch.xpu.synchronize()
    dt = time.perf_counter() - t0
    gbps = reps * N / dt / 1e9
    print(f"{tag}: {gbps:.2f} GB/s", flush=True)
    return d


pageable = torch.empty(N, dtype=torch.uint8)
pageable.random_()
print("dev", torch.xpu.get_device_name(0), flush=True)

d = bench("H2D_pageable", pageable, True)
bench("D2H_pageable", d, False)

pinned = pageable.pin_memory()
dp = bench("H2D_pinned", pinned, True)
bench("D2H_pinned", dp, False)
print("FN8_H2D_BENCH_OK")
