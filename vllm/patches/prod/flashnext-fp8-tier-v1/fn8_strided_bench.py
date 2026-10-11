#!/usr/bin/env python3
"""contiguous vs transposed (strided) D2H/H2D copy-rate probe.

Engine's repack-restore copies crawl ~50 MB/s while contiguous pageable hits
8.8 GB/s — testing whether a transposed view explains it (fp8 blockscale
repack transposes w13/w2 scales/weights).
"""
import time

import torch

torch.xpu.init()
dev = torch.device("xpu")

# w13-shaped fp8: [512*1280, 2560] ~= 1.6 GiB
E, K, N = 512 * 1280, 2560, 2560
w = torch.randint(0, 200, (N, K), dtype=torch.uint8)  # [N, K] like down_proj
wd = w.to(dev)
print("tensor", tuple(wd.shape), wd.dtype, flush=True)


def bench(tag, t, to_cpu=True, reps=3):
    t.to("cpu") if to_cpu else None
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        d = t.to("cpu") if to_cpu else t.to(dev)
        torch.xpu.synchronize()
    dt = time.perf_counter() - t0
    gb = t.numel() / 1e9
    print(f"{tag}: {reps * gb / dt:.2f} GB/s  (contig={t.is_contiguous()})", flush=True)


bench("D2H_contig_", wd.clone().contiguous(), True)
bench("D2H_transposed", wd.t(), True)
wt = wd.t().contiguous().t()  # same logical layout, materialized backing
bench("D2H_t_contigbacking", wt, True)
h = torch.empty(N, K, dtype=torch.uint8)
bench("H2D_contig_", h, False)
ht = torch.empty(K, N, dtype=torch.uint8).t()  # host-side transposed view
bench("H2D_from_t_hostview", ht, False)
print("FN8_STRIDED_BENCH_OK")
