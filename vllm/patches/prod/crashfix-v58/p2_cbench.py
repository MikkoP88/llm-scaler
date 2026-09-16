#!/usr/bin/env python3
"""p2_cbench.py — L0 copy micro-bench for the P2 (NEO copy-tax) analysis.
Run inside the container: python3 p2_cbench.py <LABEL>
Measures per-op cost of the copy shapes the decode hot path issues:
  H2D pinned small (128B / 8KB), H2D pageable small, H2D one big (512KB),
  D2D small, D2H pinned small. Reports per-op us + effective MB/s.
"""
import sys
import time

import torch

LBL = sys.argv[1] if len(sys.argv) > 1 else "cbench"
dev = torch.device("xpu")
torch.xpu.set_device(0)

def bench(tag, fn, iters=400, sync_each=False):
    # warmup
    for _ in range(20):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"CBENCH {LBL} {tag} {dt*1e6:.1f}us")
    return dt

N_SMALL = 16        # int64 x16 = 128B
N_MED = 1024        # int64 x1024 = 8KB
N_BIG = 65536       # int64 x65536 = 512KB

src_p = torch.zeros(N_BIG, dtype=torch.int64, pin_memory=True)
src_pg = torch.zeros(N_BIG, dtype=torch.int64)          # pageable
dst = torch.zeros(N_BIG, dtype=torch.int64, device=dev)
src_d = torch.zeros(N_BIG, dtype=torch.int64, device=dev)

# H2D pinned, small/med/big
bench("h2d_pin_128B", lambda: dst[:N_SMALL].copy_(src_p[:N_SMALL], non_blocking=True))
bench("h2d_pin_8KB", lambda: dst[:N_MED].copy_(src_p[:N_MED], non_blocking=True))
bench("h2d_pin_512KB", lambda: dst.copy_(src_p, non_blocking=True))
# H2D pageable small
bench("h2d_page_128B", lambda: dst[:N_SMALL].copy_(src_pg[:N_SMALL], non_blocking=True))
# D2D small + big
bench("d2d_128B", lambda: dst[:N_SMALL].copy_(src_d[:N_SMALL], non_blocking=True))
bench("d2d_512KB", lambda: dst.copy_(src_d, non_blocking=True))
# D2H pinned small
bench("d2h_pin_128B", lambda: src_p[:N_SMALL].copy_(dst[:N_SMALL], non_blocking=True))
# 20-op decode-step emulation: 20 small pinned H2D then sync
def step20():
    for _ in range(20):
        dst[:N_SMALL].copy_(src_p[:N_SMALL], non_blocking=True)
bench("step20_small_h2d", step20, iters=40)
# same payload as one big copy
def step1():
    dst.copy_(src_p, non_blocking=True)
bench("step1_512KB_h2d", step1, iters=40)

print(f"CBENCH {LBL} DONE torch={torch.__version__}")
