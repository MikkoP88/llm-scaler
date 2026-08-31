#!/usr/bin/env python3
"""v29 P4c: oneCCL collectives at real decode/prefill shapes, 2 ranks x B70.

Run:  torchrun --standalone --nproc_per_node=2 dma_ccl.py
Timing: per-op blocking wall with a post-op .item() readback fence (P4a: copy
queue is not fenced by synchronize/Event on this build). For small messages an
UNFENCED variant is also printed so the fence overhead itself is visible.

Shapes (fp16, hidden=5120, k=5 spec -> 25600):
  decode AR  (8, 5120) (64, 5120) (8, 25600) (64, 25600)
  prefill AR (2048, 5120) (8192, 5120)
  all_gather on the same inputs.
AlgoBW = 2*S*(W-1)/W / t   (ring AR algorithmic bytes)
BusBW  = AlgoBW * W / (2*(W-1))  (NCCL bus-width convention)
"""
import os
import time

import torch
import torch.distributed as dist
import torch.xpu

try:
    import oneccl_bindings_for_pytorch  # noqa: F401  (legacy name)
except Exception:
    pass  # torch 2.11+xpu ships the ccl backend built-in as "xccl"

rank = int(os.environ.get("RANK", 0))
lr = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(lr)
dist.init_process_group(backend="xccl")
dev = f"xpu:{lr}"
W = dist.get_world_size()

if rank == 0:
    for k in ("CCL_ATL_TRANSPORT", "CCL_ZE_IPC_EXCHANGE", "CCL_TOPO_P2P_ACCESS",
              "CCL_ENABLE_SYCL_KERNELS", "ZE_AFFINITY_MASK", "CCL_PROCESS_LAUNCH"):
        print(f"env {k}={os.environ.get(k, '<unset>')}")
    print(f"world_size={W} backend=xccl torch={torch.__version__}")


def bench(name, call, fence_tensor, iters=100, warm=20, fence=True):
    for _ in range(warm):
        call()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
        if fence:
            _ = fence_tensor.flatten()[0].item()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


def run_ar(shape, dt=torch.float16):
    t_ = torch.randn(shape, dtype=dt, device=dev)
    dist.barrier()
    dtf = bench("AR-fenced", lambda: dist.all_reduce(t_), t_)
    dist.barrier()
    dtn = bench("AR-raw", lambda: dist.all_reduce(t_), t_, fence=False)
    S = t_.numel() * t_.element_size()
    algo = 2 * S * (W - 1) / W
    bus = algo * W / (2 * (W - 1))
    if rank == 0:
        print(f"AR  {str(tuple(shape)):>16} fp16: {dtf*1e6:9.1f} us fenced"
              f" | {dtn*1e6:9.1f} us raw"
              f" | algo {algo/dtf/1e9:7.2f} GB/s bus {bus/dtf/1e9:7.2f} GB/s")


def run_ag(shape, dt=torch.float16):
    t_ = torch.randn(shape, dtype=dt, device=dev)
    out = torch.empty((W,) + shape, dtype=dt, device=dev)
    dist.barrier()
    dtf = bench("AG-fenced", lambda: dist.all_gather_into_tensor(out, t_), out)
    dist.barrier()
    dtn = bench("AG-raw", lambda: dist.all_gather_into_tensor(out, t_), out, fence=False)
    S = t_.numel() * t_.element_size()
    algo = S * (W - 1) / W
    if rank == 0:
        print(f"AG  {str(tuple(shape)):>16} fp16: {dtf*1e6:9.1f} us fenced"
              f" | {dtn*1e6:9.1f} us raw"
              f" | algo {algo/dtf/1e9:7.2f} GB/s")


SHAPES = [(8, 5120), (64, 5120), (8, 25600), (64, 25600), (2048, 5120), (8192, 5120)]
for sh in SHAPES:
    run_ar(sh)
for sh in SHAPES:
    run_ag(sh)

# large message sweep for the link roofline
for mb in (64, 256):
    run_ar((mb * 1024 * 1024 // 2,))

dist.barrier()
if rank == 0:
    print("DONE")
dist.destroy_process_group()
