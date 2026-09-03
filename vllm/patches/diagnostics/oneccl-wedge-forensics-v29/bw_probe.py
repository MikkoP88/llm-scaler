#!/usr/bin/env python3
"""v29 P4a: PCIe/HBM bandwidth probe with EVENT timing (host-side perf_counter
+ synchronize() proved unreliable on this build). Also a matmul sanity anchor:
B70 fp16 ~ 40-60 TFLOPs => the timing method must reproduce that order."""
import time
import torch
import torch.xpu

dev = "xpu:0"
MB = 1024 * 1024


def timed(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) / 1000.0 / iters  # seconds


# --- sanity anchor: matmul TFLOPs ---
a = torch.randn((4096, 4096), dtype=torch.float16, device=dev)
b = torch.randn((4096, 4096), dtype=torch.float16, device=dev)
t = timed(lambda: a @ b, iters=10)
flops = 2 * 4096**3 / t
print(f"matmul 4096^3 fp16: {t*1e3:.2f} ms -> {flops/1e12:.1f} TFLOPs")

# --- transfer benches ---
n = 256 * MB // 2  # 256 MiB fp16
host = torch.empty((n,), dtype=torch.float16, pin_memory=True)
gpu = torch.empty((n,), dtype=torch.float16, device=dev)
gpu2 = torch.empty((n,), dtype=torch.float16, device=dev)

t = timed(lambda: gpu.copy_(host, non_blocking=True), iters=5)
print(f"H2D 256MiB:  {256/t:8.2f} GB/s")
t = timed(lambda: host.copy_(gpu, non_blocking=True), iters=5)
print(f"D2H 256MiB:  {256/t:8.2f} GB/s")
t = timed(lambda: gpu2.copy_(gpu), iters=5)
print(f"D2D 256MiB:  {256/t:8.2f} GB/s  (device-local)")

small_gpu = torch.empty((5120,), dtype=torch.float16, device=dev)
t = timed(lambda: small_gpu.cpu(), iters=200)
print(f"D2H 10KB (mamba-align sync shape): {t*1e6:.1f} us/call")
