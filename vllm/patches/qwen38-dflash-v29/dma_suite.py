#!/usr/bin/env python3
"""v29 P4b: DMA GPU<->GPU suite, fence-safe timing.

P4a finding: on this build torch.xpu.synchronize()/Event do NOT fence the copy
queue (copies reported impossible 13 TB/s). Fence strategy here: after each
iteration, block on a 1-element readback (.item()) of the DESTINATION tensor —
a synchronous D2H enqueued behind the copy on the same in-order queue forces
completion of the whole prior DAG on that device.

Measures (fp16 unless noted):
  anchor   4096^3 matmul on xpu:0            (validates the stack: ~40-60+ TF)
  peer     can_device_access_peer(0,1)       (P2P allowed at driver level?)
  d2d0     xpu:0 -> xpu:0  256 MiB copy      (HBM path; HBM traffic = 2S)
  x01      xpu:0 -> xpu:1  256 MiB copy      (link crossing = S)
  x10      xpu:1 -> xpu:0  256 MiB copy
  viahost  xpu:0 -> pinned host -> xpu:1     (two hops; sum and per-leg)
  smalld2h 10 KB .cpu()                      (mamba-align sync shape 5120 fp16)
"""
import time
import torch
import torch.xpu

MB = 1024 * 1024
D0, D1 = "xpu:0", "xpu:1"


def fence_ok(name, fn, probe, iters=10, warm=3):
    for _ in range(warm):
        fn()
        _ = probe.flatten()[0].item()
    torch.xpu.synchronize(0)
    torch.xpu.synchronize(1)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        _ = probe.flatten()[0].item()
    torch.xpu.synchronize(0)
    torch.xpu.synchronize(1)
    return (time.perf_counter() - t0) / iters


print("torch", torch.__version__, "| devices:", torch.xpu.device_count())

# --- anchor: validates the runtime is alive and sane ---
a = torch.randn((4096, 4096), dtype=torch.float16, device=D0)
b = torch.randn((4096, 4096), dtype=torch.float16, device=D0)


def timed_mm(iters=10, warm=2):
    for _ in range(warm):
        a @ b
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        a @ b
    e.record()
    torch.xpu.synchronize()
    return s.elapsed_time(e) / 1000.0 / iters


t = timed_mm()
print(f"anchor matmul 4096^3 fp16: {t*1e3:.2f} ms -> {2*4096**3/t/1e12:.1f} TFLOPs")

# --- peer access probe ---
try:
    print("can_device_access_peer(0,1):", torch.xpu.can_device_access_peer(0, 1))
except Exception as ex:
    print("can_device_access_peer: API unavailable ->", ex)

# --- buffers ---
n = 256 * MB // 2  # 256 MiB as fp16 elements
h = torch.empty((n,), dtype=torch.float16, pin_memory=True)
g0a = torch.randn((n,), dtype=torch.float16, device=D0)
g0b = torch.empty((n,), dtype=torch.float16, device=D0)
g1 = torch.empty((n,), dtype=torch.float16, device=D1)

S = 256.0  # MiB

t = fence_ok("d2d0", lambda: g0b.copy_(g0a), g0b)
print(f"d2d0  xpu:0->xpu:0  {S:5.0f} MiB: {t*1e3:8.2f} ms  xfer {S/1024/t:7.2f} GB/s  (HBM traffic ~2x)")

t = fence_ok("x01", lambda: g1.copy_(g0a), g1)
print(f"x01   xpu:0->xpu:1  {S:5.0f} MiB: {t*1e3:8.2f} ms  link {S/1024/t:7.2f} GB/s")

t = fence_ok("x10", lambda: g0b.copy_(g1), g0b)
print(f"x10   xpu:1->xpu:0  {S:5.0f} MiB: {t*1e3:8.2f} ms  link {S/1024/t:7.2f} GB/s")


def via_host():
    h.copy_(g0a, non_blocking=True)
    g1.copy_(h)


t = fence_ok("viahost", via_host, g1)
print(f"viah  0->host->1    {S:5.0f} MiB: {t*1e3:8.2f} ms  link {S/1024/t:7.2f} GB/s (2 hops, per-hop ~half)")

# per-leg
t1_ = fence_ok("legA", lambda: h.copy_(g0a, non_blocking=True), h)
t2_ = fence_ok("legB", lambda: g1.copy_(h), g1)
print(f"        legs: D2H {S/1024/t1_:7.2f} GB/s   H2D {S/1024/t2_:7.2f} GB/s")

# --- small D2H sync latency (mamba-align shape) ---
small = torch.randn((5120,), dtype=torch.float16, device=D0)
for _ in range(20):
    small.cpu()
t0 = time.perf_counter()
N = 200
for _ in range(N):
    small.cpu()
dt = (time.perf_counter() - t0) / N
print(f"smalld2h 5120xfp16 .cpu(): {dt*1e6:8.1f} us/call (blocking; incl. ~0 copy time)")
