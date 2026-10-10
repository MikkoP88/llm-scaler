#!/usr/bin/env python3
"""llm-scaler P83 step-1 probe: XPU allocator cap / headroom mechanisms.

Throwaway container from the prod image, ONE card if the selector is honored,
max footprint ~1.5 GiB. Answers:
  A. API surface (set_per_process_memory_fraction / mem_get_info / empty_cache)
  B. baseline accounting (free/total/reserved reference frames)
  C. cap semantics + behavior at hit: catchable OOM? does DRIVER free stay
     above the preserved headroom (the anti-wedge property)?
  D. empty_cache: does it return cached blocks to the driver (f rises)?
"""
import inspect

import torch


def mb(x):
    return x >> 20


print("A torch=%s xpu=%s ndev=%d" % (torch.__version__,
                                     torch.xpu.is_available(),
                                     torch.xpu.device_count()))
for i in range(torch.xpu.device_count()):
    print("A dev%d %s" % (i, torch.xpu.get_device_name(i)))
_api = [n for n in ("set_per_process_memory_fraction", "mem_get_info",
                    "memory_reserved", "memory_allocated", "empty_cache",
                    "memory_stats") if hasattr(torch.xpu, n)]
print("A api=%s" % ",".join(_api))
if hasattr(torch.xpu, "set_per_process_memory_fraction"):
    try:
        print("A sig=%s" % inspect.signature(
            torch.xpu.set_per_process_memory_fraction))
    except Exception as e:  # pragma: no cover - signature introspection
        print("A sig=? %r" % e)

free, total = torch.xpu.mem_get_info()
r0 = torch.xpu.memory_reserved()
print("B free=%d total=%d reserved=%d MiB" % (mb(free), mb(total), mb(r0)))

# C: cap at reserved + 1 GiB, then walk 8x 128 MiB chunks into it
CAP = r0 + (1 << 30)
frac = CAP / total
print("C fraction=%.6f cap=%d MiB (= reserved+1024)" % (frac, mb(CAP)))
try:
    torch.xpu.set_per_process_memory_fraction(frac)
    print("C set OK")
except Exception as e:
    print("C set FAILED: %r" % e)
    raise SystemExit(3)

held, oom = [], 0
for i in range(8):
    try:
        # 128 MiB chunk
        held.append(torch.empty(128 << 20, dtype=torch.uint8, device="xpu"))
    except RuntimeError as e:
        oom = 1
        print("C OOM at chunk %d: %s" % (i, str(e)[:120]))
        break
    f2, _ = torch.xpu.mem_get_info()
    print("C chunk%d held=%d reserved=%d f=%d"
          % (i, mb(sum(t.numel() for t in held)),
             mb(torch.xpu.memory_reserved()), mb(f2)))
f_end = torch.xpu.mem_get_info()[0]
print("C result oom=%d f_end=%d headroom_preserved=%s"
      % (oom, mb(f_end), mb(f_end) > 400))

# D: release then empty_cache -> driver free should rise
del held
f_before = torch.xpu.mem_get_info()[0]
torch.xpu.empty_cache()
f_after = torch.xpu.mem_get_info()[0]
print("D empty_cache f %d -> %d MiB (delta %d)"
      % (mb(f_before), mb(f_after), mb(f_after - f_before)))
print("D final reserved=%d f=%d"
      % (mb(torch.xpu.memory_reserved()), mb(torch.xpu.mem_get_info()[0])))
print("PROBE_DONE")
