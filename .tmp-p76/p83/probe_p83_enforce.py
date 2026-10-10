#!/usr/bin/env python3
"""llm-scaler P83 step-1b: cap ENFORCEMENT at boundary crossing.

cap = reserved + 256 MiB, then 3x 128 MiB chunks (c2 exceeds cap):
  E1. c2 must raise a CATCHABLE RuntimeError from the allocator
  E2. driver free must never drop below (free_at_start - 256) while capped
  E3. after freeing a held chunk, a same-size alloc must succeed from the
      CACHED block (reserved flat, driver f untouched) -- the property that
      makes a cap safe under storm: pressure resolves inside the allocator.
  E4. a single alloc larger than the whole cap -> immediate clean OOM.
"""
import torch


def mb(x):
    return x >> 20


free0, total = torch.xpu.mem_get_info()
r0 = torch.xpu.memory_reserved()
CAP = r0 + (256 << 20)
torch.xpu.set_per_process_memory_fraction(CAP / total)
print("E cap=%d MiB frac=%.6f free0=%d" % (mb(CAP), CAP / total, mb(free0)))

held = []
for i in range(3):
    try:
        held.append(torch.empty(128 << 20, dtype=torch.uint8, device="xpu"))
        print("E1 c%d ok reserved=%d f=%d"
              % (i, mb(torch.xpu.memory_reserved()),
                 mb(torch.xpu.mem_get_info()[0])))
    except RuntimeError as e:
        print("E1 c%d RAISED %s: %s" % (i, type(e).__name__, str(e)[:140]))
        break

fmin = mb(torch.xpu.mem_get_info()[0])
print("E2 f_now=%d f_floor=%d preserved=%s" % (fmin, mb(free0) - 256,
                                               fmin >= mb(free0) - 256))

# E3: free one held chunk -> cached; same-size alloc must reuse it
del held[0]
r_pre = torch.xpu.memory_reserved()
f_pre = torch.xpu.mem_get_info()[0]
try:
    held.append(torch.empty(128 << 20, dtype=torch.uint8, device="xpu"))
    r_post = torch.xpu.memory_reserved()
    f_post = torch.xpu.mem_get_info()[0]
    print("E3 reuse reserved %d->%d f %d->%d (cached_reuse=%s)"
          % (mb(r_pre), mb(r_post), mb(f_pre), mb(f_post),
             mb(f_pre) == mb(f_post)))
except RuntimeError as e:
    print("E3 FAILED: %s" % str(e)[:140])

# E4: single alloc bigger than the cap -> immediate clean OOM
try:
    torch.empty(1 << 30, dtype=torch.uint8, device="xpu")
    print("E4 BIG ALLOC UNEXPECTEDLY SUCCEEDED")
except RuntimeError as e:
    print("E4 big-alloc OOM clean: %s" % str(e)[:140])
print("PROBE2_DONE")
