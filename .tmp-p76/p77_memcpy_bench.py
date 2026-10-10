#!/usr/bin/env python3
"""P77 v52k microbench: old serial batch_memcpy_kernel vs new 2-D chunked.

Correctness: byte-exact vs source (incl. odd sizes, size=1, exact-block
multiples) and no writes past `size`.
Timing: XPU events, 5 warmup (JIT) + 30 reps, min/p50/max ms + GB/s.
Workload mirrors the live storm: ~90 slots of ~917KB (GDN conv+ssm across
layers) + small/odd tails. Adaptive entry count if free VRAM is tight.
"""
import importlib.util
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

SPEC_PATH = "/root/mamba_utils_v52k.py"

# ---- OLD kernel (verbatim from live mamba_utils.py) -----------------------
@triton.jit
def old_batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size
        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def old_batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    grid = (batch,)
    old_batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=1024)


# ---- NEW module ------------------------------------------------------------
spec = importlib.util.spec_from_file_location("mamba_utils_v52k", SPEC_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dev = torch.device("xpu")
try:
    free_b, total_b = torch.xpu.mem_get_info()
except (AttributeError, RuntimeError):
    free_b, total_b = 4 * 1024**3, 8 * 1024**3
print(f"VRAM free={free_b/1e9:.2f}GB total={total_b/1e9:.2f}GB")

SLOT = 917_504  # live GDN slot size
if free_b > 2 * 1024**3:
    sizes_list = [SLOT] * 90 + [4096] * 3 + [SLOT + 3] * 2 + [1, 8192, 8192 * 3]
else:
    sizes_list = [SLOT] * 24 + [4096] * 3 + [SLOT + 3] * 2 + [1, 8192, 8192 * 3]

n = len(sizes_list)
row_bytes = max(sizes_list)
total_mb = sum(sizes_list) / 1e6
print(f"workload entries={n} slot={SLOT} row_pad={row_bytes} total={total_mb:.1f}MB")

torch.manual_seed(77)
src = torch.randint(0, 256, (n, row_bytes), dtype=torch.uint8, device=dev)
sizes = torch.tensor(sizes_list, dtype=torch.int32, device=dev)

src_ptrs = torch.tensor(
    [src.data_ptr() + i * row_bytes for i in range(n)],
    dtype=torch.uint64, device=dev)


def fresh_dst():
    d = torch.zeros((n, row_bytes), dtype=torch.uint8, device=dev)
    return d, torch.tensor(
        [d.data_ptr() + i * row_bytes for i in range(n)],
        dtype=torch.uint64, device=dev)


def verify(dst, tag):
    torch.xpu.synchronize()
    for i in range(n):
        s = sizes_list[i]
        if not torch.equal(src[i, :s], dst[i, :s]):
            print(f"FAIL {tag} entry={i} size={s} DATA MISMATCH")
            return False
        if s < row_bytes and dst[i, s:].any().item():
            print(f"FAIL {tag} entry={i} size={s} PAST-END WRITE")
            return False
    print(f"PASS {tag} byte-exact, no past-end writes")
    return True


def bench(fn, dst, dst_ptrs, max_size, tag, reps=30, warm=5):
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    for _ in range(warm):
        fn(src_ptrs, dst_ptrs, sizes, max_size) if max_size else fn(
            src_ptrs, dst_ptrs, sizes)
        dst.zero_()
    torch.xpu.synchronize()
    times = []
    for _ in range(reps):
        dst.zero_()
        ev0.record()
        fn(src_ptrs, dst_ptrs, sizes, max_size) if max_size else fn(
            src_ptrs, dst_ptrs, sizes)
        ev1.record()
        torch.xpu.synchronize()
        times.append(ev0.elapsed_time(ev1))
    times.sort()
    t50 = times[len(times) // 2]
    print(f"{tag}: p50={t50:.3f}ms min={times[0]:.3f}ms max={times[-1]:.3f}ms "
          f"eff={total_mb / t50 * 1e3 / 1e3:.2f}GB/s")
    return t50


# --- correctness first ---
dst, dst_ptrs = fresh_dst()
old_batch_memcpy(src_ptrs, dst_ptrs, sizes)
ok_old = verify(dst, "OLD")
dst, dst_ptrs = fresh_dst()
mod.batch_memcpy(src_ptrs, dst_ptrs, sizes, int(max(sizes_list)))
ok_new = verify(dst, "NEW(v52k, host max_size)")
dst, dst_ptrs = fresh_dst()
mod.batch_memcpy(src_ptrs, dst_ptrs, sizes)  # max_size=None fallback path
ok_fb = verify(dst, "NEW(v52k, None fallback)")

if not (ok_old and ok_new and ok_fb):
    raise SystemExit("CORRECTNESS FAILED — no swap")

# --- timing ---
dst, dst_ptrs = fresh_dst()
t_old = bench(old_batch_memcpy, dst, dst_ptrs, None, "OLD serial 1KB")
dst, dst_ptrs = fresh_dst()
t_new = bench(mod.batch_memcpy, dst, dst_ptrs, int(max(sizes_list)),
              "NEW chunked 8KB 2D")
print(f"SPEEDUP x{t_old / t_new:.2f}")
print("VERDICT:", "SWAP_OK" if t_new <= t_old else "NO_GAIN")
