#!/usr/bin/env python3
"""v39_check — bit-exactness + perf A/B for the v39a nibble-split patch.

Loads the pristine pre-patch module (backup written by v39_tq_nibble.py)
and the patched live module side by side, drives both through every
launcher entry point on IDENTICAL synthetic data for all TQ presets
(4bit_nc, k3v4_nc, 3bit_nc, k8v4), and compares outputs bitwise.

Also times the 4bit_nc + k8v4 decode launchers at 16k context as an
early perf preview (the live boot bench remains the real gate).

Run INSIDE the container AFTER v39_tq_nibble.py:
  python3 /tmp/v39_check.py            # old = <target>.pre39
  python3 /tmp/v39_check.py <old.py>   # explicit old copy
"""
import importlib.machinery
import importlib.util
import math
import sys
import time
import zlib

import torch

P = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/"
     "triton_turboquant_decode.py")
OLD_PATH = sys.argv[1] if len(sys.argv) > 1 else P + ".pre39"
D = 128
BLOCK = 16
DEV = torch.device("xpu")

assert torch.xpu.is_available(), "needs XPU"
assert OLD_PATH != P


def load(name, path):
    # .pre39 suffix has no inferable loader — pass SourceFileLoader explicitly
    spec = importlib.util.spec_from_file_location(
        name, path, loader=importlib.machinery.SourceFileLoader(name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old = load("tq_old", OLD_PATH)
new = load("tq_new", P)
print(f"loaded old={OLD_PATH}")
print(f"loaded new={P}")

torch.manual_seed(20260903)

# orthonormal rotation (QR of gaussian — deterministic under the seed)
Pi, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float32))
Pi = Pi.to(DEV)
PiT = Pi.T.contiguous()

results = []


def make_case(mse_bits, vqb, key_fp8, num_tokens, seed):
    """Synthetic cache + block table for one preset."""
    mse_bytes = math.ceil(D * mse_bits / 8)
    kps = D if key_fp8 else mse_bytes + 2  # fp8 K: no norm; MSE K: +fp16 norm
    val_bytes = D if vqb == 8 else math.ceil(D * vqb / 8)
    slot = kps + val_bytes + 4 + 2  # scale + zero (fp16 x2), pad
    n_blocks = math.ceil(num_tokens / BLOCK) + 4
    g = torch.Generator().manual_seed(seed)
    kv = torch.randint(0, 256, (n_blocks, BLOCK, 4, slot),
                       dtype=torch.uint8, generator=g)
    # keep the synthetic fp16 fields finite (clamp exponent bits): norm at
    # [mse_bytes, +2) when MSE, scale/zero at [kps+val_bytes, +4)
    if not key_fp8:
        kv[:, :, :, mse_bytes + 1] &= 63
    kv[:, :, :, kps + val_bytes + 1] &= 63
    kv[:, :, :, kps + val_bytes + 3] &= 63
    perm = torch.randperm(n_blocks, generator=g)[: n_blocks]
    bt = perm.to(torch.int32).view(1, -1)
    return kv.to(DEV), bt.to(DEV), kps, mse_bytes, val_bytes


def centroids_for(mse_bits):
    # DISTINCT values — equal-valued centroids would hide index bugs
    c = torch.linspace(-1.5, 1.5, 2 ** mse_bits, dtype=torch.float32)
    return c.to(DEV)


def check(tag, a, b):
    ok = torch.equal(a, b)
    results.append((tag, ok))
    if not ok:
        diff = (a.float() - b.float()).abs()
        print(f"  {tag}: FAIL max|diff|={diff.max().item():.3e} "
              f"ndiff={(diff > 0).sum().item()}/{diff.numel()}")
    return ok


def run_all(tag, mse_bits, vqb, key_fp8, nc):
    """One preset x norm-correction: decode, MQ (causal+non-causal), dequant."""
    T = 1000
    kv, bt, kps, mse_bytes, val_bytes = make_case(mse_bits, vqb, key_fp8,
                                                  T, seed=zlib.crc32(tag.encode()))
    cents = centroids_for(mse_bits)
    scale = 1.0 / math.sqrt(D)
    B, Hq, Hk = 2, 8, 4
    bt2 = torch.cat([bt, bt], 0).contiguous()  # [2, M] same blocks: fine
    seqs = torch.tensor([T, 17], dtype=torch.int32, device=DEV)
    q = (torch.randn(B, Hq, D, dtype=torch.float16) * 0.1).to(DEV)

    kw = dict(mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
              key_fp8=key_fp8, norm_correction=nc, PiT=PiT)
    o = old.triton_turboquant_decode_attention(
        q, kv, bt2, seqs, Pi, cents, scale, **kw).clone()
    n = new.triton_turboquant_decode_attention(
        q, kv, bt2, seqs, Pi, cents, scale, **kw).clone()
    check(f"{tag} decode", o, n)

    # MQ: B==1, cached prefix 500, 5 query rows
    q0 = torch.tensor([500], dtype=torch.int32, device=DEV)
    qm = (torch.randn(5, Hq, D, dtype=torch.float16) * 0.1).to(DEV)
    for nc_flag in (False, True):
        om = old.triton_turboquant_mq_decode_attention(
            qm, kv, bt, q0, Pi, cents, scale, **kw,
            non_causal=nc_flag).clone()
        nm = new.triton_turboquant_mq_decode_attention(
            qm, kv, bt, q0, Pi, cents, scale, **kw,
            non_causal=nc_flag).clone()
        check(f"{tag} mq{'_nc' if nc_flag else ''}", om, nm)

    # full dequant (direct kernel launch)
    def dequant(mod):
        K = torch.zeros(B, Hk, T, D, dtype=torch.float16, device=DEV)
        V = torch.zeros(B, Hk, T, D, dtype=torch.float16, device=DEV)
        mod._tq_full_dequant_kv[(T, B * Hk)](
            kv, bt2, cents, K, V,
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            kv.stride(0), kv.stride(1), kv.stride(2), bt2.stride(0),
            HEAD_DIM=D, BLOCK_SIZE=BLOCK, NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes, KPS=kps, VQB=vqb,
            VAL_DATA_BYTES=val_bytes, MSE_BITS=mse_bits,
            KEY_FP8=1 if key_fp8 else 0, BLOCK_D=D,
            NORM_CORRECTION=1 if nc else 0, FP8_E4B15=0,
        )
        return K.clone(), V.clone()

    ko, vo = dequant(old)
    kn, vn = dequant(new)
    check(f"{tag} dequantK", ko, kn)
    check(f"{tag} dequantV", vo, vn)


for nc in (False, True):
    run_all(f"4bit_nc{'+nc' if nc else ''}", 4, 4, False, nc)
run_all("k3v4_nc", 3, 4, False, False)   # old MSE path + new V4 path
run_all("3bit_nc", 3, 3, False, False)   # both old paths (regression)
run_all("k8v4", 4, 4, True, False)       # KEY_FP8 + new V4 path

# ---------------------------------------------------------------------------
# perf preview: decode launcher at 16k context, B=1, deep splits.
# 4bit_nc exercises K(MSE)+V fixes; k8v4 (KEY_FP8) isolates the V fix.
# ---------------------------------------------------------------------------
def bench_16k(mse_bits, vqb, key_fp8, seed, reps=30):
    T = 16384
    Hq, Hk = 32, 16
    kv, bt, kps, mse_bytes, val_bytes = make_case(mse_bits, vqb, key_fp8, T,
                                                  seed=seed)
    cents = centroids_for(mse_bits)
    scale = 1.0 / math.sqrt(D)
    seqs = torch.tensor([T], dtype=torch.int32, device=DEV)
    q = (torch.randn(1, Hq, D, dtype=torch.float16) * 0.1).to(DEV)

    def bench(mod):
        for _ in range(3):
            mod.triton_turboquant_decode_attention(
                q, kv, bt, seqs, Pi, cents, scale, mse_bits=mse_bits,
                key_packed_size=kps, value_quant_bits=vqb,
                key_fp8=key_fp8, PiT=PiT)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            mod.triton_turboquant_decode_attention(
                q, kv, bt, seqs, Pi, cents, scale, mse_bits=mse_bits,
                key_packed_size=kps, value_quant_bits=vqb,
                key_fp8=key_fp8, PiT=PiT)
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    ms_o = bench(old)
    ms_n = bench(new)
    ms_o2 = bench(old)  # re-time to bound clock drift
    label = "k8v4" if key_fp8 else "4bit_nc"
    print(f"PERF {label} decode @16k (B=1,Hq={Hq}): "
          f"old={ms_o:.2f}/{ms_o2:.2f} ms  new={ms_n:.2f} ms  "
          f"delta={100*(ms_o+ms_o2-2*ms_n)/(ms_o+ms_o2):+.1f}%")


bench_16k(4, 4, False, seed=777)
bench_16k(4, 4, True, seed=778)

nfail = sum(1 for _, ok in results if not ok)
for tag, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {tag}")
print(f"V39_CHECK: {len(results)-nfail}/{len(results)} bit-exact, {nfail} fail")
raise SystemExit(1 if nfail else 0)
