#!/usr/bin/env python3
"""v38 kernel isolation rig: does the XPU FA2 paged-decode kernel itself
produce bit-identical outputs for bit-identical inputs?

Runs INSIDE lsv-test (docker exec) against the engine's idle GPUs.
Engine geometry (Qwen3.8-27b-fp8, TP=2 per-rank): 12 Q heads, 2 KV heads,
head_dim 256, block_size 512. Repeats the SAME call 50x and compares bits.

Axes:
  kv_len: 117 (f8ref flip zone, single-split), 2080 (33 KV tiles — the
          kMinBlocksForSplit boundary), 4160 (solid multi-split)
  kv dtype: fp8_e4m3 (suspect) vs fp16 (control)
  num_splits_kv: None (heuristic) vs 32 (exact pin)
  deterministic: False vs True (interface flag)
"""
import sys
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DEV = "xpu"
HQ, HKV, D, BS = 12, 2, 256, 512
SCALE = D ** -0.5
N_ITER = 50


def run_case(kv_len, kv_dtype, splits, det, seed=0):
    torch.manual_seed(seed)
    nb = (kv_len + BS - 1) // BS + 2
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    q = (torch.randn(1, HQ, D, generator=g) * 0.1).to(torch.float16).to(DEV)
    kc = (torch.randn(nb, BS, HKV, D, generator=g) * 0.1).to(torch.float16)
    vc = (torch.randn(nb, BS, HKV, D, generator=g) * 0.1).to(torch.float16)
    if kv_dtype == "fp8":
        kc = kc.to(torch.float8_e4m3fn).to(DEV)
        vc = vc.to(torch.float8_e4m3fn).to(DEV)
        # per-TENSOR scalar scales: zero-stride single-f32 view (0-dim)
        kd = torch.ones((), dtype=torch.float32, device=DEV)
        vd = torch.ones((), dtype=torch.float32, device=DEV)
    else:
        kc = kc.to(DEV)
        vc = vc.to(DEV)
        kd = vd = None
    cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    sk = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)

    def call():
        return flash_attn_varlen_func(
            q, kc, vc, 1, cu, kv_len,
            seqused_k=sk, softmax_scale=SCALE, causal=False,
            block_table=bt, window_size=(-1, -1),
            k_descale=kd, v_descale=vd,
            num_splits_kv=splits, deterministic=det,
            is_mix_batch=False)

    outs = []
    for _ in range(N_ITER):
        o = call()
        torch.xpu.synchronize()
        outs.append(o.clone())
    ref = outs[0]
    mism = [i for i, o in enumerate(outs) if not torch.equal(o, ref)]
    # max ulp-ish diff for mismatched
    md = 0.0
    if mism:
        d = (outs[mism[0]].float() - ref.float()).abs()
        md = d.max().item()
    tag = f"kv={kv_len} dt={kv_dtype} splits={splits} det={det}"
    print(f"KRIG[{tag}] mismatch_iters={len(mism)}/{N_ITER}"
          f" first={mism[:6] if mism else '-'} maxdiff={md:.3e}", flush=True)
    return len(mism) == 0


if __name__ == "__main__":
    cases = [
        # fp8 KV, heuristic splits — the stock engine condition
        (117, "fp8", None, False),
        (2080, "fp8", None, False),
        (4160, "fp8", None, False),
        # fp8 KV, pinned 32 — the v38 condition
        (117, "fp8", 32, False),
        (2080, "fp8", 32, False),
        # fp8 KV, deterministic flag
        (117, "fp8", None, True),
        # fp16 KV control
        (117, "fp16", None, False),
        (2080, "fp16", None, False),
    ]
    if len(sys.argv) > 1:
        cases = [c for c in cases if str(c[0]) in sys.argv[1:]]
    for c in cases:
        try:
            run_case(*c)
        except Exception as e:
            print(f"KRIG[{c}] EXC: {type(e).__name__}: {e}", flush=True)
