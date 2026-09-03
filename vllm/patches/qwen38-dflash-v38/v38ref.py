#!/usr/bin/env python3
"""v38 tail-invariance rig: does the e4m3 paged-decode output depend on the
CONTENT of invalid (beyond-seqlen) KV slots in the last partial block?

Engine reality: the KV pool holds stale data from previous requests in the
unwritten tail of a sequence's last block. If the kernel's remainder masking
is correct, output must be BIT-IDENTICAL no matter what garbage sits there.
If masking has a hole, output varies with pool history => the observed #18
run-to-run bimodality (kernel deterministic for FIXED memory contents, as
the v38krig isolation proved, yet engine-unstable).

Test: same valid KV prefix; tail poisoned with (a) different randn seed,
(b) extreme values (+/-448*0.9, the e4m3 near-max), (c) zeros.
Compare kernel outputs bit-exact. Also fp32 reference check for sanity.
"""
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

DEV = "xpu"
HQ, HKV, D, BS = 12, 2, 256, 512
SCALE = D ** -0.5


def make_cache(nb, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(nb, BS, HKV, D, generator=g) * 0.1).to(
        torch.float16).to(torch.float8_e4m3fn).to(DEV)


def poison_tail(kc, vc, kv_len, mode, seed=999):
    """Overwrite ONLY slots >= kv_len of the last partial block."""
    kc, vc = kc.clone(), vc.clone()
    last = kv_len // BS
    off = kv_len % BS
    if off == 0:
        return kc, vc
    if mode == "randn":
        g = torch.Generator().manual_seed(seed)
        kc[last, off:] = (torch.randn_like(kc[last, off:].float()) *
                          0.5).to(torch.float8_e4m3fn)
        vc[last, off:] = (torch.randn_like(vc[last, off:].float()) *
                          0.5).to(torch.float8_e4m3fn)
    elif mode == "extreme":
        kc[last, off:] = torch.full_like(kc[last, off:], 400.0)
        vc[last, off:] = torch.full_like(vc[last, off:], -400.0)
    elif mode == "zeros":
        kc[last, off:] = 0
        vc[last, off:] = 0
    return kc, vc


def call(q, kc, vc, kv_len, splits=None):
    nb = kc.shape[0]
    cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    sk = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    kd = torch.ones((), dtype=torch.float32, device=DEV)
    vd = torch.ones((), dtype=torch.float32, device=DEV)
    o = flash_attn_varlen_func(
        q, kc, vc, 1, cu, kv_len, seqused_k=sk, softmax_scale=SCALE,
        causal=False, block_table=bt, window_size=(-1, -1),
        k_descale=kd, v_descale=vd, num_splits_kv=splits,
        is_mix_batch=False)
    torch.xpu.synchronize()
    return o.clone()


def ref_attn(q, kc, vc, kv_len):
    """fp32 reference over the VALID prefix only."""
    nb = (kv_len + BS - 1) // BS
    K = kc[:nb].reshape(nb * BS, HKV, D).float()[:kv_len]
    V = vc[:nb].reshape(nb * BS, HKV, D).float()[:kv_len]
    # heads: q (1,HQ,D); GQA expand kv heads
    gq = HQ // HKV
    Kx = K.repeat_interleave(gq, dim=1)  # (L,HQ,D)
    Vx = V.repeat_interleave(gq, dim=1)
    s = torch.einsum("qhd,lhd->qhl", q.float(), Kx) * SCALE
    p = torch.softmax(s, dim=-1)
    return torch.einsum("qhl,lhd->qhd", p, Vx)


for kv_len in (117, 512, 2080, 4160):
    nb = (kv_len + BS - 1) // BS + 1
    g = torch.Generator().manual_seed(7)
    q = (torch.randn(1, HQ, D, generator=g) * 0.1).to(torch.float16).to(DEV)
    kc = make_cache(nb, 11)
    vc = make_cache(nb, 12)
    base = call(q, kc, vc, kv_len)
    results = {}
    for mode in ("randn", "extreme", "zeros"):
        kp, vp = poison_tail(kc, vc, kv_len, mode)
        out = call(q, kp, vp, kv_len)
        results[mode] = torch.equal(out, base)
        if not results[mode]:
            d = (out.float() - base.float()).abs().max().item()
            results[mode] = f"DIFF max={d:.3e}"
    # splits-pinned variant of the same test
    base32 = call(q, kc, vc, kv_len, splits=32)
    kp, vp = poison_tail(kc, vc, kv_len, "extreme")
    out32 = call(q, kp, vp, kv_len, splits=32)
    tail32 = torch.equal(out32, base32)
    # fp32 reference sanity (quantization-level closeness expected)
    r = ref_attn(q.cpu(), kc.cpu(), vc.cpu(), kv_len).to(DEV)
    ref_diff = (base.float() - r).abs().max().item()
    print(f"TAILINV[kv={kv_len}] randn={results['randn']} "
          f"extreme={results['extreme']} zeros={results['zeros']} "
          f"pinned32_extreme={'SAME' if tail32 else 'DIFF'} "
          f"ref_maxabs={ref_diff:.3e}", flush=True)
