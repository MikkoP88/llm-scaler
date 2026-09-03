#!/usr/bin/env python3
"""v38 quantification: flip RATE + element localization for the
eagle_ops.page_attn_decode fp8 nondeterminism (caught at 1/50 in v38erig).

Per case: N=1000 calls, fixed inputs, bit-compare vs first.
Record: #flipping iters, per-flip differing-element count + flat indices
(first 8), maxdiff. Cases: fp8@117, fp8@2080, fp8@512 (aligned),
fp16@117 (control), fp8@117 scales=0.01, fp8@118, fp8@116.
Both eager and graph-replay modes.
"""
import torch
import custom_esimd_kernels_vllm.eagle_ops as eagle_ops

torch.cuda.graph = torch.xpu.graph
torch.cuda.CUDAGraph = torch.xpu.XPUGraph
torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle

DEV = "xpu"
HKV, D, BS, GQA, PADG = 2, 256, 512, 6, 8
HQ_PAD = HKV * PADG
N = 1000


def make(kv_len, dtype, seed=3, scale=1.0):
    nb = (kv_len + BS - 1) // BS + 1
    g = torch.Generator().manual_seed(seed)
    kv16 = (torch.randn(2, nb, BS, HKV, D, generator=g) * 0.1).to(torch.float16)
    kv = (kv16.to(torch.float8_e4m3fn) if dtype == "fp8" else kv16).to(DEV)
    q = torch.zeros(1, HQ_PAD, D, dtype=torch.float16, device=DEV)
    q[:, :HKV * GQA] = (torch.randn(1, HKV * GQA, D, generator=g) *
                        0.1).to(torch.float16).to(DEV)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    sk = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    return kv, q, bt, sk, scale


def quantify(kv_len, dtype, graph=False, scale=1.0, seed=3):
    kv, q, bt, sk, s = make(kv_len, dtype, seed, scale)
    out = torch.zeros_like(q)

    def call():
        eagle_ops.page_attn_decode(q, kv, bt, sk, out, 1, kv_len, s, s)

    if graph:
        call(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            call()
        def once():
            g.replay(); torch.xpu.synchronize()
    else:
        def once():
            out.zero_(); call(); torch.xpu.synchronize()

    once()
    ref = out.clone()
    flips = []
    for i in range(1, N):
        once()
        if not torch.equal(out, ref):
            ne = (out != ref)
            n_diff = int(ne.sum())
            md = (out.float() - ref.float()).abs().max().item()
            idx = ne.flatten().nonzero().flatten()[:8].tolist()
            flips.append((i, n_diff, md, idx))
    tag = f"{dtype}@{kv_len} {'GRAPH' if graph else 'EAGER'}" \
          f"{' scale=' + str(scale) if scale != 1.0 else ''}"
    print(f"QUANT[{tag}] flips={len(flips)}/{N - 1}", flush=True)
    for f in flips[:12]:
        print(f"    iter={f[0]} ndiff={f[1]} maxdiff={f[2]:.2e} "
              f"idx={f[3]}", flush=True)
    if len(flips) > 12:
        print(f"    ... +{len(flips) - 12} more", flush=True)
    return len(flips)


if __name__ == "__main__":
    quantify(117, "fp8")
    quantify(117, "fp16")
    quantify(512, "fp8")
    quantify(2080, "fp8")
    quantify(117, "fp8", scale=0.01)
    quantify(118, "fp8")
    quantify(116, "fp8")
    quantify(117, "fp8", graph=True)
    quantify(2080, "fp8", graph=True)
