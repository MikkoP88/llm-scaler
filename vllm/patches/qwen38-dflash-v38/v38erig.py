#!/usr/bin/env python3
"""v38 ESIMD rig: the REAL #18 suspect — eagle_ops.page_attn_decode.

Discovery chain: fp8-e4m3 nospec decode NEVER reaches the vxk FA2 kernel
(proven deterministic + correct by v38krig/v38ref). The PAGED_ATTN_ESIMD
gate (flash_attn.py:1085+) routes fp16-Q + XPU-graph + head256 + GQA>=2
decoder decode to custom_esimd_kernels_vllm.eagle_ops.page_attn_decode.
Engine call shape for Qwen3.8 per-rank (HQ=12, HKV=2, GQA=6 -> pad 8):
  q (1, 16, 256) fp16, kv [2, nb, 512, 2, 256] (fp8 view or fp16),
  block_table (1, nb) i32, seqused_k (1,) i32, out fp16, 1, max_seq_len,
  k_scale, v_scale (python floats).

This rig: fixed inputs, 50 calls, bit-compare — eager AND graph-replay.
Axes: kv dtype (fp8 e4m3 vs fp16), kv_len (117/2080), tail poison,
ghost-pad-row content, max_seq_len scalar.
"""
import torch
import custom_esimd_kernels_vllm.eagle_ops as eagle_ops

# XPU graphs: alias exactly like vllm/v1/worker/xpu_model_runner.py:101-104
torch.cuda.graph = torch.xpu.graph
torch.cuda.CUDAGraph = torch.xpu.XPUGraph
torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle

DEV = "xpu"
HKV, D, BS, GQA, PADG = 2, 256, 512, 6, 8
HQ_PAD = HKV * PADG  # 16


def make_inputs(kv_len, dtype, seed=3):
    nb = (kv_len + BS - 1) // BS + 1
    g = torch.Generator().manual_seed(seed)
    kv16 = (torch.randn(2, nb, BS, HKV, D, generator=g) * 0.1).to(torch.float16)
    if dtype == "fp8":
        kv = kv16.to(torch.float8_e4m3fn).to(DEV)
        ks = vs = 1.0
    else:
        kv = kv16.to(DEV)
        ks = vs = 1.0
    q = torch.zeros(1, HQ_PAD, D, dtype=torch.float16, device=DEV)
    q[:, :HKV * GQA] = (torch.randn(1, HKV * GQA, D, generator=g) *
                        0.1).to(torch.float16).to(DEV)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    sk = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    return kv, q, bt, sk, nb, ks, vs


def run(kv, q, bt, sk, msl, ks, vs, n=50, graph=False):
    out = torch.zeros_like(q)

    def call():
        eagle_ops.page_attn_decode(q, kv, bt, sk, out, 1, msl, ks, vs)

    if not graph:
        outs = []
        for _ in range(n):
            out.zero_()
            call()
            torch.xpu.synchronize()
            outs.append(out.clone())
    else:
        # capture once, replay n times (XPU graphs via torch.cuda API alias)
        call()
        torch.xpu.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        outs = []
        for _ in range(n):
            out.zero_()
            g.replay()
            torch.xpu.synchronize()
            outs.append(out.clone())
    ref = outs[0].clone()
    mism = [i for i, o in enumerate(outs) if not torch.equal(o, ref)]
    md = 0.0
    if mism:
        md = (outs[mism[0]].float() - ref.float()).abs().max().item()
    return len(mism) == 0, len(mism), md


def graph_phase(kv, q, bt, sk, msl, ks, vs, n=50):
    """Capture the call in an XPU graph, then:
       R1: replay n times, bit-compare (replay-to-replay stability)
       R2: poison OUT between replays, compare vs clean (read-before-write)
       R3: poison KV invalid tail between replays (tail dependence)"""
    out = torch.zeros_like(q)

    def call():
        eagle_ops.page_attn_decode(q, kv, bt, sk, out, 1, msl, ks, vs)

    call()
    torch.xpu.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    # R1
    outs = []
    for _ in range(n):
        out.zero_()
        g.replay()
        torch.xpu.synchronize()
        outs.append(out.clone())
    ref = outs[0]
    mism = [i for i, o in enumerate(outs) if not torch.equal(o, ref)]
    r1 = (len(mism) == 0, len(mism))
    # R2: poison out before replay
    g.replay(); torch.xpu.synchronize(); clean = out.clone()
    out.fill_(3.25)
    g.replay(); torch.xpu.synchronize(); poisoned = out.clone()
    r2 = torch.equal(clean, poisoned)
    # R3: poison invalid tail of last partial block between replays
    r3 = None
    kv_len = int(sk[0])
    off = kv_len % BS
    last = kv_len // BS
    if off:
        g.replay(); torch.xpu.synchronize(); base = out.clone()
        with torch.no_grad():
            kv[0, last, off:] = 400.0 if kv.dtype != torch.float16 \
                else torch.tensor(400.0, dtype=torch.float16)
            kv[1, last, off:] = 0
        g.replay(); torch.xpu.synchronize(); pois = out.clone()
        r3 = torch.equal(base, pois)
    return r1, r2, r3


def main():
    for dtype in ("fp8", "fp16"):
        for kv_len in (117, 2080):
            kv, q, bt, sk, nb, ks, vs = make_inputs(kv_len, dtype)
            msl = kv_len
            ok, nm, md = run(kv, q, bt, sk, msl, ks, vs)
            print(f"ERIG[{dtype} kv={kv_len} EAGER] stable={ok} "
                  f"mism={nm}/50 maxdiff={md:.3e}", flush=True)
            # tail poison: rewrite invalid tail of last partial block
            off = kv_len % BS
            last = kv_len // BS
            if off:
                kvp = kv.clone()
                kvp[0, last, off:] = torch.full_like(
                    kvp[0, last, off:], 400.0
                    if dtype == "fp8" else 400.0)
                kvp[1, last, off:] = 0
                _, q2, _, _, _, ks2, vs2 = make_inputs(kv_len, dtype)
                o1 = torch.zeros_like(q)
                eagle_ops.page_attn_decode(q, kv, bt, sk, o1, 1, msl, ks, vs)
                o2 = torch.zeros_like(q2)
                eagle_ops.page_attn_decode(q2, kvp, bt, sk, o2, 1, msl,
                                           ks2, vs2)
                torch.xpu.synchronize()
                same = torch.equal(o1, o2)
                d = (o1.float() - o2.float()).abs().max().item()
                print(f"ERIG[{dtype} kv={kv_len} TAILPOISON] "
                      f"invariant={same} maxdiff={d:.3e}", flush=True)
            # ghost-pad content: change q rows 12..15 (ghost heads)
            q3 = q.clone()
            q3[:, HKV * GQA:] = 7.5
            o1 = torch.zeros_like(q)
            eagle_ops.page_attn_decode(q, kv, bt, sk, o1, 1, msl, ks, vs)
            o3 = torch.zeros_like(q3)
            eagle_ops.page_attn_decode(q3, kv, bt, sk, o3, 1, msl, ks, vs)
            torch.xpu.synchronize()
            real_same = torch.equal(o1[:, :12], o3[:, :12])
            print(f"ERIG[{dtype} kv={kv_len} GHOSTPAD] "
                  f"real_rows_invariant={real_same}", flush=True)
            # graph-replay phase (engine execution mode)
            try:
                r1, r2, r3 = graph_phase(kv, q, bt, sk, msl, ks, vs)
                print(f"ERIG[{dtype} kv={kv_len} GRAPH] replay_stable={r1[0]}"
                      f" mism={r1[1]}/50 out_poison_invariant={r2}"
                      f" tail_poison_invariant={r3}", flush=True)
            except Exception as e:
                print(f"ERIG[{dtype} kv={kv_len} GRAPH] EXC: "
                      f"{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
