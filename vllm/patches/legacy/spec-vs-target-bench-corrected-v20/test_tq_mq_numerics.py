# llm-scaler v19: standalone numerics validation for the multi-query TQ
# verify kernel vs the production single-query decode kernel.
#
# Runs INSIDE the v19 container on the host (XPU):
#   docker exec <ctr> python3 /telemetry/test_tq_mq_numerics.py
#
# Ground truth #1: triton_turboquant_decode_attention called exactly like
#   the v18 backend continuation/verify path (synthetic per-row seq_lens +
#   expanded block table) — the code being replaced.
# Ground truth #2: _tq_full_dequant_kv + fp32 torch attention with causal
#   masks — independent math on the same compressed cache.
# PASS bar: max|MQ - single| <= 5e-3 and max|MQ - ref| <= 1e-2 on fp16
# outputs of O(1) magnitude (both kernels accumulate in fp32; differences
# are fp16 rounding + reduction order only).

import math
import sys

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    triton_turboquant_decode_attention,
    triton_turboquant_mq_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    triton_turboquant_store,
)

import triton

DEV = torch.device("xpu")
D = 128
HK = 8
HQ = 32  # GQA group 4
BLOCK_SIZE = 16
SCALE = 1.0 / math.sqrt(D)

# (name, mse_bits, vqb, key_fp8, norm_correction)
CONFIGS = [
    ("tq4nc-like  mse4/v4", 4, 4, False, False),
    ("k8v4-like   fp8k/v4", 4, 4, True, False),
    ("mse4/v4/nc+ (normcorr)", 4, 4, False, True),
    ("mse3/v3     3bit", 3, 3, False, False),
]

# (cached_len, q_len) — includes 32-split cdiv boundary crossings
CASES = [
    (1, 5),
    (31, 4),   # row limits 32..35: cdiv(35,32)=2 vs cdiv(32,32)=1
    (33, 2),
    (63, 5),   # row limits 64..68: cdiv crosses 2 -> 3
    (95, 8),
    (129, 5),
    (2000, 5),
    (2000, 3),
]


def build_hadamard(d: int, device) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device=device, dtype=torch.float32)


def run_case(cfg, cached_len, q_len, seed):
    name, mse_bits, vqb, key_fp8, norm_corr = cfg
    torch.manual_seed(seed)
    total = cached_len + q_len
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    mse_bytes = math.ceil(D * mse_bits / 8)
    val_data_bytes = math.ceil(D * vqb / 8)
    if key_fp8:
        kps = D + 2  # D fp8 key bytes + fp16 norm
    else:
        kps = mse_bytes + 2  # packed mse idx + fp16 norm
    slot_size = kps + val_data_bytes + 4  # + fp16 scale + fp16 zero

    cache = torch.zeros(
        num_blocks, BLOCK_SIZE, HK, slot_size, dtype=torch.uint8, device=DEV
    )
    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV)[None, :]
    slot_mapping = torch.arange(total, dtype=torch.int32, device=DEV)

    Pi = build_hadamard(D, DEV)
    PiT = Pi.T.contiguous()
    centroids = get_centroids(D, mse_bits).to(device=DEV, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    k = (torch.randn(total, HK, D, device=DEV, dtype=torch.float32) * 0.5).half()
    v = (torch.randn(total, HK, D, device=DEV, dtype=torch.float32) * 0.5).half()
    triton_turboquant_store(
        k, v, cache, slot_mapping, PiT, midpoints,
        mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
        key_fp8=key_fp8,
    )

    q = (torch.randn(q_len, HQ, D, device=DEV, dtype=torch.float32) * 0.5).half()
    synth_seq_lens = torch.arange(
        cached_len + 1, total + 1, dtype=torch.int32, device=DEV
    )

    # Ground truth #1: production single-query kernel (v18 verify path)
    out_single = triton_turboquant_decode_attention(
        query=q, kv_cache=cache, block_table=bt.expand(q_len, -1),
        seq_lens=synth_seq_lens, Pi=Pi, centroids=centroids, scale=SCALE,
        mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
        key_fp8=key_fp8, norm_correction=norm_corr, PiT=PiT,
    )

    # v19 multi-query kernel
    out_mq = triton_turboquant_mq_decode_attention(
        query=q, kv_cache=cache, block_table=bt[:1],
        q0_seq_lens=synth_seq_lens[:1], Pi=Pi, centroids=centroids,
        scale=SCALE, mse_bits=mse_bits, key_packed_size=kps,
        value_quant_bits=vqb, key_fp8=key_fp8, norm_correction=norm_corr,
        PiT=PiT,
    )

    # Ground truth #2: dequant + fp32 torch attention (independent math)
    k_deq = torch.empty(1, HK, total, D, dtype=torch.float16, device=DEV)
    v_deq = torch.empty(1, HK, total, D, dtype=torch.float16, device=DEV)
    grid_dq = (total, HK)
    _tq_full_dequant_kv[grid_dq](
        cache, bt, centroids, k_deq, v_deq,
        k_deq.stride(0), k_deq.stride(1), k_deq.stride(2),
        v_deq.stride(0), v_deq.stride(1), v_deq.stride(2),
        cache.stride(0), cache.stride(1), cache.stride(2),
        bt.stride(0),
        HEAD_DIM=D, BLOCK_SIZE=BLOCK_SIZE, NUM_KV_HEADS=HK,
        MSE_BYTES=mse_bytes, KPS=kps, VQB=vqb,
        VAL_DATA_BYTES=val_data_bytes, MSE_BITS=mse_bits,
        KEY_FP8=1 if key_fp8 else 0,
        BLOCK_D=triton.next_power_of_2(D),
        NORM_CORRECTION=1 if norm_corr else 0,
    )
    ref = torch.empty(q_len, HQ, D, dtype=torch.float32, device=DEV)
    kf = k_deq[0].float()  # [HK, total, D]
    vf = v_deq[0].float()
    for j in range(q_len):
        lim = cached_len + 1 + j
        qf = q[j].float().view(HK, 4, D)  # HQ=32, HK=8 -> group 4: [h, g, d]
        # Stored MSE keys are in ROTATED space (store quantizes k @ PiT;
        # the kernels score q_rot = q @ PiT against the centroids), so the
        # reference must rotate the query too. FP8 keys are stored raw and
        # the kernels use unrotated queries.
        if not key_fp8:
            qf = qf @ PiT
        sc = torch.einsum("hgd,htd->hgt", qf, kf[:, :lim, :]) * SCALE
        pr = torch.softmax(sc, dim=-1)
        ref[j] = torch.einsum("hgt,htd->hgd", pr, vf[:, :lim, :]).reshape(HQ, D)

    d_single = (out_single.float() - out_mq.float()).abs().max().item()
    d_ref = (ref - out_mq.float()).abs().max().item()
    return d_single, d_ref


def bench(cfg, cached_len, q_len, iters=200):
    """Rough timing: v18-style single-query call vs MQ call."""
    name, mse_bits, vqb, key_fp8, norm_corr = cfg
    total = cached_len + q_len
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    mse_bytes = math.ceil(D * mse_bits / 8)
    val_data_bytes = math.ceil(D * vqb / 8)
    kps = (D + 2) if key_fp8 else (mse_bytes + 2)
    slot_size = kps + val_data_bytes + 4
    cache = torch.zeros(
        num_blocks, BLOCK_SIZE, HK, slot_size, dtype=torch.uint8, device=DEV
    )
    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV)[None, :]
    slot_mapping = torch.arange(total, dtype=torch.int32, device=DEV)
    Pi = build_hadamard(D, DEV)
    PiT = Pi.T.contiguous()
    centroids = get_centroids(D, mse_bits).to(device=DEV, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    k = (torch.randn(total, HK, D, device=DEV) * 0.5).half()
    v = (torch.randn(total, HK, D, device=DEV) * 0.5).half()
    triton_turboquant_store(
        k, v, cache, slot_mapping, PiT, midpoints,
        mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
        key_fp8=key_fp8,
    )
    q = (torch.randn(q_len, HQ, D, device=DEV) * 0.5).half()
    synth = torch.arange(cached_len + 1, total + 1, dtype=torch.int32, device=DEV)

    def run_single():
        triton_turboquant_decode_attention(
            query=q, kv_cache=cache, block_table=bt.expand(q_len, -1),
            seq_lens=synth, Pi=Pi, centroids=centroids, scale=SCALE,
            mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
            key_fp8=key_fp8, norm_correction=norm_corr, PiT=PiT,
        )

    def run_mq():
        triton_turboquant_mq_decode_attention(
            query=q, kv_cache=cache, block_table=bt[:1],
            q0_seq_lens=synth[:1], Pi=Pi, centroids=centroids, scale=SCALE,
            mse_bits=mse_bits, key_packed_size=kps, value_quant_bits=vqb,
            key_fp8=key_fp8, norm_correction=norm_corr, PiT=PiT,
        )

    for f in (run_single, run_mq):
        f()  # JIT warmup
    torch.xpu.synchronize()
    results = {}
    for tag, f in (("single", run_single), ("mq", run_mq)):
        t0 = torch.Event(enable_timing=True)
        t1 = torch.Event(enable_timing=True)
        torch.xpu.synchronize()
        t0.record()
        for _ in range(iters):
            f()
        t1.record()
        torch.xpu.synchronize()
        results[tag] = t0.elapsed_time(t1) / iters
    return results


def main():
    print(f"device={DEV} torch={torch.__version__}")
    fails = 0
    for cfg in CONFIGS:
        for cached_len, q_len in CASES:
            d_single, d_ref = run_case(cfg, cached_len, q_len, seed=42 + cached_len)
            ok = d_single <= 5e-3 and d_ref <= 1e-2
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] {cfg[0]:<24} cached={cached_len:<5} q_len={q_len} "
                f"max|mq-single|={d_single:.2e} max|mq-ref|={d_ref:.2e}"
            )
            if not ok:
                fails += 1

    print("\n--- perf smoke (per-call ms, one attention layer) ---")
    for cfg in CONFIGS[:2]:
        for depth in (2000, 4096, 16384):
            r = bench(cfg, depth, 5)
            sp = r["single"] / r["mq"] if r["mq"] > 0 else float("inf")
            print(
                f"{cfg[0]:<24} cached={depth:<6} q_len=5  "
                f"single={r['single']:.3f}ms mq={r['mq']:.3f}ms speedup={sp:.2f}x"
            )

    if fails:
        print(f"\nRESULT: FAIL ({fails} case(s))")
        sys.exit(1)
    print("\nRESULT: ALL PASS")


if __name__ == "__main__":
    main()
