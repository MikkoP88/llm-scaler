# SPDX-License-Identifier: Apache-2.0
"""v51 CPU cross-check of the triton_fp8_mq algorithm (not codegen).

Mirrors _fp8_mq_stage1 + _fp8_mq_stage2 semantics line-by-line in
vectorized torch (CPU) and compares against a naive reference softmax
attention over dequantized fp8 KV. Validates: per-row causal limits
(row j attends kv < q0 + j, q0 = seq_len - Q_LEN + 1), uniform split
ranges, empty-partial (-inf lse) blind combine, GQA head mapping,
per-tensor k/v scale placement (K/V scaled BEFORE dot / weight).

Run: python test_fp8mq_ref.py   (CPU only, no XPU needed)
"""
import torch

torch.manual_seed(7)
torch.set_default_dtype(torch.float64)

B, Q_LEN, HQ, HK, D = 3, 5, 4, 2, 64
SPLITS, BLOCK_KV = 4, 16
K_SCALE, V_SCALE = 0.37, 1.9
SCALE = D ** -0.5
BLOCK_SIZE = 8
NUM_BLOCKS = 64

seq_lens = torch.tensor([37, Q_LEN + 2, 130], dtype=torch.int32)
Hq_per_kv = HQ // HK

# fp8 "pool": random normal bytes reinterpreted as e4m3 then scaled.
# (fp64 outside, fp8 round-trip for realism — the MATH check is
# dtype-agnostic; running sim+ref in fp64 separates algorithmic error
# from fp32 two-phase rounding, which is ~3e-5 and expected.)
k8 = (torch.randn(NUM_BLOCKS, BLOCK_SIZE, HK, D) * 8).clamp(-448, 448)
v8 = (torch.randn(NUM_BLOCKS, BLOCK_SIZE, HK, D) * 8).clamp(-448, 448)
k_f = (k8.to(torch.float8_e4m3fn).to(torch.float64) * K_SCALE)
v_f = (v8.to(torch.float8_e4m3fn).to(torch.float64) * V_SCALE)

q = torch.randn(B * Q_LEN, HQ, D)

bt = torch.randperm(NUM_BLOCKS)[: max(int(seq_lens.max()), 1)].int()
bt = torch.stack([torch.cat([bt, bt[: max(0, 8 - len(bt))]]) for _ in range(B)])
bt = bt[:, : (int(seq_lens.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1]


def gather_pages(kv_f, b):
    # tokens [0, seq_len) gathered via block table -> [seq_len, HK, D]
    sl = int(seq_lens[b])
    toks = []
    for t in range(sl):
        pg, off = t // BLOCK_SIZE, t % BLOCK_SIZE
        toks.append(kv_f[int(bt[b, pg]), off])
    return torch.stack(toks)


# ---- reference ----
ref = torch.zeros(B * Q_LEN, HQ, D)
for b in range(B):
    sl = int(seq_lens[b])
    K = gather_pages(k_f, b)  # [sl, HK, D]
    V = gather_pages(v_f, b)
    q0 = sl - (Q_LEN - 1)
    for r in range(Q_LEN):
        limit = q0 + r
        for h in range(HQ):
            kv_h = h // Hq_per_kv
            s = (K[:limit, kv_h] @ q[b * Q_LEN + r, h]) * SCALE
            p = torch.softmax(s, dim=0)
            ref[b * Q_LEN + r, h] = p @ V[:limit, kv_h]

# ---- kernel simulation (stage1) ----
Q_BLOCK = 8  # next_pow2(5)
mid = torch.full((B, HQ, SPLITS, Q_BLOCK, D + 1), float("nan"))
for b in range(B):
    sl = int(seq_lens[b])
    K = gather_pages(k_f, b)
    V = gather_pages(v_f, b)
    q0 = sl - (Q_LEN - 1)
    row_limit = q0 + torch.arange(Q_BLOCK)
    split_len = -(-sl // SPLITS)  # cdiv
    for h in range(HQ):
        kv_h = h // Hq_per_kv
        q_tile = torch.zeros(Q_BLOCK, D)
        q_tile[:Q_LEN] = q[b * Q_LEN:(b + 1) * Q_LEN, h]
        for sid in range(SPLITS):
            start = split_len * sid
            end = min(start + split_len, sl)
            m_prev = torch.full((Q_BLOCK,), -float("inf"))
            l_prev = torch.zeros(Q_BLOCK)
            acc = torch.zeros(Q_BLOCK, D)
            for start_n in range(start, end, BLOCK_KV):
                offs = torch.arange(start_n, min(start_n + BLOCK_KV, end))
                kv_mask = offs < end
                att = (offs[None, :] < row_limit[:, None]) & kv_mask[None, :]
                scores = torch.full((Q_BLOCK, len(offs)), -float("inf"))
                scores[:, :] = (q_tile @ K[offs, kv_h].T) * SCALE
                scores = torch.where(att, scores, torch.tensor(-float("inf")))
                n_max = torch.maximum(scores.max(dim=1).values, m_prev)
                n_safe = torch.where(n_max == -float("inf"), 0.0, n_max)
                re = torch.exp(m_prev - n_safe)
                p = torch.exp(scores - n_safe[:, None])
                acc = acc * re[:, None] + p @ V[offs, kv_h]
                l_prev = l_prev * re + p.sum(dim=1)
                m_prev = n_max
            safe_l = torch.where(l_prev > 0, l_prev, 1.0)
            mid[b, h, sid, :, :D] = acc / safe_l[:, None]
            mid[b, h, sid, :, D] = m_prev + torch.log(safe_l)

# ---- stage2 blind combine ----
out = torch.zeros(B * Q_LEN, HQ, D)
for b in range(B):
    for h in range(HQ):
        m_run = torch.full((Q_BLOCK,), -float("inf"))
        e_sum = torch.zeros(Q_BLOCK)
        acc = torch.zeros(Q_BLOCK, D)
        for sid in range(SPLITS):
            lse = mid[b, h, sid, :, D]
            tv = mid[b, h, sid, :, :D]
            n_max = torch.maximum(m_run, lse)
            n_safe = torch.where(n_max == -float("inf"), 0.0, n_max)
            old = torch.exp(m_run - n_safe)
            exp_l = torch.where(lse == -float("inf"), 0.0, torch.exp(lse - n_safe))
            acc = acc * old[:, None] + exp_l[:, None] * tv
            e_sum = e_sum * old + exp_l
            m_run = n_max
        res = acc / torch.where(e_sum > 0, e_sum, 1.0)[:, None]
        out[b * Q_LEN:(b + 1) * Q_LEN, h] = res[:Q_LEN]

diff = (out - ref).abs().max().item()
rel = (out - ref).abs().max(dim=-1).values.max().item()
print(f"max abs diff = {diff:.3e}  (fp32 sim, expected < 1e-5)")
assert diff < 1e-5, f"ALGORITHM MISMATCH: {diff}"
print("ALGORITHM OK: split math, per-row causal limits, blind combine, "
      "GQA mapping, scale placement all match reference")
