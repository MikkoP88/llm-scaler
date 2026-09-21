#!/usr/bin/env python3
"""llm-scaler T2 offline repro v3: drive the REAL op with degenerate
metadata shapes to find which input makes it return token 0.

Serve facts (16:06-16:07): prefill (1-row) samples fine; no-draft decode
(1-row) deterministically samples 0 despite masked real logits. Metadata
(k/p/temp slots) is rebuilt per step from input_batch — a stale or
zero-valued slot, or a row-count mismatch vs logits rows, is the
remaining suspect class.
"""
import torch

import vllm._xpu_ops  # noqa: F401
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

dev = torch.device("xpu")
s = TopKTopPSampler()
V = 248320
torch.manual_seed(0)
torch.xpu.manual_seed_all(0)


def raw(desc, logits, kT, pT, seeds=None):
    out = torch.empty(logits.shape[0], dtype=torch.int64, device=dev)
    if seeds is None:
        seeds = torch.tensor([123, 456], dtype=torch.int64)
    torch.ops.vllm.xpu_topk_topp_sampler(
        out, None, logits, kT, pT, "raw_logprobs", seeds)
    vals = [int(x) for x in out.reshape(-1).tolist()][:6]
    print(f"{desc}: out={vals}")
    return vals


def masked(rows=1, forbid=15):
    t = torch.randn(rows, V, device=dev, dtype=torch.float32) * 4.0
    if forbid:
        t[:, :forbid] = float("-inf")
    return t


print("=== k/p degenerate values (1-row logits) ===")
raw("k=0", masked(), torch.tensor([0], dtype=torch.int64, device=dev), None)
raw("k=-1", masked(), torch.tensor([-1], dtype=torch.int64, device=dev), None)
raw("k=-1 p.8", masked(), torch.tensor([-1], dtype=torch.int64, device=dev),
    torch.tensor([0.8], dtype=torch.float32, device=dev))
raw("p=0.0", masked(), None,
    torch.tensor([0.0], dtype=torch.float32, device=dev))
raw("p=1.0", masked(), None,
    torch.tensor([1.0], dtype=torch.float32, device=dev))
raw("k=0 p=0.0", masked(), torch.tensor([0], dtype=torch.int64, device=dev),
    torch.tensor([0.0], dtype=torch.float32, device=dev))

print("=== row-count mismatch: logits 1 row, k/p 5 rows ===")
raw("k5rows_on_1row_logits", masked(1),
    torch.full((5,), 20, dtype=torch.int64, device=dev),
    torch.full((5,), 0.8, dtype=torch.float32, device=dev))

print("=== non-contiguous logits ===")
big = masked(4)
view = big[2:3]  # a row view — non-contiguous for a (1,V) row? check stride
print("view stride:", view.stride(), "contig:", view.is_contiguous())
raw("rowview k20p08", view,
    torch.tensor([20], dtype=torch.int64, device=dev),
    torch.tensor([0.8], dtype=torch.float32, device=dev))

print("=== all -inf logits row ===")
dead = masked(forbid=None)
dead[0, :] = float("-inf")
raw("all_-inf k20", dead,
    torch.tensor([20], dtype=torch.int64, device=dev), None)

print("=== seeds variants ===")
raw("seeds zeros", masked(),
    torch.tensor([20], dtype=torch.int64, device=dev),
    torch.tensor([0.8], dtype=torch.float32, device=dev),
    seeds=torch.tensor([0, 0], dtype=torch.int64))
print("REPRO3_DONE")
