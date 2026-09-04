#!/usr/bin/env python3
"""dt_b65 — pure 65k-context decode with per-round acceptance delta: does
the spec path even run at 65k, or does the lane silently fall back?"""
import sys
sys.path.insert(0, "/root/build")
from dt_bench3 import long_prompt, stream_once, metrics

for i, ctx in enumerate((2000, 16000)):
    m0 = metrics()
    stream_once(f"ctx{ctx}_r{i}", long_prompt(ctx), 256)
    m1 = metrics()
    d = m1.get("spec_decode_num_draft_tokens", 0) - m0.get(
        "spec_decode_num_draft_tokens", 0)
    a = m1.get("spec_decode_num_accepted_tokens", 0) - m0.get(
        "spec_decode_num_accepted_tokens", 0)
    print(f"ACCEPT65 r{i}: draft_delta={d:.0f} accepted_delta={a:.0f}")

