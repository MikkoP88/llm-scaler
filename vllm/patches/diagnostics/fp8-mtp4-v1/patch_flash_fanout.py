#!/usr/bin/env python3
"""fp8-mtp4-v4 patcher: position-sliced flash fan-out for fp8 paged verify.

Measurement chain (fp8-mtp4-v1/REPORT.md §4.1-4.3): the mtp4 verify step
slope on fp8 KV is 1.98 µs/KVtok through the v51 Triton vector kernel,
3.35 through the stock flash q5 varlen call (branch1/chunk_prefill), and
0.075 for the flash q=1 decode path (nospec lane). The only
measured-healthy fp8 paged shape on this stack is q=1.

This patch adds a route (knob VLLM_XPU_FP8_FANOUT, default configurable)
that decomposes the UNIFORM q_len fp8 paged verify into q_len flash q=1
calls — one per position, sharing the paged KV pool, block table and
per-tensor descales. Per-position causal limits come from
seqused_k - (q_len - 1 - pos), device-side arithmetic (graph-capture
safe). Output rows are written back position-sliced, preserving the
exact row layout contract. All per-call arguments mirror the stock
fallback call in _inner_forward (same objects, same flags).

Patcher protocol: anchors must match exactly once; py_compile gates the
write; --fanout-default sets the knob default.
"""

import argparse
import pathlib
import py_compile
import sys

TARGET = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/"
    "flash_attn.py"
)

KNOB_ANCHOR = '_V51_FP8_MQ = os.environ.get("VLLM_XPU_FP8_MQ", "1") != "0"\n'
KNOB_NEW = (
    '_V51_FP8_MQ = os.environ.get("VLLM_XPU_FP8_MQ", "1") != "0"\n'
    "\n"
    "# llm-scaler fp8-mtp4-v4: position-sliced flash fan-out for UNIFORM\n"
    "# multi-row fp8 paged verify (q_len flash q=1 calls on the measured-\n"
    "# healthy decode shape). Default {FAD}; A/B via VLLM_XPU_FP8_FANOUT.\n"
    '_V4_FANOUT = os.environ.get("VLLM_XPU_FP8_FANOUT", "{FAD}") == "1"\n'
)

GATE_ANCHOR = (
    "                # llm-scaler v51: UNIFORM multi-row fp8 verify -> dedicated\n"
)
GATE_NEW = '''                # llm-scaler fp8-mtp4-v4: position-sliced flash fan-out
                # for UNIFORM multi-row fp8 paged verify. The stock q5
                # varlen call runs the branch1/chunk_prefill kernel
                # (3.35 µs/KVtok step slope) and the v51 Triton kernel
                # 1.98; the flash q=1 decode path is the only
                # measured-healthy fp8 paged shape (0.075, nospec lane).
                # Decompose verify into q_len flash q=1 calls, one per
                # position; per-position causal limit via
                # seqused_k - (q_len - 1 - pos) (device-side, capture
                # safe). Rollback: VLLM_XPU_FP8_FANOUT=0.
                if (
                    _V4_FANOUT
                    and isinstance(self.kv_cache_dtype, str)
                    and self.kv_cache_dtype.startswith("fp8")
                    and block_table is not None
                    and seqused_k is not None
                    and 1 < max_seqlen_q <= 8
                    and num_actual_tokens
                    == max_seqlen_q * (cu_seqlens_q.shape[0] - 1)
                    and attn_metadata.causal is True
                    and (sliding_window_size is None
                         or sliding_window_size[0] < 0)
                    and not self.logits_soft_cap
                    and q_descale is None
                    and self.head_size <= 256
                ):
                    _b = cu_seqlens_q.shape[0] - 1
                    _cu1 = getattr(self, "_v4_fanout_cu", None)
                    if _cu1 is None or _cu1.shape[0] < _b + 1:
                        _cu1 = torch.arange(
                            _b + 1,
                            dtype=cu_seqlens_q.dtype,
                            device=query.device,
                        )
                        self._v4_fanout_cu = _cu1
                    _cu1 = _cu1[: _b + 1]
                    for _pos in range(max_seqlen_q):
                        _q_sl = query[_pos::max_seqlen_q][:_b].contiguous()
                        _o_sl = torch.empty_like(_q_sl)
                        flash_attn_varlen_func(
                            q=_q_sl,
                            k=key_cache,
                            v=value_cache,
                            out=_o_sl,
                            cu_seqlens_q=_cu1,
                            max_seqlen_q=1,
                            seqused_k=seqused_k
                            - (max_seqlen_q - 1 - _pos),
                            max_seqlen_k=max_seqlen_k,
                            softmax_scale=self.scale,
                            causal=True,
                            alibi_slopes=self.alibi_slopes,
                            window_size=sliding_window_size,
                            block_table=block_table,
                            softcap=self.logits_soft_cap,
                            scheduler_metadata=scheduler_metadata,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=None,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=attn_metadata.max_num_splits,
                            s_aux=self.sinks,
                            is_mix_batch=False,
                        )
                        output[_pos::max_seqlen_q][:_b].copy_(_o_sl)
                    return output

'''

DOCKER_NOTE = "patched file: vllm/v1/attention/backends/flash_attn.py"


def patch(path: str, fanout_default: int) -> None:
    p = pathlib.Path(path)
    src = p.read_text()
    fad = "1" if fanout_default else "0"

    if "_V4_FANOUT" in src:
        print("PATCH-SKIP: fan-out route already present")
        return

    def sub_once(anchor: str, new: str, what: str) -> str:
        n = src.count(anchor)
        if n != 1:
            sys.exit(f"PATCH-ANCHOR-FAIL {what}: found {n} occurrences")
        return src.replace(anchor, new)

    src = sub_once(KNOB_ANCHOR, KNOB_NEW.format(FAD=fad), "knob")
    src = sub_once(GATE_ANCHOR, GATE_NEW + GATE_ANCHOR, "gate-insert")
    p.write_text(src)
    py_compile.compile(str(p), doraise=True)
    print(f"PATCHED flash_attn.py: +VLLM_XPU_FP8_FANOUT route "
          f"(default {fad}) — {DOCKER_NOTE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--fanout-default", type=int, choices=[0, 1], default=0)
    a = ap.parse_args()
    patch(a.target, a.fanout_default)
