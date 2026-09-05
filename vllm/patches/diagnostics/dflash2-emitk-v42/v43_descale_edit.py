#!/usr/bin/env python3
"""v43_descale_edit.py — llm-scaler v43a (crash-1 fix): fp32 descale contract.

DFlash2 + --kv-cache-dtype fp8_e4m3 crashed at the drafter's first attention:

    flash_attn_interface.py: AssertionError: k_descale must be view of single
    float32 scalar tensor
    (k_descale shape=[1,4] stride=(0,0) dtype=torch.bfloat16 — repro log
    /root/build/crash1_fp8.log)

Root cause: the draft model's attention layers carry MODEL-DTYPE (bf16) kv
scale tensors (inherited quant-config scale params), and
FlashAttentionBackend._inner_forward expands layer._k/_v/_q_scale directly.
The XPU flash interface only accepts a float32 scalar-view descale, so the
dtype half of the assert fails on every fp8-KV draft attention call. Target
layers carry fp32 scales and were unaffected; turboquant draft pools use the
TQ backend and never reach this path (why the validated lane never crashed).

Fix: cast non-fp32 scales to float32 ONCE per layer (they are static after
load), cached by id(layer) like the v19c ESIMD scale cache; fp32 layers pass
through untouched (identity — no behavior change for any certified lane).

Era-3 conventions: fail-loud anchors, exactly-once asserts, idempotent,
writes /w/patched only (boot phase 3 / bake cp loop installs it).
"""
import re
import subprocess
import sys

SP = "/opt/venv/lib/python3.12/site-packages"
SRC = f"{SP}/vllm/v1/attention/backends/flash_attn.py"
DST = "/w/patched/vllm/v1/attention/backends/flash_attn.py"

A1_OLD = """        self._esimd_kv_scales: dict[int, tuple[float, float]] = {}
"""
A1_NEW = """        self._esimd_kv_scales: dict[int, tuple[float, float]] = {}
        # llm-scaler v43 (crash-1 fix): XPU FA descale contract is a single
        # float32 scalar-view; draft-model layers can carry model-dtype
        # (bf16) kv scales via inherited quant config -> cast once per
        # layer (scales are static after load) and cache. Identity for the
        # fp32 layers every certified lane uses.
        self._descale_f32: dict[int, tuple] = {}
"""

A2_OLD = """            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
"""
A2_NEW = """            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            # llm-scaler v43 (crash-1 fix): expand from a float32 CAST of
            # the scale so model-dtype (bf16) draft scales satisfy the
            # interface's scalar-fp32 contract; cached per layer, identity
            # when the scale is already float32.
            _sc = self._descale_f32.get(id(layer))
            if _sc is None:
                def _f32(_t):
                    if _t is None or _t.dtype == torch.float32:
                        return _t
                    return _t.detach().to(torch.float32)
                _sc = (
                    _f32(getattr(layer, "_q_scale", None)),
                    _f32(layer._k_scale),
                    _f32(layer._v_scale),
                )
                self._descale_f32[id(layer)] = _sc
            q_descale = (
                _sc[0].expand(descale_shape)
                if (self.supports_quant_query_input and _sc[0] is not None)
                else None
            )
            k_descale = _sc[1].expand(descale_shape)
            v_descale = _sc[2].expand(descale_shape)
"""


def die(msg: str) -> None:
    print(f"v43_descale_edit: FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def main() -> None:
    src = open(SRC, encoding="utf-8").read()
    if "llm-scaler v43 (crash-1 fix)" in src:
        print("v43_descale_edit: already applied (marker present)")
    else:
        for name, old in (("A1", A1_OLD), ("A2", A2_OLD)):
            n = src.count(old)
            if n != 1:
                die(f"anchor {name} count {n} != 1 in {SRC}")
        src = src.replace(A1_OLD, A1_NEW, 1)
        src = src.replace(A2_OLD, A2_NEW, 1)
        dst = src
        if dst.count("llm-scaler v43 (crash-1 fix)") != 2:
            die("post-edit marker count != 2")
        import os

        os.makedirs("/w/patched/vllm/v1/attention/backends", exist_ok=True)
        with open(DST, "w", encoding="utf-8", newline="\n") as f:
            f.write(dst)
    # Idempotent path: materialize the patched file even if the tree was
    # already edited in place (bake reruns).
    if not __import__("os").path.exists(DST):
        import os

        os.makedirs("/w/patched/vllm/v1/attention/backends", exist_ok=True)
        with open(DST, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", DST], capture_output=True, text=True
    )
    if r.returncode != 0:
        die(f"py_compile failed: {r.stderr}")
    with open(DST, encoding="utf-8") as f:
        dst = f.read()
    if "llm-scaler v43 (crash-1 fix)" not in dst:
        die("marker missing from patched file")
    print("v43_descale_edit: OK (flash_attn.py -> /w/patched, py_compile pass)")


if __name__ == "__main__":
    main()
