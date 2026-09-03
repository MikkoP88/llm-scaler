#!/usr/bin/env python3
"""llm-scaler v38 (#18): exact KV-split pin for the XPU FA2 decode call.

ROOT CAUSE (source-verified in vllm-xpu-kernels @ 3cab97a):
  - C++ flash_api.cpp mha_varlen_fwd: `num_kv_splits =
    num_splits.value_or(get_num_splits(...))` — a caller-provided value is
    used EXACTLY; get_num_splits() (batch_size + max_seqlen_k dependent)
    is the #18 bimodality source and is bypassed entirely.
  - vllm_xpu_kernels flash_attn_interface.flash_attn_varlen_func exposes
    `num_splits_kv` which maps 1:1 onto that C++ exact-override param.
  - vLLM's flash_attn.py passes `num_splits=attn_metadata.max_num_splits`
    (the FA2 path RAISES NotImplementedError if that exceeds 1) and NEVER
    passes num_splits_kv — so the heuristic always ran. v34's
    "cap-not-exact" verdict is corrected: the pin was never forwarded.
  - Combine kernel (cutlass ReduceSplitK, reduce_split_k.h:194) reduces
    splits in a serial ascending loop → deterministic for fixed N.

PATCH: pass `num_splits_kv=N` on the main paged-decode call when
VLLM_XPU_FA2_EXACT_SPLITS=N (N>=1) is set. Default (unset/0) = None =
stock behavior — bit-identical, inert for all other lanes.

Known residual risk (rig must falsify): mixed prefill+decode steps
hardcode num_kv_splits=1 in the C++ branch1 path; if decode sequences
execute there under load, step-type becomes a second divergence site and
a one-line wheel patch (num_splits.value_or(1) at that site) is needed.

Idempotent; FAILS on tree drift.
"""
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"
X = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "# llm-scaler v38: exact KV-split pin"

MODULE_OLD = "@dataclass\nclass FlashAttentionMetadata:"
MODULE_NEW = '''# llm-scaler v38: exact KV-split pin (#18 bimodality fix).
# VLLM_XPU_FA2_EXACT_SPLITS=N (N>=1) forwards N as num_splits_kv on the
# paged-decode call — the only param the C++ treats as an EXACT split
# count (flash_api.cpp num_splits.value_or(get_num_splits(...))). The
# heuristic get_num_splits(batch_size, max_seqlen_k) is the bimodality
# source; fixed N + serial ReduceSplitK combine = batch-invariant.
import os as _os38
try:
    _V38_EXACT_SPLITS = int(_os38.environ.get("VLLM_XPU_FA2_EXACT_SPLITS", "0") or 0)
except ValueError:
    _V38_EXACT_SPLITS = 0


@dataclass
class FlashAttentionMetadata:'''

CALL_OLD = """                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                    # XPU MTP+graph: route paged multi-query to branch1"""
CALL_NEW = """                    num_splits=attn_metadata.max_num_splits,
                    # llm-scaler v38: exact KV-split pin (#18). num_splits
                    # (above) is rejected >1 by FA2 and never reaches the
                    # split heuristic; num_splits_kv is the exact override.
                    num_splits_kv=(
                        _V38_EXACT_SPLITS if _V38_EXACT_SPLITS >= 1 else None
                    ),
                    s_aux=self.sinks,
                    # XPU MTP+graph: route paged multi-query to branch1"""

# vLLM's XPU wrapper (vllm/_xpu_ops.py) re-exports the vxk interface with
# its own fixed signature — num_splits_kv must be added there and forwarded
# to the vllm_xpu_kernels.flash_attn_interface call (which already accepts
# it and maps it 1:1 onto the C++ exact-override param).
X_SIG_OLD = """        num_splits=0,
        return_softmax_lse: bool | None = False,"""
X_SIG_NEW = """        num_splits=0,
        # llm-scaler v38: exact KV-split pin (#18) — forwarded to the vxk
        # interface's num_splits_kv (C++ exact split-count override).
        num_splits_kv: int | None = None,
        return_softmax_lse: bool | None = False,"""
X_CALL_OLD = """            per_seq_causal=per_seq_causal,
            is_mix_batch=is_mix_batch,
        )"""
X_CALL_NEW = """            per_seq_causal=per_seq_causal,
            is_mix_batch=is_mix_batch,
            num_splits_kv=num_splits_kv,
        )"""

src = open(P).read()
if MARKER in src:
    print("V38_EXACT: already patched")
    raise SystemExit(0)
n_mod = src.count(MODULE_OLD)
n_call = src.count(CALL_OLD)
if n_mod != 1 or n_call != 1:
    print("V38_EXACT FAIL: anchors mod=%d call=%d (tree drift)" % (n_mod, n_call))
    raise SystemExit(1)
src = src.replace(MODULE_OLD, MODULE_NEW).replace(CALL_OLD, CALL_NEW)
tmp = P + ".v38tmp"
open(tmp, "w").write(src)
py_compile.compile(tmp, doraise=True)
import os as _os
_os.replace(tmp, P)
py_compile.compile(P, doraise=True)

# --- second file: vllm/_xpu_ops.py wrapper passthrough ---
x = open(X).read()
n_sig = x.count(X_SIG_OLD)
n_xc = x.count(X_CALL_OLD)
if n_sig != 1 or n_xc != 1:
    print("V38_EXACT FAIL(xpu_ops): anchors sig=%d call=%d (tree drift)" % (n_sig, n_xc))
    raise SystemExit(1)
x = x.replace(X_SIG_OLD, X_SIG_NEW).replace(X_CALL_OLD, X_CALL_NEW)
tmp = X + ".v38tmp"
open(tmp, "w").write(x)
py_compile.compile(tmp, doraise=True)
_os.replace(tmp, X)
py_compile.compile(X, doraise=True)
print("V38_EXACT OK: num_splits_kv pin wired (flash_attn.py + _xpu_ops.py)")
