#!/usr/bin/env python3
"""p22b_kernel_patch.py — v125 P22B: fp8-native StateT in the SYCL
gdn_attention kernel dispatch (host tree /root/build/vxk, the source of
truth the builder mounts).

Two edits to csrc/xpu/gdn_attn/gated_delta_rule.hpp, markers 'v125 P22B':
  1. explicit c10 fp8 includes (torch/all.h does not guarantee them)
  2. DISPATCH_STATE_DTYPE: two new branches — at::kFloat8_e4m3fn ->
     c10::Float8_e4m3fn, at::kFloat8_e5m2 -> c10::Float8_e5m2 — plus an
     updated reject message.

Every state touch site is a scalar static_cast (decode load :120, decode
store :247, chunk-init :409, chunk-final :539) and the launcher passes
reinterpret_cast<state_scalar_t*>(data_ptr()) — the c10 fp8 structs
(operator float() + float ctor, header-only bit conversions) satisfy all
of them with zero math changes. Idempotent: keys on the 'v125 P22B'
marker. Run on HOST python3 (file lives on the host fs, not in a
container).
"""
import sys

P = "/root/build/vxk/csrc/xpu/gdn_attn/gated_delta_rule.hpp"
src = open(P).read()

if "v125 P22B" in src:
    print("ALREADY_APPLIED")
    sys.exit(0)

# --- edit 1: includes ---
inc_anchor = "#include <torch/all.h>\n"
assert inc_anchor in src, "include anchor missing"
src = src.replace(
    inc_anchor,
    inc_anchor
    + "#include <c10/util/Float8_e4m3fn.h>  // v125 P22B: fp8 state StateT\n"
    + "#include <c10/util/Float8_e5m2.h>    // v125 P22B: fp8 state StateT\n",
    1,
)

# --- edit 2: dispatch branches ---
# Anchor on the unique reject message, then back up to the enclosing
# `    } else {` (4-space indent; the KERNEL_LAUNCHER else is 2-space, and
# rfind-from-message guarantees we hit the dispatch one).
MSG = '"ssm_state dtype must be float32/float16/bfloat16, but got ",'
assert src.count(MSG) == 1, f"reject message not unique: {src.count(MSG)}"
i = src.index(MSG)
j = src.rfind("    } else {", 0, i)
assert j != -1, "dispatch else not found before reject message"

# NOTE (v125 build fix): every physical line inside the DISPATCH_STATE_DTYPE
# #define body MUST end with a backslash. The first version of this patch used
# a two-line `//` comment WITHOUT continuations, which terminated the #define
# mid-body and dumped the branch code at file scope ("use of undeclared
# identifier 'scalar_t'" in gdn_attn_interface.cpp). Comments inside the macro
# use a backslash-continued /* */ block instead.
branch = (
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e4m3fn) {        \\\n"
    "      /* v125 P22B: fp8-native state (bit-exact dequant/quant at the  \\\n"
    "         scalar load/store sites; all math stays float) */            \\\n"
    "      using state_scalar_t = c10::Float8_e4m3fn;                       \\\n"
    "      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)         \\\n"
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e5m2) {          \\\n"
    "      using state_scalar_t = c10::Float8_e5m2;                         \\\n"
    "      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)         \\\n"
)
src = src[:j] + branch + src[j:]

# updated reject message (informational — the branches above make fp8 valid)
src = src.replace(
    "ssm_state dtype must be float32/float16/bfloat16, but got ",
    "ssm_state dtype must be float32/float16/bfloat16/fp8_e4m3fn/fp8_e5m2, "
    "but got ",
    1,
)

# --- post-conditions ---
assert src.count("v125 P22B") >= 3, "markers missing after edit"
assert "at::kFloat8_e4m3fn" in src and "at::kFloat8_e5m2" in src
assert "c10::Float8_e4m3fn" in src and "c10::Float8_e5m2" in src
assert "#include <c10/util/Float8_e4m3fn.h>" in src
# macro hygiene: every inserted line inside the macro must end with ' \'
for ln in branch.splitlines():
    if "v125 P22B" not in ln and "// scalar" not in ln and "// the" not in ln:
        assert ln.endswith(" \\"), f"macro line lacks continuation: {ln!r}"

open(P, "w").write(src)
print("P22B_KERNEL_PATCHED")
print(f"markers={src.count('v125 P22B')}")
