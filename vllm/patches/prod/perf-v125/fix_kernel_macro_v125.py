#!/usr/bin/env python3
"""fix_kernel_macro_v125.py — repair the v125 P22B dispatch-macro insertion in
/root/build/vxk/csrc/xpu/gdn_attn/gated_delta_rule.hpp.

Defect: DISPATCH_STATE_DTYPE is a #define; every physical line of its body
must end with a backslash. The inserted two-line `//` comment had NO
continuation backslashes, which TERMINATED the #define at the comment —
the fp8 branch bodies landed at file scope ("use of undeclared identifier
'scalar_t'" at the BUCKET_DISPATCH line, gdn_attn_interface.cpp TU).

Fix: convert the comment to a /* */ block (backslash-continued) and assert
every line between `#define DISPATCH_STATE_DTYPE` and `} while (0)` ends
with a backslash. Idempotent (keyed on the corrected comment text).
"""
import sys

P = "/root/build/vxk/csrc/xpu/gdn_attn/gated_delta_rule.hpp"
src = open(P).read()

FIXED_MARK = "/* v125 P22B: fp8-native state (bit-exact dequant/quant at the"
if FIXED_MARK in src:
    print("ALREADY_FIXED")
    sys.exit(0)

OLD = (
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e4m3fn) {        \\\n"
    "      // v125 P22B: fp8-native state (bit-exact dequant/quant at the\n"
    "      // scalar load/store sites; all math stays float)\n"
    "      using state_scalar_t = c10::Float8_e4m3fn;                       \\\n"
    "      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)         \\\n"
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e5m2) {          \\\n"
    "      using state_scalar_t = c10::Float8_e5m2;                         \\\n"
    "      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)         \\\n"
)
NEW = (
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e4m3fn) {        \\\n"
    "      /* v125 P22B: fp8-native state (bit-exact dequant/quant at the  \\\n"
    "         scalar load/store sites; all math stays float) */            \\\n"
    "      using state_scalar_t = c10::Float8_e4m3fn;                       \\\n"
    "      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)         \\\n"
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e5m2) {          \\\n"
    "      using state_scalar_t = c10::Float8_e5m2;                         \\\n"
    "      BUCKET_DISPATCH(scalar_t, state_scalar_t, k_bucket_size)         \\\n"
)

assert OLD in src, "broken fp8 branch block not found verbatim"
src = src.replace(OLD, NEW, 1)

# post-condition: inside the DISPATCH_STATE_DTYPE macro body, every non-empty
# line ends with a backslash (from the #define line to '} while (0)')
i = src.index("#define DISPATCH_STATE_DTYPE")
j = src.index("} while (0)", i) + len("} while (0)")
body = [l for l in src[i:j].split("\n")]
bad = [n for n, l in enumerate(body) if l.strip() and not l.rstrip().endswith("\\")
       and not l.rstrip().endswith("} while (0)")]
assert not bad, f"macro lines missing backslash-continuation: {bad[:5]}"

open(P, "w").write(src)
print("KERNEL_MACRO_V125_FIXED all-macro-lines-continued")
