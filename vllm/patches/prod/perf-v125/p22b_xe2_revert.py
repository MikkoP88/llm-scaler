#!/usr/bin/env python3
"""p22b_xe2_revert.py — revert the BLOCKED xe2 fp8 patch from the source
tree /root/build/vxk (host-side builder tree, NOT the container).

WHY REVERT: the xe2 C++ route (fp8 StateT in the chunk-prefill kernel) is
evaluated-blocked — cute/algorithm/reorder.hpp upcast_subbyte_t eagerly
instantiates cutlass::platform::numeric_limits<c10::Float8_e4m3fn/e5m2>
(undefined specialization) -> 12 build errors; the rebuild aborted at
ABORT_NINJA and NO .so was swapped (lsv-test keeps the working
decode-patched build). The fix shipped instead is python-side
(p22b2_python_patch.py: prefill batches take the fp16 bridge), so the xe2
header edit is dead code that would break every future incremental build
incl. the P24 full wheel — restore the tree to the exact pre-patch text.

Inverse of p22b_xe2_patch.py. Idempotent (no-op when the marker is gone).
"""
import sys

P = "/root/build/vxk/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp"
src = open(P).read()

if "v125 P22B" not in src:
    print("ALREADY_REVERTED")
    sys.exit(0)

# --- inverse of edit 2: fp8 branches -> original reject ---
NEW = (
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e4m3fn) {         \\\n"
    "      using state_scalar_t = c10::Float8_e4m3fn;                        \\\n"
    "      KERNEL_LAUNCHER(scalar_t, state_scalar_t)                         \\\n"
    "    } else if (ssm_state.scalar_type() == at::kFloat8_e5m2) {           \\\n"
    "      using state_scalar_t = c10::Float8_e5m2;                          \\\n"
    "      KERNEL_LAUNCHER(scalar_t, state_scalar_t)                         \\\n"
    "    } else {                                                            \\\n"
    "      TORCH_CHECK(                                                      \\\n"
    "          false,                                                        \\\n"
    '          "ssm_state dtype must be float32/float16/bfloat16/fp8_e4m3fn/fp8_e5m2, but got ", \\\n'
    "          ssm_state.scalar_type());                                     \\\n"
    "    }                                                                   \\\n"
)
OLD = (
    "    } else {                                                            \\\n"
    "      TORCH_CHECK(                                                      \\\n"
    "          false,                                                        \\\n"
    '          "ssm_state dtype must be float32/float16/bfloat16, but got ", \\\n'
    "          ssm_state.scalar_type());                                     \\\n"
    "    }                                                                   \\\n"
)
assert src.count(NEW) == 1, "xe2 fp8 branch block not found: %d" % src.count(NEW)
src = src.replace(NEW, OLD, 1)

# --- inverse of edit 1: drop the two c10 fp8 includes ---
INC = (
    '#include "csrc/utils.h"\n'
    "#include <c10/util/Float8_e4m3fn.h>  // v125 P22B: fp8-native state StateT\n"
    "#include <c10/util/Float8_e5m2.h>    // v125 P22B: fp8-native state StateT"
)
assert src.count(INC) == 1, "include block not found: %d" % src.count(INC)
src = src.replace(INC, '#include "csrc/utils.h"', 1)

# --- post-conditions: tree textually identical to pre-P22B ---
assert "v125 P22B" not in src, "marker still present after revert"
assert "at::kFloat8_e4m3fn" not in src and "at::kFloat8_e5m2" not in src
assert "c10::Float8_e4m3fn" not in src and "c10::Float8_e5m2" not in src
assert "float32/float16/bfloat16, but got" in src
assert src.count("KERNEL_LAUNCHER(scalar_t, state_scalar_t)") == 4  # pre-P22B count (patched tree had 6)

open(P, "w").write(src)
print("P22B_XE2_REVERTED")
