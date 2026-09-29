#!/usr/bin/env python3
"""p22b_xe2_patch.py — v125 P22B addendum: fp8-native ssm_state in the xe2
chunked-PREFILL kernel host (chunk_gated_delta_rule_kernels_xe2.hpp).

WHY: the P22-B python switch (VLLM_XPU_GDN_FP8_NATIVE=1) passes the fp8 SSM
pool straight to the SYCL gdn_attention op for EVERY call. That host was
patched fp8-native for DECODE, but prefill-length sequences route internally
to xe_2/chunk_gated_delta_rule_kernels_xe2.hpp whose own DISPATCH_STATE_DTYPE
still rejected fp8 -> leg e4m3n died at EngineCore-init profile prefill:
  RuntimeError: Worker failed with error 'ssm_state dtype must be
  float32/float16/bfloat16, but got Float8_e4m3fn'

KERNEL SAFETY (verified by read): the xe2 chunk kernel is templated
<typename T, typename StateT>. State touches are dtype-generic:
  - initial-state load  :1057  ssm_state_f32_ptr[e] = static_cast<float>(ssm_state_ptr[e])
  - chunk recursion     :1117  StateT* S_ptr = ssm_state_ptr  (CUTLASS make_tensor)
  - f32 shadow          :1123  S_f32_ptr = ssm_state_f32_ptr
All math stays float; StateT only frames the load/quantize boundaries — the
same property the v125 decode patch (gated_delta_rule.hpp) relied on. The c10
Float8 structs (operator float() + float ctor, header-only bit conversions)
satisfy every site, so the change is two dispatch branches + includes.

Quality note: one quantization per prefill final-state write — identical
semantics to the stage-A bridge (pool quantize once per prefill), so prefill
quality is unchanged vs the certified stage-A e4m3/e5m2 runs.

Macro hygiene (the v125 ninja-failure lesson): no // comments inside the
#define body; every physical macro line ends with ' \\'. Idempotent.
"""
import sys

P = "/root/build/vxk/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp"
src = open(P).read()

if "v125 P22B" in src:
    print("ALREADY_APPLIED")
    sys.exit(0)

# --- edit 1: c10 fp8 includes (outside any macro) ---
INC_ANCHOR = '#include "csrc/utils.h"'
assert INC_ANCHOR in src, "include anchor not found"
src = src.replace(
    INC_ANCHOR,
    INC_ANCHOR + "\n"
    "#include <c10/util/Float8_e4m3fn.h>  // v125 P22B: fp8-native state StateT\n"
    "#include <c10/util/Float8_e5m2.h>    // v125 P22B: fp8-native state StateT",
    1,
)

# --- edit 2: fp8 branches in DISPATCH_STATE_DTYPE ---
OLD = (
    "    } else {                                                            \\\n"
    "      TORCH_CHECK(                                                      \\\n"
    "          false,                                                        \\\n"
    '          "ssm_state dtype must be float32/float16/bfloat16, but got ", \\\n'
    "          ssm_state.scalar_type());                                     \\\n"
    "    }                                                                   \\\n"
)
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
assert src.count(OLD) == 1, "xe2 reject anchor not unique: %d" % src.count(OLD)
src = src.replace(OLD, NEW, 1)

# --- post-conditions ---
assert src.count("at::kFloat8_e4m3fn") >= 1 and src.count("at::kFloat8_e5m2") >= 1
assert "c10::Float8_e4m3fn" in src and "c10::Float8_e5m2" in src
assert "#include <c10/util/Float8_e4m3fn.h>" in src
# macro hygiene: every inserted line inside the macro must end with ' \'
for ln in NEW.splitlines():
    assert ln.endswith(" \\"), "macro line lacks continuation: %r" % ln
# the two original KERNEL_LAUNCHER-using branches remain (fp32/bf16/fp16 = 3 + ours = 2)
assert src.count("KERNEL_LAUNCHER(scalar_t, state_scalar_t)") >= 5

open(P, "w").write(src)
print("P22B_XE2_PATCHED")
