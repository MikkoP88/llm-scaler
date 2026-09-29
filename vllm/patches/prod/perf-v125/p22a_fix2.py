#!/usr/bin/env python3
"""p22a_fix2.py — fix the three ESIMD rev1 compile-error classes on the
ALREADY-PATCHED host tree (folds into p22a_kernel_patch.py for fresh
applies; this delta script exists because the tree carries rev1).

  1. utils.h has no include guard -> every kernel header re-includes it
     -> redefinition of the P22A helpers. Fix: `#pragma once` at top.
  2. bit_cast_view is &-qualified (lvalue-only): the denorm magic was
     built on a temporary simd. Fix: named `magic_bits` lvalue.
  3. gdn_spec_update_seq row pointers (state_base :82-83, save_base
     :109-110) were not templated in rev1 -> fp16* = StateT* errors.

Idempotent (marker: pragma + magic_bits + StateT* sr checks).
"""
import sys

BASE = "/root/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels"
changed = False

# --- 1. pragma once on utils.h ---
PU = BASE + "/utils.h"
u = open(PU).read()
if not u.startswith("#pragma once"):
    assert u.startswith("#include <sycl/sycl.hpp>"), "utils.h head drift"
    u = "#pragma once\n" + u
    open(PU, "w").write(u)
    changed = True
    print("FIX2 utils.h: pragma once added")
else:
    print("FIX2 utils.h: pragma already present")

# --- 2. magic rvalue -> named lvalue ---
OLD_MAGIC = (
    "    simd<float, 64> magic =\n"
    "        (simd<uint32_t, 64>(141u << 23)).template bit_cast_view<float>().read();\n"
)
NEW_MAGIC = (
    "    simd<uint32_t, 64> magic_bits(141u << 23);\n"
    "    simd<float, 64> magic = magic_bits.template bit_cast_view<float>().read();\n"
)
if OLD_MAGIC in u:
    assert u.count(OLD_MAGIC) == 1
    u = u.replace(OLD_MAGIC, NEW_MAGIC, 1)
    open(PU, "w").write(u)
    changed = True
    print("FIX2 utils.h: magic rvalue -> lvalue")
else:
    assert "magic_bits" in u, "utils.h: magic block not found in either form"
    print("FIX2 utils.h: magic already lvalue")

# --- 3. spec.h row pointers ---
PS = BASE + "/gdn_conv_fused_seq_spec.h"
s = open(PS).read()
for old, new in (
    ("fp16* sr0 = state_base +", "StateT* sr0 = state_base +"),
    ("fp16* sr1 = state_base +", "StateT* sr1 = state_base +"),
    ("fp16* sr0 = save_base +", "StateT* sr0 = save_base +"),
    ("fp16* sr1 = save_base +", "StateT* sr1 = save_base +"),
):
    n = s.count(old)
    if n == 0:
        assert new in s, PS + ": neither form of " + old
        print("FIX2 spec.h: already patched:", new)
        continue
    assert n == 1, PS + ": count %d for %r" % (n, old)
    s = s.replace(old, new, 1)
    changed = True
    print("FIX2 spec.h:", new)
open(PS, "w").write(s)

# --- post-conditions ---
u2 = open(PU).read()
assert u2.startswith("#pragma once")
assert "magic_bits" in u2
assert "(simd<uint32_t, 64>(141u << 23)).template" not in u2
s2 = open(PS).read()
assert "fp16* sr" not in s2, "spec.h still has fp16* sr pointers"
assert s2.count("StateT* sr0") == 2 and s2.count("StateT* sr1") == 2

print("P22A_FIX2_DONE changed=%s" % changed)
