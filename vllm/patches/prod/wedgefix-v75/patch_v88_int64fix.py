#!/usr/bin/env python3
"""v88 — int64 pool-offset fix for GDN conv-state kernels (wedge root cause).

chunk_causal_conv1d_xe2.hpp:186/547 and causal_conv1d.hpp:232/452/736/833
compute the per-request state pointer as
    conv_states + states_id * conv_states_stride_0
with `int states_id` and `const int conv_states_stride_0`. The deployed pool
has stride(0) = 524,288 elements (padded 1-MiB unified page), so the product
overflows signed int32 for every state id >= 4097 (4096 wraps to INT32_MIN):
wrapped-negative offsets -> silent state corruption when mapped,
UR_RESULT_ERROR_DEVICE_LOST when unmapped (the lsv-test wedge).

Fix: widen the id to int64 before the multiply, matching the existing safe
pattern in chunk_gated_delta_rule_kernels_xe2.hpp:1046 and
gated_delta_rule.hpp:111. Stride itself stays int (524,288 fits).

Idempotent; writes .v88bak backups next to each patched file.
"""
import re
import shutil
import sys

FILES = [
    "/root/build/vxk/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp",
    "/root/build/vxk/csrc/xpu/gdn_attn/causal_conv1d.hpp",
]

# (id variable, expected bare-multiply source line pattern)
PATTERNS = [
    ("states_id", r"conv_states \+ states_id \* conv_states_stride_0"),
    ("init_state_id", r"conv_states \+ init_state_id \* conv_states_stride_0"),
    ("save_state_id", r"conv_states \+ save_state_id \* conv_states_stride_0"),
]

MARK = "[V88FIX]"


def main() -> int:
    total = 0
    for path in FILES:
        with open(path) as f:
            src = f.read()
        if MARK in src:
            print(f"{path}: ALREADY_PATCHED")
            continue
        n = 0
        out = src
        for var, pat in PATTERNS:
            bare = re.compile(r"(?<!static_cast<int64_t>\()" + re.escape(var) + r" \* conv_states_stride_0")
            new_sub = f"static_cast<int64_t>({var}) * conv_states_stride_0"
            out, k = bare.subn(new_sub, out)
            n += k
        if n == 0:
            print(f"{path}: NO_SITES_FOUND (unexpected)")
            return 1
        # annotate the decl line of the stride var once per file for grep-able marks
        out = out.replace(
            "const int conv_states_stride_0 = conv_states.stride(0);",
            "const int conv_states_stride_0 = conv_states.stride(0);  "
            f"// {MARK} int64 pool offsets (id*stride overflows int32 at id>=4097)",
            1,
        )
        shutil.copy2(path, path + ".v88bak")
        with open(path, "w") as f:
            f.write(out)
        print(f"{path}: patched {n} multiply sites + 1 mark")
        total += n
    print(f"TOTAL_SITES_PATCHED={total}")
    # verify: no bare int multiplies remain
    bad = 0
    for path in FILES:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if re.search(r"(?<!int64_t>\()\w+_id \* conv_states_stride_0", line) and "static_cast" not in line:
                    print(f"REMAINING BARE SITE {path}:{i}: {line.strip()}")
                    bad += 1
    print("VERIFY_BARE_REMAINING=0" if bad == 0 else f"VERIFY_BARE_REMAINING={bad}")
    return 0 if bad == 0 and total >= 6 else 1


if __name__ == "__main__":
    sys.exit(main())
