#!/usr/bin/env python3
"""p29dbg_dedup.py — v126 P29B-DIAG: collapse duplicated installer
insertions back to exactly one per file.

The original installer guard ('MARK in txt and tag in txt') never matched
because the tags never occur in the inserted text, so re-runs stacked
copies: 2x dbg include in esimd_kernel_lgrf.sycl, 4x decl in
kernel_ops.h, 4x m.def/m.impl in torch_extension_lgrf.cc. The 4x m.def
aborts at import ("same name and overload name multiple times"). This
script removes every duplicate block keyed on the P29B-DIAG comment
markers, keeping the first occurrence of each. Idempotent.
"""
import re
import sys

TREE = "/root/llm-scaler/vllm/custom-esimd-kernels-vllm"

DECL = re.compile(
    r"\n// v126 P29B-DIAG: instrumented spec variant \(diagnostic only\)\n"
    r"at::Tensor esimd_gdn_conv_fused_seq_spec_dbg\(.*?"
    r"double scale, at::Tensor dbg\);\n",
    re.DOTALL)

OP = re.compile(
    r"\n  // v126 P29B-DIAG: instrumented spec op \(diagnostic only\)\n"
    r"  m\.def\(\"esimd_gdn_conv_fused_seq_spec_dbg.*?"
    r"&esimd_gdn_conv_fused_seq_spec_dbg\);\n",
    re.DOTALL)

INC = '#include "esimd_kernels/gdn_conv_fused_seq_spec_dbg.h"  // v126 P29B-DIAG\n'


def dedup_regex(path, pat, tag):
    with open(path) as f:
        txt = f.read()
    hits = pat.findall(txt)
    if len(hits) <= 1:
        print("%s: %d block(s), nothing to do" % (tag, len(hits)))
        return
    first = pat.search(txt)
    # remove all, then re-insert the first at its original offset
    txt2 = pat.sub("\n", txt)
    txt2 = txt2[:first.start()] + first.group(0) + txt2[first.start():]
    with open(path, "w") as f:
        f.write(txt2)
    print("%s: removed %d duplicate(s), kept 1" % (tag, len(hits) - 1))


def dedup_line(path, line, tag):
    with open(path) as f:
        lines = f.readlines()
    n = sum(1 for l in lines if l == line)
    if n <= 1:
        print("%s: %d occurrence(s), nothing to do" % (tag, n))
        return
    kept = False
    out = []
    for l in lines:
        if l == line and kept:
            continue
        if l == line:
            kept = True
        out.append(l)
    with open(path, "w") as f:
        f.writelines(out)
    print("%s: removed %d duplicate(s), kept 1" % (tag, n - 1))


SYCL = TREE + "/csrc/xpu/esimd_kernel_lgrf.sycl"
KOPS = TREE + "/include/kernel_ops.h"
TCC = TREE + "/csrc/xpu/torch_extension_lgrf.cc"

dedup_line(SYCL, INC, "sycl include")
dedup_regex(KOPS, DECL, "kernel_ops decl")
dedup_regex(TCC, OP, "torch_ext op")

# final proof: exact occurrence counts
for path, probe in (
        (SYCL, 'gdn_conv_fused_seq_spec_dbg.h"'),
        (SYCL, "esimd_gdn_conv_fused_seq_spec_dbg("),
        (KOPS, "at::Tensor esimd_gdn_conv_fused_seq_spec_dbg("),
        (TCC, 'm.def("esimd_gdn_conv_fused_seq_spec_dbg'),
        (TCC, '&esimd_gdn_conv_fused_seq_spec_dbg);')):
    with open(path) as f:
        c = f.read().count(probe)
    print("count %-55s = %d" % (probe, c))
    if c != 1:
        print("UNEXPECTED COUNT in", path)
        sys.exit(1)
print("P29DBG_DEDUP_DONE")
