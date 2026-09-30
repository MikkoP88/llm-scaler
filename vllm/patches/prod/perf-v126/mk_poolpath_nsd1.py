#!/usr/bin/env python3
"""mk_poolpath_nsd1.py — build patch_v126_poolpath_nsd1.py from the P29C
poolpath patch: restrict the native fp8 spec ride to num_spec_decodes == 1
(fp16-gate semantics) so nsd>1 batches fall to the P28 bridge. LEG-D
bisection leg: native-fp8-in-graph allowed ONLY in the unpadded batch-1
graph; multi-seq/padded graphs run bridge.
"""
SRC = "/root/build/patch_v126_poolpath.py"
DST = "/root/build/patch_v126_poolpath_nsd1.py"

OLD = """                or (
                    attn_metadata.num_spec_decodes != 1
                    and not ssm_pool_fp8
                )
                or (
                    ssm_pool_fp8
                    and attn_metadata.num_spec_decodes > 1
                    and attn_metadata.num_decodes > 0
                )
"""
NEW = """                or attn_metadata.num_spec_decodes != 1
"""

with open(SRC) as f:
    txt = f.read()
n = txt.count(OLD)
if n != 1:
    raise SystemExit("anchor count %d != 1 (E2B_NEW clause)" % n)
txt = txt.replace(OLD, NEW)
txt = txt.replace(
    "# v126 P29C: with an fp8 SSM pool the spec kernel runs ANY number",
    "# v126 P29C (P29D-LEGD): native fp8 spec ride restricted to nsd == 1")
with open(DST, "w") as f:
    f.write(txt)
print("wrote", DST)
print("old clause occurrences left:", txt.count("and not ssm_pool_fp8"))
print("markers:", txt.count("v126 P29C"))
