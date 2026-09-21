#!/usr/bin/env python3
"""llm-scaler: stage5 gate fix — mark_gmr_v60 substring matched the new
v60g (WEDGEFIX-E) marks (3 hits vs expected 1). Assert B and E marks
explicitly, forbid stale v60e."""
F = "/root/build/stage5_bake_v1215.sh"
lines = open(F).readlines()
NEW = [
    "ck mark_gmr_v60b  1 \"$(docker exec lsv-test grep -c 'llm-scaler v60 (wedge fix): barrier DEFAULT ON' $SP/v1/worker/gpu_model_runner.py)\"\n",
    "ck mark_gmr_v60g  1 \"$(docker exec lsv-test grep -c 'llm-scaler v60g (WEDGEFIX-E)' $SP/v1/worker/gpu_model_runner.py)\"\n",
    "ck mark_gmr_v60e  0 \"$(docker exec lsv-test grep -c 'llm-scaler v60e' $SP/v1/worker/gpu_model_runner.py)\"\n",
]
out, n = [], 0
for ln in lines:
    if ln.startswith("ck mark_gmr_v60 "):
        out.extend(NEW)
        n += 1
    else:
        out.append(ln)
assert n == 1, f"anchor count {n}"
open(F, "w").writelines(out)
print("S5_GATE_FIX_OK")
