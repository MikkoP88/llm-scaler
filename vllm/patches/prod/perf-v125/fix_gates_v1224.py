#!/usr/bin/env python3
"""fix_gates_v1224.py — repair gates_v1224_lane.sh after the 16:52 ship-gate
abort. Two defects, both introduced by the v124 delta-gate insertion:

1. ORDER: the v124 block ran BEFORE the lsv-gate throwaway container exists
   (it was inserted between the section header and its `docker run`), so every
   `docker exec lsv-gate` failed with "No such container" -> empty `got=`.
2. VAR CLOBBER: the block redefined SP=/opt/venv/lib/python3.12/site-packages,
   dropping the inherited SP=.../vllm -> every later gate grepping
   $SP/v1/... or $SP/_xpu_ops.py hit a nonexistent path.

Fix: relocate the block to AFTER the container-run abort line and rename its
variable to SPR. Inherited SP semantics restored; v124 gates then run against
a live lsv-gate. Idempotent (keyed on SPR presence).
"""
import sys

P = "/root/build/gates_v1224_lane.sh"
src = open(P).read()

if "SPR=/opt/venv/lib/python3.12/site-packages" in src:
    print("ALREADY_FIXED")
    sys.exit(0)

START = 'echo "--- v124 deltas: P16 gate fix + P19.5a fp8 plumbing (inert on fp16 serve) ---"'
END = ('echo "$V124RT" | grep -q float8_e5m2 && echo "GATE-OK   v124_fp8_e5m2_resolves" '
       '|| { echo "GATE-FAIL v124_fp8_e5m2_resolves"; FAIL=1; }')

assert START in src, "v124 block start marker not found"
assert END in src, "v124 block end marker not found"

i = src.index(START)
j = src.index(END) + len(END)
block = src[i:j]
rest = src[:i] + src[j:]

# rename the clobbering var inside the block (SP= line + its 4 uses)
assert "SP=/opt/venv/lib/python3.12/site-packages\n" in block, "SP clobber line not in block"
block = block.replace("SP=/opt/venv/lib/python3.12/site-packages\n",
                      "SPR=/opt/venv/lib/python3.12/site-packages\n")
n_uses = block.count("$SP/vllm/")
block = block.replace("$SP/vllm/", "$SPR/vllm/")
assert n_uses == 4, f"expected 4 $SP/vllm uses in block, found {n_uses}"

# re-insert AFTER the lsv-gate container-run abort line
ANCHOR = 'llm-scaler-exp:v1.2.24 || { echo "ABORT: gate container run failed"; exit 1; }\n'
assert ANCHOR in rest, "container-run anchor not found"
fixed = rest.replace(ANCHOR, ANCHOR + "\n" + block + "\n", 1)

# post-conditions: no stray $SP/vllm outside SPR lines, single SPR def,
# original header directly precedes docker rm again
assert fixed.count("SPR=/opt/venv/lib/python3.12/site-packages") == 1
assert "$SPR/vllm/" in fixed
assert fixed.count("$SP/vllm/") == 0
hdr = 'echo "--- throwaway gate container from the shipped image ---"'
assert rest.index(hdr) < rest.index("docker rm -f lsv-gate"), "header/docker-rm order sane in rest"

open(P, "w").write(fixed)
print("GATES_V1224_FIXED block_moved+renamed SPR_uses=4")
