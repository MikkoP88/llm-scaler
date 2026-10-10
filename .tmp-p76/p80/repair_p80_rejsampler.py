#!/usr/bin/env python3
"""llm-scaler P80 repair: fix the mis-anchored d2h rung in
rejection_sampler.py (4-space anchor matched a wrong site; correct site is
the 8-space parse_output body). Removes the bad insert, re-inserts at the
unique 8-space anchor, byte-compiles."""
import hashlib
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/sample/rejection_sampler.py"
BAD = (
    '    from vllm._p80_tracer import p80_ev  # llm-scaler P80\n'
    '    p80_ev("d2h")  # llm-scaler P80\n'
)
GOOD_ANCHOR = "        output_token_ids_np = output_token_ids.cpu().numpy()"
GOOD = (
    '        from vllm._p80_tracer import p80_ev  # llm-scaler P80\n'
    '        p80_ev("d2h")  # llm-scaler P80\n'
)

src = open(P).read()
n_bad = src.count(BAD)
assert n_bad == 1, "bad insert count %d != 1" % n_bad
src = src.replace(BAD, "", 1)
assert "llm-scaler P80" not in src, "marker still present after removal"
n_anchor = src.count(GOOD_ANCHOR)
assert n_anchor == 1, "good anchor count %d != 1" % n_anchor
src = src.replace(GOOD_ANCHOR, GOOD + GOOD_ANCHOR, 1)
open(P, "w").write(src)
py_compile.compile(P, doraise=True, cfile="/tmp/p80c_rej.pyc")
print("REPAIRED md5 %s markers=%d"
      % (hashlib.md5(src.encode()).hexdigest(), src.count("llm-scaler P80")))
