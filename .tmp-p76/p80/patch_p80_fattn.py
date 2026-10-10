#!/usr/bin/env python3
"""llm-scaler P80: fattn rung in attention.py. The bare anchor
`layer_name = _resolve_layer_name(layer_name)` appears 3x; disambiguate
with the preceding `del kv_cache_dummy_dep` line unique to
unified_attention_with_output."""
import hashlib
import py_compile

P = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
     "attention/attention.py")
ANCHOR = (
    "    del kv_cache_dummy_dep\n"
    "    layer_name = _resolve_layer_name(layer_name)"
)
RUNG = (
    "    del kv_cache_dummy_dep\n"
    "    layer_name = _resolve_layer_name(layer_name)\n"
    "    from vllm._p80_tracer import p80_ev  # llm-scaler P80\n"
    '    p80_ev("fattn")  # llm-scaler P80'
)

src = open(P).read()
if "llm-scaler P80" in src:
    print("SKIP(already)")
else:
    n = src.count(ANCHOR)
    assert n == 1, "anchor count %d != 1" % n
    src = src.replace(ANCHOR, RUNG, 1)
    open(P, "w").write(src)
    py_compile.compile(P, doraise=True, cfile="/tmp/p80c_fattn.pyc")
    print("PATCHED md5 %s" % hashlib.md5(src.encode()).hexdigest())
