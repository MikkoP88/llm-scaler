#!/usr/bin/env python3
"""FN8_06o (DEC40): prestage spec-width clamp — the ACTUAL first-spec-
step abort site.

06n (PLE in-graph clamp) did not change the assert: ids 5..24 on a
5-row dim at the T=5 verify, IDENTICAL pre/post.  New evidence chain:
  - No capturing=False PLE sentinel at runtime -> _ple_forward_spec
    never ran eagerly -> the crash is BEFORE the model call.
  - The 06g prestage hook (fn8_ple_prestage) runs EAGER right after
    the 06M print, before dispatch.
  - Its spec branch: S = sq.shape[0]-1; n_spec = S*L; on map mismatch
    s_indx = arange(n_spec); then input_ids[s_indx].view(S, L).
  - At the runtime verify sq is PER-TOKEN (T+1 entries): S=T=5,
    n_spec=25, map has 5 entries -> arange(25) over input_ids[T=5]
    -> Indexing.h:622 ids 5..24 of a 5-row dim.  EXACT MATCH.
  - (The 06j capture sentinel "S=20 L=5" at T=20 proves the capture
    dummy has the same per-token shape -> 06n's clamp was still
    needed for the graph bake; this op is the eager prestage twin.)

Fix: mirror 06n — on mismatch recover S = min(map_len, T) // L, clamp
n_spec, slice s_slots[:S], stage exactly S*L rows.  Plus a one-shot
eager probe print of the true runtime sq/S/L/n_spec/T/map_len (banks
ground truth even if another site still asserts).

Edit (qwen4_exp.py, fn8_ple_prestage spec branch):
  O1  probe + width clamp.

Idempotent; py_compile gate; backup .fn8_06o.bak; EOL aware;
anchor==1; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06o"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06o"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD1_LF = '''        if S > 0:
            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is None or s_indx.shape[0] != n_spec:
                s_indx = torch.arange(n_spec, device=device, dtype=torch.long)
            else:
                s_indx = s_indx.to(torch.long)
            ids_pad = input_ids[s_indx].view(S, L)
'''
NEW1_LF = '''        if S > 0:
            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is not None:
                s_indx = s_indx.to(torch.long)
            if not globals().get("_FN8_06O_PROBED", False):
                globals()["_FN8_06O_PROBED"] = True
                print(
                    "FN8_06o prestage sq_len=%d S=%d L=%d n_spec=%d T=%d "
                    "map=%s" % (
                        sq.shape[0], S, L, n_spec, T,
                        (tuple(s_indx.shape) if s_indx is not None else None)),
                    flush=True)
            if s_indx is None or s_indx.shape[0] != n_spec:
                # 06o (DEC40): spec_query_start_loc is PER-TOKEN here
                # (T+1 entries), so S*L = T*L exceeds the real token
                # count and arange(S*L) indexed a T-row input_ids —
                # the d9 first-spec-step Indexing.h:622 abort (ids
                # 5..24 of a 5-row dim at bs=1 k=4; EAGER prestage,
                # pre-dispatch — why the 06n graph-bake clamp alone
                # did not change it).  Same recovery as 06n: real seq
                # count from the token map (or T), slice s_slots to
                # its first S rows, stage exactly S*L rows.
                S = min(s_indx.shape[0] if s_indx is not None
                        and s_indx.numel() > 0 else T, T) // L
                n_spec = S * L
                s_indx = torch.arange(n_spec, device=device, dtype=torch.long)
                s_slots = s_slots[:S]
            ids_pad = input_ids[s_indx].view(S, L)
'''

EDITS = [
    ("O1_prestage_clamp", OLD1_LF, NEW1_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06o.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if MARKER in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


out = src
for name, old, new in EDITS:
    n = out.count(ed(old))
    if n != 1:
        print(f"{TAG} FAIL: anchor {name} occurs {n}x (want 1)")
        sys.exit(2)
    out = out.replace(ed(old), ed(new))

bak = path + ".fn8_06o.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8", newline=""
) as tf:
    tf.write(out)
    tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print(f"{TAG} FAIL: compile error -> {e}")
    sys.exit(3)
finally:
    os.unlink(tmp)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(out)
print(
    f"{TAG} [{REL}] edited "
    f"(eol={'CRLF' if eol == chr(13) + chr(10) else 'LF'}), md5={md5(path)}"
)
print(f"{TAG} ALL_EDITS_OK")
