#!/usr/bin/env python3
"""FN8_06n (DEC40): ROOT-CAUSE FIX for the first-spec-step abort.

06m stage map (13:36 boot, /stage/serve_d9.log lines 460-539): draft
phase fully survives (S3/S4 x3 passes, S5 propose_done, all ids/pos/
slots valid) -> 06M step=1 prints the verify (T=5, valid ids) ->
Indexing.h:622 immediately: 20 threads [5..24,0,0] indexing a
5-element dim.

Root cause chain (this patch closes it):
  1. spec_query_start_loc is PER-TOKEN on this build (T+1 entries) —
     06j's capture sentinel proves it: site=spec S=20 L=5 at the T=20
     capture (a per-seq build would print S=4).
  2. _ple_forward_spec computes S = sq.shape[0]-1 = T and
     n_spec_toks = S*L = T*L.  For the size-5 verify graph capture:
     n_spec_toks = 5*5 = 25 while the md token map has 5 entries.
  3. The 06b guard replaces the map with arange(25) ->
     hidden_states[arange(25)] over a T=5 tensor -> ids 5..24 OOB.
  4. XPU graph capture suppresses device asserts -> the bad gather
     BAKES into the size-5 verify graph -> first replay asserts.
     (d9b no-MTP never ran this path -> clean.)

Fix: on the mismatch, recover the REAL seq count from the token map
(or T): S = min(map_len, T) // L; clamp s_slots to its first S rows;
bake arange(S*L) at the correct width.  Shape-only python (capture-
legal, no .item()); mixed spec+flat steps resolve correctly too
(map holds S*L spec entries, non-spec group takes arange(S*L, T)).

Edits (qwen4_exp.py, _ple_forward_spec):
  N1  map-missing fallback: arange(T), not arange(S*L).
  N2  mismatch guard: clamp S/n_spec_toks/s_slots + 06n sentinel at
      capture (prints the clamped S/L/T the graph actually bakes).

Idempotent; py_compile gate; backup .fn8_06n.bak; EOL aware;
anchor==1 per edit; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06n"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06n"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# N1: map-missing fallback sized to T (S*L can exceed T; see N2 clamp).
OLD1_LF = '''            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is None:
                s_indx = torch.arange(n_spec_toks, device=device)
            s_indx = s_indx.to(torch.long)
'''
NEW1_LF = '''            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is None:
                # 06n: T-sized map (S*L can exceed T -> see N2 clamp).
                s_indx = torch.arange(T, device=device)
            s_indx = s_indx.to(torch.long)
'''

# N2: mismatch guard -> width clamp (ASCII-only anchor lines).
OLD2_LF = '''                s_indx = torch.arange(n_spec_toks, device=device)
            D = hidden_states.shape[-1]
'''
NEW2_LF = '''                # 06n (DEC40) ROOT-CAUSE FIX (FN8_06n): sq is
                # PER-TOKEN (T+1 entries) on this build, so S*L = T*L
                # exceeds the real token count and the 06b arange(S*L)
                # baked a T*L-wide gather over a T-row tensor (d9
                # first-spec-step Indexing.h:622, ids 5..24 of a
                # 5-row dim at bs=1 k=4; capture suppresses device
                # asserts so the bad op baked into the size-5 verify
                # graph).  Recover the REAL seq count from the token
                # map (or T), clamp s_slots to its first S rows, and
                # bake the uniform path at the CORRECT width.
                S = min(s_indx.shape[0] if s_indx.numel() > 0 else T,
                        T) // L
                n_spec_toks = S * L
                s_indx = torch.arange(n_spec_toks, device=device)
                s_slots = s_slots[:S]
                _fn8_ple_capt_sentinel(
                    "spec_fix", "spec",
                    extra=f"S={S} L={L} n={n_spec_toks} T={T}")
            D = hidden_states.shape[-1]
'''

EDITS = [
    ("N1_map_fallback_T", OLD1_LF, NEW1_LF),
    ("N2_width_clamp", OLD2_LF, NEW2_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06n.bak"
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

bak = path + ".fn8_06n.bak"
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
