#!/usr/bin/env python3
"""FN8_06i (DEC40): make _ple_forward_flat capture-legal (no D2H syncs).

Root cause (d9b try-3 boot fatal, 2026-10-12 12:22:22, banked
/stage/bank/d9b_dead.log):

    qwen4_exp.py, line 1052, in _ple_forward_flat
        max_len = int(lens.max().item())
    RuntimeError: wait method cannot be used for an event associated
    with a command graph.

06h removed the capture-illegal PLE op from the flat path; the NEXT
landmine is the flat path's own host syncs (the spec branch got this
treatment in 06b/06e; the flat path never did — it only runs under
FULL_DECODE_ONLY capture on NO-MTP boots, which never existed before
the d9b isolation leg).

Sync inventory of the flat path (exhaustive grep): exactly three
.item() sites — max_len (1052), the torch.all(...).item() fresh-seed
gate (1072), fresh.any().item() (1075).  _ple_telemetry is already
capture-gated (patch 05, _FN8_XPU_CAPTURING guard).

Edits:
  I1  max_len: static bound 1 while capturing (FULL_DECODE_ONLY
      captures decode-only steps — lens all == 1, so the static bound
      is EXACT, not an upper bound).  Eager steps (prefill/mixed)
      keep the exact sync — legal outside capture.
  I2  fresh-seed: branchless masked write (06e law) — identical
      result set (eos where ~has_state); all-having-state = all-False
      mask = numerical no-op; no .item() guards.

Idempotent; py_compile gate; backup .fn8_06i.bak; EOL aware;
anchor==1 per edit; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06i"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06i"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# I1: max_len .item() -> static bound while capturing.
OLD1_LF = '''        lens = (qsl[1:] - qsl[:-1])
        max_len = int(lens.max().item())
'''
NEW1_LF = '''        lens = (qsl[1:] - qsl[:-1])
        # FN8_06i: `.item()` is a D2H event wait — capture-illegal under
        # FULL_DECODE_ONLY (d9b try-3 fatal at exactly this line, banked
        # /stage/bank/d9b_dead.log).  Captured steps are decode-only
        # (lens all == 1), so the static bound is EXACT.  Eager steps
        # (prefill/mixed) keep the exact sync — legal outside capture.
        if _FN8_XPU_CAPTURING is not None and _FN8_XPU_CAPTURING():
            max_len = 1
        else:
            max_len = int(lens.max().item())
'''

# I2: fresh-seed .item() guards -> branchless masked write.
OLD2_LF = '''        # --- read states (fresh requests: history inits to EOS, not zeros) ---
        prev_ctx = hist_pool[block_ids, :ple.ple_embedding.context_len, 0].clone()
        if has_initial_state is not None and not bool(
            torch.all(has_initial_state[:num_reqs]).item()
        ):
            fresh = ~has_initial_state[:num_reqs].to(torch.bool)
            if bool(fresh.any().item()):
                prev_ctx[fresh] = eos
'''
NEW2_LF = '''        # --- read states (fresh requests: history inits to EOS, not zeros) ---
        prev_ctx = hist_pool[block_ids, :ple.ple_embedding.context_len, 0].clone()
        if has_initial_state is not None:
            # FN8_06i: branchless fresh-seed (06e law) — the two .item()
            # guards were D2H event waits, capture-illegal under
            # FULL_DECODE_ONLY.  Same result set (eos where ~has_state);
            # all-having-state = all-False mask = numerical no-op.
            prev_ctx[~has_initial_state[:num_reqs].to(torch.bool)] = eos
'''

EDITS = [
    ("I1_maxlen_static", OLD1_LF, NEW1_LF),
    ("I2_fresh_branchless", OLD2_LF, NEW2_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06i.bak"
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

bak = path + ".fn8_06i.bak"
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
