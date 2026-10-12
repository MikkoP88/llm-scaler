#!/usr/bin/env python3
"""FN8_06e (DEC40 D8s): PLE decode-path host-sync sweep for capture.

D8s crash-4 (08:50:57, with FN8_PLE_STUB=1 active — the 06d stub WORKED):
the whole-step capture got PAST the n-gram embedding and died at the next
blocking site, qwen4_exp.py:626 in _tail_gather:
    if bool(from_old.any().item()):
— a D2H branch inside the captured decode step.  Inventory of the
captured path (_ple_forward_spec + the PLE module forward, the flatten
prefill path stays eager and is untouched):

  :626  _tail_gather    from_old.any().item() branch        <- crash-4
  :1033 non-spec group  int(lens2.max().item()) width alloc
  :1045 non-spec group  torch.all(his).item() + fresh.any().item()

Fixes (all mathematically identical, branchless / shape-static):
  E1 _tail_gather: unconditional torch.where when old_state is given
     (where(False, old, g) == g).  Semantic note: old_state=None with
     short seqs no longer raises — decode always passes conv_state
     (the guard was defensive only).
  E2 non-spec pad width: static upper bound n_nonspec_toks instead of
     the max().item(); pad columns are garbage-but-unread (the same law
     _short_conv already documents).  lens2 (true lengths) still feeds
     ple() so state math is unchanged.
  E3 fresh-state seed: unconditional masked write prev2[fresh] = eos
     (device masked_fill kernel — capture-legal; all-False mask is a
     no-op numerically).

Also expected downstream (NOT fixed here — next probe answers it): the
ESIMD moe kernels behind the tier have never executed under capture in
this series (PLE precedes moe in every layer and crashed first every
time).  xpu.py's own comment: "ESIMD MoE kernels do their own host-side
allocation / sync and cannot live inside a captured graph."

Idempotent; py_compile gate; backup .fn8_06e.bak; EOL aware;
anchor==1 gates; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06e"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "06e:"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- E1: _tail_gather branchless ----------------------------------------------
OLD1_LF = '''        gathered = x.gather(1, xi_safe.unsqueeze(-1).expand(R, state_len, x.shape[-1]))
        if bool(from_old.any().item()):
            if old_state is None:
                raise ValueError("_tail_gather needs old_state for short sequences")
            gathered = torch.where(from_old.unsqueeze(-1), old_state, gathered)
        return gathered
'''
NEW1_LF = '''        gathered = x.gather(1, xi_safe.unsqueeze(-1).expand(R, state_len, x.shape[-1]))
        if old_state is not None:
            # 06e: unconditional where — mathematically identical
            # (where(False, old, g) == g) and capture-legal; the
            # from_old.any().item() host branch was a D2H sync inside
            # the captured decode step (D8s crash-4 site).  old_state
            # is always passed on the decode path.
            gathered = torch.where(from_old.unsqueeze(-1), old_state, gathered)
        return gathered
'''

# --- E2: non-spec static pad width ----------------------------------------------
OLD2_LF = '''                lens2 = nq[1:] - nq[:-1]
                max_len2 = int(lens2.max().item())
'''
NEW2_LF = '''                lens2 = nq[1:] - nq[:-1]
                # 06e: static shape-only upper bound — lens2.max().item()
                # was a D2H sync inside the captured decode step; pad
                # columns are garbage-but-unread (the _short_conv law).
                max_len2 = n_nonspec_toks
'''

# --- E3: fresh-state seed branchless ---------------------------------------------
OLD3_LF = '''                his = getattr(md, "has_initial_state", None)
                if his is not None and not bool(torch.all(his[:N]).item()):
                    fresh = ~his[:N].to(torch.bool)
                    if bool(fresh.any().item()):
                        prev2[fresh] = eos
'''
NEW3_LF = '''                his = getattr(md, "has_initial_state", None)
                if his is not None:
                    # 06e: branchless fresh-seed — the masked write is a
                    # device kernel (capture-legal); an all-False mask is
                    # a numerical no-op.  The .item() branches were D2H
                    # syncs inside the captured decode step.
                    fresh = ~his[:N].to(torch.bool)
                    prev2[fresh] = eos
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06e.bak"
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


for name, old_lf in (
    ("tail_gather", OLD1_LF),
    ("nonspec_width", OLD2_LF),
    ("nonspec_fresh", OLD3_LF),
):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_06e.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = (
    src.replace(ed(OLD1_LF), ed(NEW1_LF))
    .replace(ed(OLD2_LF), ed(NEW2_LF))
    .replace(ed(OLD3_LF), ed(NEW3_LF))
)

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
