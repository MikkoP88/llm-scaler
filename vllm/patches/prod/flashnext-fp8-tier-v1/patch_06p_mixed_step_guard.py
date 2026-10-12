#!/usr/bin/env python3
"""FN8_06p (d10 live-death fix): PLE mixed-step fallback instead of raise.

d10 live leg died 16:00:12 2026-10-12 under real traffic:
  - V63/V64 interleave budgets produced a MIXED spec+non-spec step
    (FN8_06j spec_fix S=1 L=5 n=5 T=8: one spec verify row + a 3-token
    chunk tail from the co-running request's chunked prefill).
  - The GDN metadata builder supplied NO non-spec group fields for that
    shape (non_spec_query_start_loc / non_spec_state_indices_tensor
    None) -> the 06-series guard raised ->
    RuntimeError -> EngineDeadError -> every subsequent request 500.
  - Two earlier mixed steps DID get metadata and ran the ns group fine
    (FN8_PLE_PATH ns=2, spec_ns N=1 max_len2=1825) — the builder's
    coverage is shape-dependent, so the engine-death condition can
    recur whenever it declines to build the group.

Fix: replace the raise with a logged degrade —
  - run the non-spec rows through _ple_forward_stateless (zero conv
    state == the correct seed for a first chunk; a continuation chunk
    keeps its last good boundary state — degraded but finite),
  - touch NO pool slots (no cross-seq state corruption),
  - loud FN8_06p ERROR log, first 10 hits + every 100th.
Capture-safety: FULL_DECODE_ONLY dummies are pure spec so this branch
never bakes (nonzero() is shape-dynamic); guarded with
is_current_stream_capturing() regardless.

Companion serve-side disables (d11 env): VLLM_V63_CONTENDED_BUDGET=0,
VLLM_V64_DECODE_INTERLEAVE=0, VLLM_V60_ZOMBIE_STRIKES=0 (the 27B-lane
guard misfired strike-1 on a real 5904-token chunked-prefill request).

Edit (qwen4_exp.py, _ple_forward_spec tail):
  P1  raise -> logged stateless fallback.

Idempotent; py_compile gate; backup .fn8_06p.bak; EOL aware;
anchor==1; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06p"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06p"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD1_LF = '''        elif n_nonspec_toks > 0:
            raise RuntimeError(
                "Qwen4Exp PLE spec: non-spec tokens present but "
                "non-spec metadata missing"
            )
        return outs
'''

NEW1_LF = '''        elif n_nonspec_toks > 0:
            # 06p (d10 live-death fix): mixed spec+non-spec step where
            # the GDN builder supplied NO non-spec group metadata
            # (2026-10-12 16:00:12 under V63/V64 interleave: T=8 =
            # 5 spec + 3 chunk tail, nq/n_slots None -> the old raise
            # killed EngineCore and every later request 500'd; two
            # earlier mixed steps HAD metadata and ran the ns group
            # fine, so builder coverage is shape-dependent).  Degrade
            # instead of dying: run the non-spec rows through the
            # stateless PLE pass (zero conv state == the correct seed
            # for a first chunk; a continuation chunk keeps its last
            # good boundary state), touch NO pool slots, log loudly.
            # Capture never takes this branch (FULL_DECODE_ONLY
            # dummies are pure spec; nonzero() is shape-dynamic) but
            # guard anyway.
            _hits = globals().setdefault("_FN8_06P_HITS", [0])
            _hits[0] += 1
            try:
                _capt = torch.xpu.is_current_stream_capturing()
            except Exception:
                _capt = False
            if _hits[0] <= 10 or _hits[0] % 100 == 0:
                logger.error(
                    "FN8_06p PLE mixed-step fallback: T=%d n_spec=%d "
                    "nonspec=%d capturing=%s hit=%d",
                    T, n_spec_toks, n_nonspec_toks, _capt, _hits[0])
            if not _capt:
                _mask = hidden_states.new_ones(T, dtype=torch.bool)
                _mask[s_indx] = False
                _ns_rows = _mask.nonzero(as_tuple=True)[0]
                if _ns_rows.numel() > 0:
                    outs[_ns_rows] = self._ple_forward_stateless(
                        hidden_states[_ns_rows], input_ids[_ns_rows]
                    )
        return outs
'''

EDITS = [
    ("P1_mixed_step_fallback", OLD1_LF, NEW1_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06p.bak"
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

bak = path + ".fn8_06p.bak"
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
