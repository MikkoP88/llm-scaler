#!/usr/bin/env python3
"""FN8_06j (DEC40): comprehensive PLE-path telemetry (user directive
2026-10-12: "add comprehensive telemetry points capture issue to
helping solve issues").

Forensics that motivated each counter (this series, banked
/stage/bank/serve_d9b_fatal.log + d9b_dead.log):

  1.  "Did capture bake the bank fast path or the op?" took TWO fatal
      boots to answer (FN8_PLE_OP calls=2/PRE calls=0 inference).
      -> CAPTURE-COMBO SENTINELS: one INFO line the first time each
      (capturing, call-site, group) combination executes, with tensor
      shapes + bank cap.  Directly prints what the captured graph
      baked.
  2.  Which call site ran how often (flat / stateless / spec / ns) and
      fast-path vs op split, cumulative + at shutdown.
      -> per-site counters in _FN8_PLE_PATH_REC; fast/op split per
      group in _FN8_PLE_FAST_REC; both appended to the atexit dump.
  3.  Prestage staging stats per group (rows, overflow, ms) —
      _FN8_PLE_PRE_REC is global-only today; per-group rows tell
      whether the hook staged the shapes the graphs assume.
      -> _FN8_PLE_PRE_G_REC {group: rows}; appended to the dump.
  4.  Capture-phase marker at the sentinel: includes whether
      is_current_stream_capturing() at EVERY site's first call (eager
      first vs captured first distinguishes warmup order).

Volume discipline: sentinels fire at most once per combo per process
(~6 lines/boot); counters are dict increments on the step path
(nanoseconds); the dump is atexit-only.  Nothing on the hot path logs
per-step.

Edits:
  J1  module telemetry records + extended _fn8_ple_op_dump.
  J2  embedding.forward: fast/op split counters + capture sentinel
      (site="emb", group, numel vs cap).
  J3  _ple_forward_flat call site: counter + sentinel site="flat".
  J4  _ple_forward_stateless call site: counter + sentinel
      site="stateless".
  J5  _ple_forward_spec spec call site: counter + sentinel
      site="spec".
  J6  _ple_forward_spec ns call site: counter + sentinel site="ns".
  J7  embedding.fn8_prestage: per-group row/overflow counters.

Idempotent; py_compile gate; backup .fn8_06j.bak; EOL aware;
anchor==1 per edit; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06j"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06j"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# J1: telemetry records + dump extension.
OLD1_LF = '''def _fn8_ple_op_dump():
    import sys as _s
    print(f"FN8_PLE_OP {_FN8_PLE_OP_REC}", file=_s.stderr, flush=True)
    print(f"FN8_PLE_PRE {_FN8_PLE_PRE_REC}", file=_s.stderr, flush=True)
'''
NEW1_LF = '''# FN8_06j: per-site / per-path / per-group PLE telemetry (see
# patch_06j header for the forensic each counter answers).
_FN8_PLE_PATH_REC = {"flat": 0, "stateless": 0, "spec": 0, "ns": 0,
                     "fast": 0, "op": 0, "stub": 0}
_FN8_PLE_FAST_REC = {}          # group -> [fast_calls, op_calls]
_FN8_PLE_PRE_G_REC = {}         # group -> [rows, overflows]
_FN8_PLE_CAPT_SEEN = set()      # (site, group, capturing) sentinels


def _fn8_ple_capt_sentinel(site, group, extra=""):
    """One INFO line per new (site, group, capturing) combo — prints
    what the captured graph baked (or what eager ran first)."""
    _cap = (_FN8_XPU_CAPTURING is not None and _FN8_XPU_CAPTURING())
    _key = (site, group, _cap)
    if _key in _FN8_PLE_CAPT_SEEN:
        return
    _FN8_PLE_CAPT_SEEN.add(_key)
    logger.info(
        "FN8_06j CAPTURE_SENTINEL site=%s group=%s capturing=%s %s",
        site, group, _cap, extra)


def _fn8_ple_op_dump():
    import sys as _s
    print(f"FN8_PLE_OP {_FN8_PLE_OP_REC}", file=_s.stderr, flush=True)
    print(f"FN8_PLE_PRE {_FN8_PLE_PRE_REC}", file=_s.stderr, flush=True)
    print(f"FN8_PLE_PATH {_FN8_PLE_PATH_REC}", file=_s.stderr, flush=True)
    print(f"FN8_PLE_FAST {dict(_FN8_PLE_FAST_REC)}", file=_s.stderr,
          flush=True)
    print(f"FN8_PLE_PRE_G {dict(_FN8_PLE_PRE_G_REC)}", file=_s.stderr,
          flush=True)
    print(f"FN8_PLE_CAPT_SEEN {sorted(_FN8_PLE_CAPT_SEEN)}",
          file=_s.stderr, flush=True)
'''

# J2: embedding.forward fast-path counters + sentinel.
OLD2_LF = '''            return _fn8_bank[:ngram_ids.numel()].view(
                *ngram_ids.shape[:-1], self.embedding_dim)
'''
NEW2_LF = '''            _FN8_PLE_PATH_REC["fast"] += 1
            _fg = _FN8_PLE_FAST_REC.setdefault(fn8_group, [0, 0])
            _fg[0] += 1
            _fn8_ple_capt_sentinel(
                "emb", fn8_group,
                extra=f"numel={ngram_ids.numel()} cap={_fn8_bank.shape[0]}")
            return _fn8_bank[:ngram_ids.numel()].view(
                *ngram_ids.shape[:-1], self.embedding_dim)
'''

# J2b: op-path counter (the stub branch keeps its own counter).
OLD2B_LF = '''        emb = torch.ops.vllm.ple_ngram_gather(
            ngram_ids, self._host_table, self._host_scale_inv,
            self._host_scale, self._col_block_size or 0, self.embedding_dim,
        )
'''
NEW2B_LF = '''        _FN8_PLE_PATH_REC["op"] += 1
        if fn8_group is not None:
            _fg = _FN8_PLE_FAST_REC.setdefault(fn8_group, [0, 0])
            _fg[1] += 1
        else:
            _FN8_PLE_FAST_REC.setdefault("None", [0, 0])[1] += 1
        _fn8_ple_capt_sentinel(
            "emb_op", str(fn8_group),
            extra=f"numel={ngram_ids.numel()}")
        emb = torch.ops.vllm.ple_ngram_gather(
            ngram_ids, self._host_table, self._host_scale_inv,
            self._host_scale, self._col_block_size or 0, self.embedding_dim,
        )
'''

# J3: flat call-site counter + sentinel.
OLD3_LF = '''        out_pad, new_conv_state, new_ctx = ple(
            hidden_pad, token_history, lens, conv_state, fn8_group="ns")
'''
NEW3_LF = '''        _FN8_PLE_PATH_REC["flat"] += 1
        _fn8_ple_capt_sentinel(
            "flat", "ns",
            extra=f"reqs={num_reqs} max_len={max_len} T={T}")
        out_pad, new_conv_state, new_ctx = ple(
            hidden_pad, token_history, lens, conv_state, fn8_group="ns")
'''

# J4: stateless call-site counter + sentinel.
OLD4_LF = '''        out_pad, _, _ = ple(
            hidden_states.unsqueeze(0), token_history, lens, conv_state,
            fn8_group="ns")
'''
NEW4_LF = '''        _FN8_PLE_PATH_REC["stateless"] += 1
        _fn8_ple_capt_sentinel("stateless", "ns", extra=f"T={T}")
        out_pad, _, _ = ple(
            hidden_states.unsqueeze(0), token_history, lens, conv_state,
            fn8_group="ns")
'''

# J5: spec call-site counter + sentinel.
OLD5_LF = '''            out_pad, _, _ = ple(hs_pad, token_history, lens, conv_state,
                                fn8_group="spec")
'''
NEW5_LF = '''            _FN8_PLE_PATH_REC["spec"] += 1
            _fn8_ple_capt_sentinel(
                "spec", "spec", extra=f"S={S} L={L}")
            out_pad, _, _ = ple(hs_pad, token_history, lens, conv_state,
                                fn8_group="spec")
'''

# J6: ns call-site counter + sentinel.
OLD6_LF = '''                o2, nc2, nw2 = ple(hp2, th2, lens2, cs2, fn8_group="ns")
'''
NEW6_LF = '''                _FN8_PLE_PATH_REC["ns"] += 1
                _fn8_ple_capt_sentinel(
                    "spec_ns", "ns",
                    extra=f"N={N} max_len2={max_len2}")
                o2, nc2, nw2 = ple(hp2, th2, lens2, cs2, fn8_group="ns")
'''

# J7: prestage per-group counters.
OLD7_LF = '''        if m > dev.shape[0]:
            _FN8_PLE_PRE_REC["overflow"] += 1
            return
'''
NEW7_LF = '''        if m > dev.shape[0]:
            _FN8_PLE_PRE_REC["overflow"] += 1
            _FN8_PLE_PRE_G_REC.setdefault(group, [0, 0])[1] += 1
            return
'''

EDITS = [
    ("J1_records_dump", OLD1_LF, NEW1_LF),
    ("J2_fastpath_counters", OLD2_LF, NEW2_LF),
    ("J2b_oppath_counters", OLD2B_LF, NEW2B_LF),
    ("J3_flat_site", OLD3_LF, NEW3_LF),
    ("J4_stateless_site", OLD4_LF, NEW4_LF),
    ("J5_spec_site", OLD5_LF, NEW5_LF),
    ("J6_ns_site", OLD6_LF, NEW6_LF),
    ("J7_prestage_groups", OLD7_LF, NEW7_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06j.bak"
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

bak = path + ".fn8_06j.bak"
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
