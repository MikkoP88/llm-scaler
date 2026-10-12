#!/usr/bin/env python3
"""FN8_04d (DEC40 D1): enable the Qwen4Exp PLE path under MTP spec verify
+ comprehensive PLE telemetry (user directive 2026-10-12).

Boot law (D1, 2026-10-12 03:19): the FULL spec stack boots (drafter shares
embed+lm_head, local argmax, eagle kernels JIT'd, v60c guard armed, drafter
proposal ran) and the first target VERIFY forward dies at the P4-era guard
qwen4_exp.py:665:

    NotImplementedError: Qwen4Exp PLE does not support speculative decoding
    yet (P4); run without a speculative config.

Why the guard can fall (D1 analysis, same session):
  * spec verify = k+1 multi-token queries per seq — the EXACT (R, L) shape
    prefill already exercises; _ple_forward_flat is length-generic
    (searchsorted flatten -> pad, causal token_history keys, _tail_gather
    state extraction);
  * PLE state rows were DESIGNED for spec — get_state_shape appends
    +num_spec tails to both PLE slots so the fork's eagle pool machinery
    slices them exactly like the native GDN conv/temporal slots;
  * the only missing piece was field selection: under spec_sequence_masks
    the metadata must feed the SPEC-aware query_start_loc /
    state_indices_tensor (the fields the native GDN core consumes), not the
    non_spec_* single-token view.

Telemetry (same patch, three surgical edits):
  T1  module helper _ple_telemetry: cumulative mode mix (spec share) +
      200-call window of reqs/call, toks/call, avg/max ms + periodic
      isfinite NaN guard on the PLE output (PLE-state corruption watch).
  T2  _ple_metadata records _ple_spec_mode for T1.
  T3  _ple_forward_flat call site: time the ple(...) call, feed T1.

Idempotent; py_compile gate; backup .fn8_04d.bak; EOL-style aware.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_04D"
REL = "model_executor/models/qwen4_exp.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD1_LF = (
    '        if getattr(md, "spec_sequence_masks", None) is not None:\n'
    '            raise NotImplementedError(\n'
    '                "Qwen4Exp PLE does not support speculative decoding yet (P4); "\n'
    '                "run without a speculative config."\n'
    '            )\n'
    '        qsl = getattr(md, "non_spec_query_start_loc", None)\n'
    '        if qsl is None:\n'
    '            qsl = getattr(md, "query_start_loc", None)\n'
    '        block_ids = getattr(md, "non_spec_state_indices_tensor", None)\n'
    '        if block_ids is None:\n'
    '            block_ids = getattr(md, "state_indices_tensor", None)\n'
)
NEW1_LF = (
    '        if getattr(md, "spec_sequence_masks", None) is not None:\n'
    '            # DEC40 D1 law (2026-10-12): spec verify = multi-token queries\n'
    '            # per seq -- the exact shape prefill already exercises -- so take\n'
    '            # the spec-aware fields (the same ones the native GDN core\n'
    '            # consumes under spec).  PLE state rows carry the +num_spec tails\n'
    '            # (get_state_shape), so the fork\'s eagle pool machinery slices\n'
    '            # them like the native conv/temporal slots.\n'
    '            self._ple_spec_mode = True\n'
    '            qsl = getattr(md, "query_start_loc", None)\n'
    '            block_ids = getattr(md, "state_indices_tensor", None)\n'
    '        else:\n'
    '            self._ple_spec_mode = False\n'
    '            qsl = getattr(md, "non_spec_query_start_loc", None)\n'
    '            if qsl is None:\n'
    '                qsl = getattr(md, "query_start_loc", None)\n'
    '            block_ids = getattr(md, "non_spec_state_indices_tensor", None)\n'
    '            if block_ids is None:\n'
    '                block_ids = getattr(md, "state_indices_tensor", None)\n'
)

OLD2_LF = "logger = init_logger(__name__)\n"
NEW2_LF = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# -- DEC40 D1 PLE telemetry (user directive 2026-10-12) --------------------\n"
    "import time as _time_mod\n"
    "_perf = _time_mod.perf_counter\n"
    "_PLE_TELEM = {\"calls\": 0, \"spec\": 0}\n"
    "_PLE_TELEM_WIN = {\"calls\": 0, \"reqs\": 0, \"toks\": 0, \"s\": 0.0, \"max_ms\": 0.0}\n"
    "\n"
    "\n"
    "def _ple_telemetry(spec_mode, num_reqs, max_len, T, dt, out_pad) -> None:\n"
    "    \"\"\"PLE flat-forward cost + spec-mode mix; one INFO line per 200 calls\n"
    "    plus a periodic isfinite NaN guard on the PLE output.\"\"\"\n"
    "    c, w = _PLE_TELEM, _PLE_TELEM_WIN\n"
    "    c[\"calls\"] += 1\n"
    "    c[\"spec\"] += int(spec_mode)\n"
    "    w[\"calls\"] += 1\n"
    "    w[\"reqs\"] += num_reqs\n"
    "    w[\"toks\"] += T\n"
    "    w[\"s\"] += dt\n"
    "    w[\"max_ms\"] = max(w[\"max_ms\"], dt * 1e3)\n"
    "    if w[\"calls\"] >= 200:\n"
    "        try:\n"
    "            if not bool(out_pad.isfinite().all()):\n"
    "                logger.error(\n"
    "                    \"FN8_PLE_TELEM NaN: ple out_pad non-finite \"\n"
    "                    \"(spec=%s reqs=%d maxlen=%d)\",\n"
    "                    spec_mode, num_reqs, max_len,\n"
    "                )\n"
    "        except Exception:\n"
    "            pass\n"
    "        logger.info(\n"
    "            \"FN8_PLE_TELEM cum_calls=%d cum_spec=%d win_reqs_per_call=%.2f \"\n"
    "            \"win_toks_per_call=%.2f win_avg_ms=%.3f win_max_ms=%.3f\",\n"
    "            c[\"calls\"], c[\"spec\"], w[\"reqs\"] / w[\"calls\"],\n"
    "            w[\"toks\"] / w[\"calls\"], w[\"s\"] / w[\"calls\"] * 1e3, w[\"max_ms\"],\n"
    "        )\n"
    "        w.update(calls=0, reqs=0, toks=0, s=0.0, max_ms=0.0)\n"
)

OLD3_LF = (
    "        out_pad, new_conv_state, new_ctx = ple(hidden_pad, token_history, "
    "lens, conv_state)\n"
)
NEW3_LF = (
    "        _t0 = _perf()\n"
    "        out_pad, new_conv_state, new_ctx = ple(hidden_pad, token_history, "
    "lens, conv_state)\n"
    "        _ple_telemetry(\n"
    "            bool(getattr(self, \"_ple_spec_mode\", False)),\n"
    "            num_reqs, max_len, T, _perf() - _t0, out_pad,\n"
    "        )\n"
)


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_04d.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if "DEC40 D1 law (2026-10-12)" in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


for name, old_lf, new_lf in (
    ("ple_spec_guard", OLD1_LF, NEW1_LF),
    ("telemetry_helper", OLD2_LF, NEW2_LF),
    ("telemetry_callsite", OLD3_LF, NEW3_LF),
):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_04d.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src
for old_lf, new_lf in ((OLD1_LF, NEW1_LF), (OLD2_LF, NEW2_LF), (OLD3_LF, NEW3_LF)):
    out = out.replace(ed(old_lf), ed(new_lf))

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
