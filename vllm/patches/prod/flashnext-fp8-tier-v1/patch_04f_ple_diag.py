#!/usr/bin/env python3
"""FN8_04f (DEC40 D1 diagnostics): dump the exact GDNAttentionMetadata state
at the PLE metadata failure (04e-era raise fired with masks=None AND
non_spec fields=None — impossible per every gdn_attn.py build() branch, so
capture ground truth: step counters, field None-ness, context dict keys,
prefix; log at _ple_forward_flat entry per unique (mode) too).

Diagnostic only — replaces the bare raise with a dump+raise; adds an INFO
log at flat entry.  Idempotent; py_compile gate; backup .fn8_04f.bak.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_04F"
REL = "model_executor/models/qwen4_exp.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD1_LF = '''        if qsl is None or block_ids is None:
            raise RuntimeError(
                "Qwen4Exp PLE: missing query_start_loc/state indices in "
                f"attention metadata ({type(md)})"
            )
'''
NEW1_LF = '''        if qsl is None or block_ids is None:
            _fd = {
                k: (getattr(md, k, None) is not None)
                for k in (
                    "spec_query_start_loc", "non_spec_query_start_loc",
                    "spec_state_indices_tensor",
                    "non_spec_state_indices_tensor", "spec_sequence_masks",
                    "spec_token_indx", "non_spec_token_indx",
                    "num_accepted_tokens", "chunk_indices", "chunk_offsets",
                    "has_initial_state", "nums_dict",
                )
            }
            _cnt = {
                k: getattr(md, k, None)
                for k in ("num_prefills", "num_decodes", "num_spec_decodes",
                          "num_spec_decode_tokens", "num_actual_tokens")
            }
            _keys = list(raw.keys()) if isinstance(raw, dict) else "NOT_DICT"
            logger.error(
                "FN8_PLE_DIAG prefix=%s ctx_keys=%s fields_set=%s counts=%s "
                "spec_mode=%s",
                self.linear_attn.prefix, _keys, _fd, _cnt,
                getattr(self, "_ple_spec_mode", None),
            )
            raise RuntimeError(
                "Qwen4Exp PLE: missing query_start_loc/state indices in "
                f"attention metadata ({type(md)})"
            )
'''

OLD2_LF = '''        # DEC40 D1 (04e): spec verify takes the dedicated spec path
        if getattr(attn_md, "spec_sequence_masks", None) is not None:
            return self._ple_forward_spec(hidden_states, input_ids, attn_md)
'''
NEW2_LF = '''        # DEC40 D1 (04e): spec verify takes the dedicated spec path
        _masks = getattr(attn_md, "spec_sequence_masks", None)
        if _FN8_PLE_DIAG_SEEN.add(
            (
                _masks is not None,
                getattr(attn_md, "num_prefills", -1),
                getattr(attn_md, "num_decodes", -1),
                getattr(attn_md, "num_spec_decodes", -1),
            )
        ):
            logger.info(
                "FN8_PLE_DIAG_ENTRY masks=%s npf=%s ndec=%s nspec=%s T=%d",
                _masks is not None,
                getattr(attn_md, "num_prefills", -1),
                getattr(attn_md, "num_decodes", -1),
                getattr(attn_md, "num_spec_decodes", -1),
                hidden_states.shape[0],
            )
        if _masks is not None:
            return self._ple_forward_spec(hidden_states, input_ids, attn_md)
'''

OLD3_LF = "_PLE_TELEM_WIN = {\"calls\": 0, \"reqs\": 0, \"toks\": 0, \"s\": 0.0, \"max_ms\": 0.0}\n"
NEW3_LF = (
    "_PLE_TELEM_WIN = {\"calls\": 0, \"reqs\": 0, \"toks\": 0, \"s\": 0.0, "
    "\"max_ms\": 0.0}\n"
    "_FN8_PLE_DIAG_SEEN = set()\n"
)


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_04f.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if "FN8_PLE_DIAG" in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


for name, old_lf in (
    ("raise_dump", OLD1_LF),
    ("flat_entry", OLD2_LF),
    ("seen_set", OLD3_LF),
):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_04f.bak"
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
