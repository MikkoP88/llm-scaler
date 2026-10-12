#!/usr/bin/env python3
"""FN8_04b (DEC40 D1): proposer target-layout unwrap for qwen4_exp FCG.

Boot-14b law: patch_04 constructs the draft and loads ALL its weights under
TP2+EP2, then llm_base_proposer.py dies at _maybe_share_embeddings (line
~1333): "Target model does not have 'model' attribute".  Root cause dumped
2026-10-12: the call site does

    if supports_multimodal(target_model): ... get_language_model()
    else: target_language_model = target_model

and our Qwen4ExpForConditionalGeneration is a multimodal-STYLE wrapper
(text lives at language_model.model) that supports_multimodal() reports
False for — so the raw FCG is passed through and has no `.model`.

The MTP branches below the unwrap are already correct (share_embeddings /
share_lm_head True; draft payload exposes .model.embed_tokens and .lm_head).

Edit: in the `else` branch, unwrap `.language_model` when the wrapper has no
`.model` (idempotent; py_compile gate before write; backup .fn8_04b.bak;
anchor must occur exactly once).
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_04B"
REL = "v1/spec_decode/llm_base_proposer.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

ANCHOR = (
    "        else:\n"
    "            target_language_model = target_model\n"
    "\n"
    "        self._maybe_share_embeddings(target_language_model)\n"
    "        self._maybe_share_lm_head(target_language_model)\n"
)
REPLACEMENT = (
    "        else:\n"
    "            target_language_model = target_model\n"
    "            # FN8_04b (DEC40 D1): text-only FCG wrappers (qwen4_exp\n"
    "            # flash-next) nest the causal model at language_model.model;\n"
    "            # supports_multimodal is False for them yet the wrapper has\n"
    "            # no .model — unwrap so the MTP share branches find the\n"
    "            # target embed_tokens / lm_head.\n"
    "            if not hasattr(target_language_model, \"model\") and hasattr(\n"
    "                target_language_model, \"language_model\"\n"
    "            ):\n"
    "                target_language_model = target_language_model.language_model\n"
    "\n"
    "        self._maybe_share_embeddings(target_language_model)\n"
    "        self._maybe_share_lm_head(target_language_model)\n"
)
ALREADY = "FN8_04b (DEC40 D1)"


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_04b.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

if ALREADY in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

n = src.count(ANCHOR)
if n != 1:
    print(f"{TAG} FAIL: anchor occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_04b.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src.replace(ANCHOR, REPLACEMENT)
with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8"
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

with open(path, "w", encoding="utf-8") as f:
    f.write(out)
print(f"{TAG} [{REL}] edited, md5={md5(path)}")
print(f"{TAG} ALL_EDITS_OK")
