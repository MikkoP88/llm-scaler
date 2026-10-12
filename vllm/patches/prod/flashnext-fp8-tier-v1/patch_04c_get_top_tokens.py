#!/usr/bin/env python3
"""FN8_04c (DEC40 D1): Qwen4ExpMTP.get_top_tokens for TP MTP greedy drafting.

Boot law (D1 attempt 1, 2026-10-12 02:04): after patch_04b the proposer
shares embed+lm_head and then dies at llm_base_proposer.py:1514:

    ValueError: use_local_argmax_reduction is enabled but draft model
    Qwen4ExpMTP does not implement get_top_tokens().

_greedy_sample feeds the draft forward()'s RAW output to get_top_tokens
(qwen3_5 law: predictor output is final-hidden).  OUR forward returns the
pre-mixer (T, hc*H) hyper stream (dsv4 law), so get_top_tokens must apply
the deferred mixer first — the same collapse compute_logits does — then
delegate to LogitsProcessor.get_top_tokens (local argmax, O(2*tp) comm
instead of full-vocab all-gather; also serves VLLM_XPU_MTP_LOCAL_ARGMAX
batch=1, v25 KNOWN_ISSUES #11).

Idempotent; py_compile gate; backup .fn8_04c.bak; anchor occurs once.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_04C"
REL = "model_executor/models/qwen4_exp_mtp.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

ANCHOR = (
    "    def load_weights(\n"
    "        self,\n"
    "        weights: Iterable[tuple[str, torch.Tensor]]\n"
    "    ) -> set[str]:\n"
)
METHOD = (
    "    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    "        \"\"\"Greedy draft tokens without full-vocab logits (TP MTP law:\n"
    "        fork's use_local_argmax_reduction / VLLM_XPU_MTP_LOCAL_ARGMAX\n"
    "        path).  forward() returns the raw pre-mixer hyper stream (dsv4\n"
    "        law) — collapse it here exactly like compute_logits before the\n"
    "        local argmax.\"\"\"\n"
    "        mixed = self.model.hyper_connection_mixer(hidden_states)\n"
    "        return self.logits_processor.get_top_tokens(self.lm_head, mixed)\n"
    "\n"
)
ALREADY = "FN8_04C"  # placeholder; real idempotence marker below
IDEMPOTENT = "def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:"


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_04c.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

if "get_top_tokens" in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

n = src.count(ANCHOR)
if n != 1:
    print(f"{TAG} FAIL: anchor occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_04c.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src.replace(ANCHOR, METHOD + ANCHOR)
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
