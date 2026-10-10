#!/usr/bin/env python3
"""llm-scaler v79a: drafter loop-skip at DFLASH2_EMIT_K=1.

Patches the k==1 early-exit in the bind-source llm_base_proposer.py
(host: /root/build/p77/code/llm_base_proposer.py, ro-mounted into
lsv-test at the vllm spec_decode path) so that when
DFLASH2_EMIT_K=1 the drafter stops after its first forward: position 0
is drafted for real, positions 1..k-1 are zero-filled in the [B, k]
emission buffer. The v79b scheduler trim never schedules those
positions, so the zeros can never reach verify (unlike the stock
fits-in-drafter gate, which SCHEDULES its zeros at width k+1).

Effect: drafter deep-forward exposure drops 4x (one forward per step,
no dforward_loop); propose cost drops accordingly. Idempotent; asserts
anchor uniqueness; byte-compiles the result.
"""
import hashlib
import os
import py_compile
import sys

P = "/root/build/p77/code/llm_base_proposer.py"
MARKER = "llm-scaler v79a"
ANCHOR = (
    "        # Early exit if there is only one draft token to be generated.\n"
    "        if self.num_speculative_tokens == 1 or self.parallel_drafting:\n"
    "            draft_token_ids = self._greedy_sample(sample_hidden_states)\n"
    "            return draft_token_ids.view(-1, self.num_speculative_tokens)\n"
)

REPL = (
    "        # Early exit if there is only one draft token to be generated.\n"
    "        # llm-scaler v79a: DFLASH2_EMIT_K=1 loop-skip — with the v79b\n"
    "        # scheduler trim active, draft positions >= 1 are never\n"
    "        # scheduled or verified, so drafting them is pure wasted\n"
    "        # deep-forward exposure. Draft position 0 for real and\n"
    "        # zero-fill the remaining emission-buffer slots (the zeros\n"
    "        # never leave the buffer: scheduled width is 1).\n"
    "        _v79a_ek = getattr(self, \"_v79a_emit_k\", False)\n"
    "        if _v79a_ek is False:\n"
    "            _os = __import__(\"os\")\n"
    "            _e = _os.environ.get(\"DFLASH2_EMIT_K\", \"\")\n"
    "            _v79a_ek = None\n"
    "            if _e.isdigit() and 1 <= int(_e) <= (\n"
    "                self.num_speculative_tokens or 0\n"
    "            ):\n"
    "                _v79a_ek = int(_e)\n"
    "            self._v79a_emit_k = _v79a_ek\n"
    "        if (\n"
    "            self.num_speculative_tokens == 1\n"
    "            or self.parallel_drafting\n"
    "            or _v79a_ek == 1\n"
    "        ):\n"
    "            draft_token_ids = self._greedy_sample(sample_hidden_states)\n"
    "            if _v79a_ek == 1 and self.num_speculative_tokens > 1:\n"
    "                _b = draft_token_ids.numel()\n"
    "                _pad = self.num_speculative_tokens - 1\n"
    "                return torch.cat(\n"
    "                    [\n"
    "                        draft_token_ids.view(-1, 1),\n"
    "                        draft_token_ids.new_zeros(_b, _pad),\n"
    "                    ],\n"
    "                    dim=1,\n"
    "                )\n"
    "            return draft_token_ids.view(-1, self.num_speculative_tokens)\n"
)

src = open(P).read()
if MARKER in src:
    print("ALREADY_PATCHED")
else:
    n = src.count(ANCHOR)
    assert n == 1, f"anchor count {n} != 1 — aborting"
    src = src.replace(ANCHOR, REPL, 1)
    open(P, "w").write(src)
    print("PATCHED")

src = open(P).read()
assert MARKER in src, "marker missing after patch"
assert src.count(ANCHOR) == 0, "anchor should be fully replaced by patch"
py_compile.compile(P, doraise=True, cfile=os.devnull)
print("COMPILE_OK")
print("md5", hashlib.md5(src.encode()).hexdigest())
sys.exit(0)
