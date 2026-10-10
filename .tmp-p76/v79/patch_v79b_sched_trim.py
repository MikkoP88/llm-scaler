#!/opt/venv/bin/python
"""llm-scaler v79b: scheduler-side DFLASH2_EMIT_K trim (one site).

Patches update_draft_token_ids() in the in-image
vllm/v1/core/sched/scheduler.py so the stored spec row is trimmed to
DFLASH2_EMIT_K when set (1..num_spec_tokens). num_tokens_with_spec =
len(output_token_ids) + len(spec_token_ids), so the trimmed row makes
the spec-block arithmetic schedule width k+1 verify rows — uniform at
udql=k+1, matching the v42d graph lattice. Idempotent; asserts anchor
uniqueness; byte-compiles the result.
"""
import hashlib
import py_compile
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
MARKER = "llm-scaler v79b"
ANCHOR = "            request.spec_token_ids = spec_token_ids\n"

REPL = (
    "            # llm-scaler v79b: honor DFLASH2_EMIT_K at draft-token\n"
    "            # storage. Trimming the stored row makes\n"
    "            # num_tokens_with_spec (= output + spec row) schedule\n"
    "            # width k+1 verify rows — uniform at udql=k+1, matching\n"
    "            # the v42d graph lattice. Same contract as the daily\n"
    "            # near-finish clamp; the drafter still drafts all\n"
    "            # positions (trim is scheduler-side, like the daily\n"
    "            # near-finish trim).\n"
    "            _v79_k = getattr(self, \"_v79_emit_k\", False)\n"
    "            if _v79_k is False:\n"
    "                _os = __import__(\"os\")\n"
    "                _e = _os.environ.get(\"DFLASH2_EMIT_K\", \"\")\n"
    "                _v79_k = None\n"
    "                if _e.isdigit() and 1 <= int(_e) <= (\n"
    "                    self.num_spec_tokens or 0\n"
    "                ):\n"
    "                    _v79_k = int(_e)\n"
    "                self._v79_emit_k = _v79_k\n"
    "            if _v79_k is not None and len(spec_token_ids) > _v79_k:\n"
    "                spec_token_ids = spec_token_ids[:_v79_k]\n"
    + ANCHOR
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
assert src.count(ANCHOR) == 1, "anchor not unique after patch"
py_compile.compile(P, doraise=True)
print("COMPILE_OK")
print("md5", hashlib.md5(src.encode()).hexdigest())
sys.exit(0)
