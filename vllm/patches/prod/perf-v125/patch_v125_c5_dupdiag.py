#!/usr/bin/env python3
"""patch_v125_c5_dupdiag.py — v125 P22C5 diagnostic add-on.

The static fp8 spec bridge convicts a real correctness defect at n>1
(x8-identical 0/10 clean rounds; pair20 10/20; slot-stable BAD streams;
n=1 probes all clean). Top hypotheses split on evidence we don't yet have:
duplicate slot ids across the gathered rows (prefix-cache-shared
spec_state_indices -> lossy index_copy_ scatter) vs capture-replay
interaction vs deeper kernel semantics.

This patch adds a ONCE-ONLY diagnostic print at the first n>1 static
gather: n, W, row count, unique-row count, duplicate count, and the first
20 slot ids. Guarded against firing inside graph capture (torch.unique +
tolist would poison a capturing stream), so it is only informative in
EAGER mode — p22c5_diag2.sh boots exactly that leg.

MUST be applied AFTER patch_v125_s4fp8_bridge.py (anchors the C5-inserted
construction). Idempotent via the 'v125 C5DUPD' marker. Run inside the
container:
  docker cp patch_v125_c5_dupdiag.py lsv-test:/root/
  docker exec lsv-test python3 /root/patch_v125_c5_dupdiag.py
"""
import py_compile
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "v125 C5DUPD"

src = open(P).read()

if MARKER in src:
    print("V125_C5DUPD_ALREADY")
    sys.exit(0)

if "v125 P22C5" not in src:
    print("ABORT_DUPDIAG_NEEDS_C5_FIRST")
    sys.exit(1)

ANCHOR = "            _static_flat = _st.reshape(-1).to(torch.int64)"
assert src.count(ANCHOR) == 1, f"anchor not unique (count={src.count(ANCHOR)})"

NEW = ANCHOR + """
            if not globals().get("_GDN_FP8_DUPDIAG_DONE"):
                _capturing = (
                    hasattr(torch.xpu, "is_current_stream_capturing")
                    and torch.xpu.is_current_stream_capturing()
                )
                if not _capturing:
                    globals()["_GDN_FP8_DUPDIAG_DONE"] = True
                    _u = torch.unique(_static_flat)
                    print(
                        "v125 C5DUPD n=%d W=%d rows=%d uniq=%d dup=%d"
                        " slots=%s"
                        % (
                            num_spec_decodes,
                            int(_W),
                            int(_static_flat.numel()),
                            int(_u.numel()),
                            int(_static_flat.numel() - _u.numel()),
                            _static_flat.tolist()[:20],
                        ),
                        flush=True,
                    )"""

src = src.replace(ANCHOR, NEW, 1)

bak = P + ".pre_v125_dupdiag"
try:
    open(bak)
except FileNotFoundError:
    with open(bak, "w") as f:
        f.write(open(P).read())
    print(f"backup written: {bak}")

with open(P, "w") as f:
    f.write(src)

py_compile.compile(P, doraise=True)
print("V125_C5DUPD_OK")
