#!/usr/bin/env python3
"""stage_boot_t0.py — build repro_bootV1222_T0.sh from certified V1221 lineage.

Derived copy (auditable one-anchor insert): inserts the T0 parser decision
(qwen3_coder, the flag the user ran by hand all day on Sep 27) immediately
before the serve launch. Everything else byte-identical to V1221.
"""
SRC = "/root/build/repro_bootV1221.sh"
DST = "/root/build/repro_bootV1222_T0.sh"
ANCHOR = "docker exec -d lsv-test bash /root/serve_user.sh"
INSERT = (
    '# [v89 T0] parser decision: qwen3_coder (ran clean all day Sep 27 via hand launch)\n'
    'docker exec lsv-test sed -i "s/--tool-call-parser qwen3_xml/--tool-call-parser qwen3_coder/" /root/serve_user.sh\n'
    'docker exec lsv-test grep -c -- "--tool-call-parser qwen3_coder" /root/serve_user.sh\n'
)

src = open(SRC).read()
assert src.count(ANCHOR) == 1, f"anchor count {src.count(ANCHOR)} != 1"
out = src.replace(ANCHOR, INSERT + ANCHOR)
open(DST, "w").write(out)
print(f"written {DST} ({len(out)} bytes, insert {len(INSERT)} bytes)")
