#!/usr/bin/env python3
"""fix_restore_v128c.py — quoting ROOT FIX of the V1227_XGCOMPACT knob.

The 128b block inserted the flag UNQUOTED into the generated
serve_user_v1227.sh cmdline line:

  --structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}

Bash brace-expands `{"a":"x","b":"y"}` into two words, so vllm received
`--structured-outputs-config backend:xgrammar` -> pydantic
"Invalid JSON: expected value at line 1 column 1" -> serve died at
arg-parse (LEG_XGCOMPACT boot 11:35, health stayed 000).

Fix: the serve line must carry the JSON SINGLE-QUOTED:

  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'

Implementation: the boot script's sed/grep lines switch to double-quoted
programs (where \\\" collapses to " at boot-script parse time), so the
replacement/pattern can contain both quote kinds; sed treats bare ' and "
as ordinary replacement chars (only & \\ and the | delimiter are special).
Byte literals below are built from raw strings concatenated at quote
boundaries — no escape-arithmetic.

Idempotent: no-op if the fixed form is present; replaces the broken form
(both restore-script copies).
"""
import subprocess

FILES = ["/root/build/v127_stage/boot_v1227_restore.sh",
         "/root/build/boot_v1227_restore.sh"]

# the JSON blob as it must appear (bash-escaped) inside the double-quoted
# sed program / grep pattern of the boot script
Q = r'{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}'

BROKEN_SED = ('  docker exec lsv-test sed -i \'s|--async-scheduling|'
              '--structured-outputs-config {"backend":"xgrammar",'
              '"disable_any_whitespace":true} --async-scheduling|\' '
              '/root/serve_user_v1227.sh\n')
BROKEN_GATE = ('  docker exec lsv-test grep -q -F -- \'--structured-outputs-config '
               '{"backend":"xgrammar","disable_any_whitespace":true}\' '
               '/root/serve_user_v1227.sh \\\n')

FIXED_SED = ('  docker exec lsv-test sed -i "s|--async-scheduling|'
             "--structured-outputs-config '" + Q + "' --async-scheduling|\" "
             '/root/serve_user_v1227.sh\n')
FIXED_GATE = ('  docker exec lsv-test grep -q -F -- "--structured-outputs-config '
              "'" + Q + "'\" "
              '/root/serve_user_v1227.sh \\\n')

for path in FILES:
    with open(path) as f:
        t = f.read()

    if FIXED_SED in t:
        print(path, "-> fixed form already present, no-op")
        continue

    assert t.count(BROKEN_SED) == 1, \
        "%s broken sed count=%d" % (path, t.count(BROKEN_SED))
    assert t.count(BROKEN_GATE) == 1, \
        "%s broken gate count=%d" % (path, t.count(BROKEN_GATE))
    t = t.replace(BROKEN_SED, FIXED_SED)
    t = t.replace(BROKEN_GATE, FIXED_GATE)

    with open(path, "w") as f:
        f.write(t)

    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    with open(path) as f:
        t2 = f.read()
    print(path)
    print("  fixed sed line present:", FIXED_SED in t2)
    print("  fixed gate line present:", FIXED_GATE in t2)
    print("  broken form gone:", BROKEN_SED not in t2 and BROKEN_GATE not in t2)
    print("  bash -n:", "SYNTAX_OK" if r.returncode == 0 else "FAIL " + r.stderr)

print("fix_restore_v128c DONE")
