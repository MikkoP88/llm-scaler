#!/bin/bash
# mk_serve_boot_min2.sh (host: /root/build) — generate serve_boot_min2.sh:
# serve_boot_var.sh with the baked-in --compilation-config extended to
# cudagraph_capture_sizes=[1,2,4,8]. Needed because ${EXTRAFLAGS} lands BEFORE
# the baked --compilation-config in serve_boot_var.sh, so an extra
# --compilation-config is silently overridden (learned the hard way: the first
# E-min attempt booted the default <=128 list while reporting success).
# Usage: bash mk_serve_boot_min2.sh   (then serve_boot_min2.sh "" "" <log> "" 512 <img>)
python3 - <<'PYEOF'
p = '/root/build/serve_boot_var.sh'
s = open(p).read()
needle = 'FULL_DECODE_ONLY\\"}'
repl = 'FULL_DECODE_ONLY\\",\\"cudagraph_capture_sizes\\":[1,2,4,8]}'
if needle not in s:
    print('NEEDLE_MISSING — serve_boot_var.sh layout changed?'); raise SystemExit(1)
s = s.replace(needle, repl, 1)
open('/root/build/serve_boot_min2.sh', 'w').write(s)
print('WROTE /root/build/serve_boot_min2.sh')
PYEOF
chmod +x /root/build/serve_boot_min2.sh
bash -n /root/build/serve_boot_min2.sh && echo SYNTAX_OK
grep -n 'compilation-config' /root/build/serve_boot_min2.sh
