#!/usr/bin/env python3
"""mk_ship_v1226.py — build ship_v1226.sh from the certified ship_v1225.sh
(host-side transform).

v126 ship chain is structurally identical to v1225 (clean warm relaunch from
-raw -> lane-commit -> full gates on the committed image -> fresh-boot prod
verify -> CC battery through litellm :4000 -> watchdog repoint+re-arm), with
these v126 deltas:

  - battery completion literal: validate_v1226_run.sh was generated from the
    v1225 script by a v1225->v1226 rename, and its caps literals
    ('VALIDATE_V1225_RUN ALL GATES PASS' / 'V1225_VALIDATION_COMPLETE')
    survived the rename (known cosmetic carryover). The ship precondition
    must therefore grep V1225_VALIDATION_COMPLETE against the *v1226* log.
  - lane-commit cleanup also removes patch_v126_dualbridge.py.
  - fresh-boot verification additionally asserts the v126 DUAL BRIDGE marker
    on the running prod container.

Transforms (ordered):
  1. blanket v1225 -> v1226 (names, logs, boot scripts, prompts, backups)
  2. blanket v1.2.25 -> v1.2.26 (image tags incl. the prod-boot sed)
  3. restore the battery completion grep literal to V1225_VALIDATION_COMPLETE
  4. final images grep range 'v1.2.2[45]' -> 'v1.2.2[56]'
  5. rm list += patch_v126_dualbridge.py
  6. dualbridge fresh-boot gate before the section-5 PASS echo
  7. header note
"""
SRC = "/root/build/ship_v1225.sh"
DST = "/root/build/ship_v1226.sh"

with open(SRC) as f:
    t = f.read()

# 1) + 2) blanket renames
n_v = t.count("v1225")
t = t.replace("v1225", "v1226")
n_img = t.count("v1.2.25")
t = t.replace("v1.2.25", "v1.2.26")

# 2b) uppercase carries the lowercase blanket misses: boot-script names
#     (repro_bootV1225*), the ESCAPED sed pattern v1\.2\.25-raw, and the
#     SHIP_V1225 banner literals. (V1225_VALIDATION_COMPLETE and
#     SANITY_V1225_EXTRA_DONE are deliberately kept — rename carryovers in
#     the v1226 battery/sanity scripts.)
n_boot = t.count("repro_bootV1225")
assert n_boot == 10, "repro_boot refs=%d" % n_boot
t = t.replace("repro_bootV1225", "repro_bootV1226")
n_sed = t.count("v1\\.2\\.25-raw")
assert n_sed == 1, "escaped sed pattern=%d" % n_sed
t = t.replace("v1\\.2\\.25-raw", "v1\\.2\\.26-raw")
n_ban = t.count("SHIP_V1225")
assert n_ban == 5, "ship banners=%d" % n_ban
t = t.replace("SHIP_V1225", "SHIP_V1226")

# 3) battery completion literal: the v1226 battery emits the v1225 caps
#    literal (rename carryover; the lowercase blanket does not touch the
#    uppercase V1225_ literal, so it is already correct — only annotate it).
OLD_PRE = ("grep -q V1225_VALIDATION_COMPLETE /root/build/lce1/v1226_validate_run.log "
           "2>/dev/null || { echo \"SHIP ABORT: validation not complete\"; exit 1; }")
NEW_PRE = ("# v126 note: validate_v1226_run.sh emits the V1225 caps literal (rename "
           "carryover)\n" + OLD_PRE)
assert t.count(OLD_PRE) == 1, "precondition anchor=%d" % t.count(OLD_PRE)
t = t.replace(OLD_PRE, NEW_PRE)

# 4) final images range grep
n_rng = t.count("'v1.2.2[45]'")
assert n_rng == 1, "images range anchor=%d" % n_rng
t = t.replace("'v1.2.2[45]'", "'v1.2.2[56]'")

# 5) lane-commit cleanup adds the v126 patch file
OLD_RM = "/root/patch_v125_c7_fp8state_raise.py 2>/dev/null"
NEW_RM = "/root/patch_v125_c7_fp8state_raise.py /root/patch_v126_dualbridge.py 2>/dev/null"
assert t.count(OLD_RM) == 1, "rm anchor=%d" % t.count(OLD_RM)
t = t.replace(OLD_RM, NEW_RM)

# 6) dualbridge marker on the fresh prod boot
anchor = 'echo "prod fresh-boot sanity + admission + v123 posture + v125 opsall PASS"'
assert t.count(anchor) == 1, "sec5 anchor=%d" % t.count(anchor)
dbgate = ('DBP=$(docker exec lsv-test sh -c \'grep -c "llm-scaler v126 DUAL BRIDGE" '
          '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\' | '
          "tr -d '[:space:]')\n"
          '[ "$DBP" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: dualbridge marker '
          'missing on prod boot got=$DBP"; exit 1; }\n'
          'echo "prod fresh-boot sanity + admission + v123 posture + v125 opsall + '
          'v126 dualbridge PASS"')
t = t.replace(anchor, dbgate)

# 7) header note
hdr = ("# mk_ship_v1226.py note: v126 ship = v1225 chain + dualbridge surface\n"
       "# (rm patch_v126_dualbridge.py in lane-commit cleanup, DUAL BRIDGE marker\n"
       "# on the fresh prod boot; battery completion literal stays V1225 caps —\n"
       "# rename carryover in validate_v1226_run.sh; gates_v1226_lane.sh adds the\n"
       "# v126 marker/.so-sha/P29-absence gates).\n")
t = t.replace("#!/bin/bash\n", "#!/bin/bash\n" + hdr, 1)

with open(DST, "w") as f:
    f.write(t)

print("wrote", DST)
print("v1226 count:", t.count("v1226"), " leftover v1225 count:", t.count("v1225"))
print("v1.2.26 count:", t.count("v1.2.26"), " leftover v1.2.25 count:", t.count("v1.2.25"))
print("battery literal V1225_VALIDATION_COMPLETE:", t.count("V1225_VALIDATION_COMPLETE"))
print("dualbridge boot gate:", t.count("dualbridge marker missing on prod boot"))
print("rm patch_v126:", t.count("patch_v126_dualbridge.py"))
