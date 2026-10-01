#!/usr/bin/env python3
"""mk_ship_v1227.py — build ship_v1227.sh from the certified ship_v1226.sh
(host-side transform).

v127 ship chain is structurally identical to v1226 (clean warm relaunch from
-raw -> lane-commit -> full gates on the committed image -> fresh-boot prod
verify -> CC battery through litellm :4000 -> watchdog repoint+re-arm), with
these v127 deltas:

  - battery completion literal: validate_v1227_run.sh (built by
    mk_validate_v1227.py from the v1226 script) inherits the V1225 caps
    literals (rename carryover through the whole lineage) — the ship
    precondition greps V1225_VALIDATION_COMPLETE against the v1227 log.
  - lane-commit REFUSES to commit an ARMED posture: /root/.v127_m0live and
    /root/.v127_scaledspace must both be absent on the warm lane before
    docker commit (a marker present at commit time would bake telemetry
    into the production image / the P43 wrong-answer mode respectively).
  - lane-commit cleanup also removes m0_live_collect_v127.py.
  - fresh-boot verification additionally asserts the v127 M0LIVE-COLLECTOR
    marker present AND dormant on the running prod container.

Transforms (ordered):
  1. blanket v1226 -> v1227 (names, logs, boot scripts, prompts, backups)
  2. blanket v1.2.26 -> v1.2.27 (image tags incl. the prod-boot sed)
  3. uppercase carries: repro_bootV1226 -> V1227, escaped sed tag, banners
  4. final images grep range 'v1.2.2[56]' -> 'v1.2.2[67]'
  5. rm list += m0_live_collect_v127.py; dormant-refusal gates pre-commit
  6. m0live fresh-boot gates; PASS echo extended
  7. header note
"""
SRC = "/root/build/ship_v1226.sh"
DST = "/root/build/ship_v1227.sh"

with open(SRC) as f:
    t = f.read()

# 1) + 2) blanket renames
n_v = t.count("v1226")
t = t.replace("v1226", "v1227")
n_img = t.count("v1.2.26")
t = t.replace("v1.2.26", "v1.2.27")

# 3) uppercase carries
n_boot = t.count("repro_bootV1226")
assert n_boot == 10, "repro_boot refs=%d" % n_boot
t = t.replace("repro_bootV1226", "repro_bootV1227")
n_sed = t.count("v1\\.2\\.26-raw")
assert n_sed == 1, "escaped sed pattern=%d" % n_sed
t = t.replace("v1\\.2\\.26-raw", "v1\\.2\\.27-raw")
n_ban = t.count("SHIP_V1226")
assert n_ban == 5, "ship banners=%d" % n_ban
t = t.replace("SHIP_V1226", "SHIP_V1227")

# 3b) battery completion literal stays V1225 caps (lineage carryover) —
#     only annotate it once
OLD_PRE = ("grep -q V1225_VALIDATION_COMPLETE /root/build/lce1/v1227_validate_run.log "
           "2>/dev/null || { echo \"SHIP ABORT: validation not complete\"; exit 1; }")
assert t.count(OLD_PRE) == 1, "precondition anchor=%d" % t.count(OLD_PRE)
NEW_PRE = ("# v127 note: validate_v1227_run.sh emits the V1225 caps literal "
           "(lineage rename carryover)\n" + OLD_PRE)
t = t.replace(OLD_PRE, NEW_PRE)

# 4) final images range grep
n_rng = t.count("'v1.2.2[56]'")
assert n_rng == 1, "images range anchor=%d" % n_rng
t = t.replace("'v1.2.2[56]'", "'v1.2.2[67]'")

# 5) lane-commit cleanup + dormant-refusal gates BEFORE docker commit
OLD_RM = ("/root/patch_v125_c7_fp8state_raise.py /root/patch_v126_dualbridge.py "
          "2>/dev/null")
assert t.count(OLD_RM) == 1, "rm anchor=%d" % t.count(OLD_RM)
t = t.replace(OLD_RM, OLD_RM.replace(
    "/root/patch_v126_dualbridge.py 2>/dev/null",
    "/root/patch_v126_dualbridge.py /root/m0_live_collect_v127.py 2>/dev/null"))

anchor = "docker commit lsv-test llm-scaler-exp:v1.2.27 || { echo \"SHIP ABORT: commit failed\"; exit 1; }"
assert t.count(anchor) == 1, "commit anchor=%d" % t.count(anchor)
guard = ('M0A=$(docker exec lsv-test sh -c \'test ! -e /root/.v127_m0live '
         "&& echo 1 || echo 0')\n"
         '[ "$M0A" = "1" ] || { echo "SHIP ABORT: lane has ARMED m0live marker — '
         'refuse to commit"; exit 1; }\n'
         'SSA=$(docker exec lsv-test sh -c \'test ! -e /root/.v127_scaledspace '
         "&& echo 1 || echo 0')\n"
         '[ "$SSA" = "1" ] || { echo "SHIP ABORT: lane has scaled-space marker — '
         'refuse to commit (P43)"; exit 1; }\n') + anchor
t = t.replace(anchor, guard)

# 6) m0live collector gates on the fresh prod boot
OLD_ECHO = 'echo "prod fresh-boot sanity + admission + v123 posture + v125 opsall + v126 dualbridge PASS"'
assert t.count(OLD_ECHO) == 1, "sec5 PASS echo anchor=%d" % t.count(OLD_ECHO)
m0gates = ('M0P=$(docker exec lsv-test sh -c \'grep -c "llm-scaler v127 M0LIVE-COLLECTOR" '
           '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\' | '
           "tr -d '[:space:]')\n"
           '[ "$M0P" -ge 1 ] 2>/dev/null || { echo "SHIP ABORT: m0live collector '
           'marker missing on prod boot got=$M0P"; exit 1; }\n'
           'M0D=$(docker exec lsv-test sh -c \'test ! -e /root/.v127_m0live '
           "&& echo 1 || echo 0')\n"
           '[ "$M0D" = "1" ] || { echo "SHIP ABORT: prod boot has ARMED m0live '
           'marker got=$M0D"; exit 1; }\n'
           'echo "prod fresh-boot sanity + admission + v123 posture + v125 opsall + '
           'v126 dualbridge + v127 m0live collector (dormant) PASS"')
t = t.replace(OLD_ECHO, m0gates)

# 7) header note
hdr = ("# mk_ship_v1227.py note: v127 ship = v1226 chain + dormant-collector\n"
       "# surface (pre-commit refusal of ARMED markers, m0live marker+dormant\n"
       "# gate on the fresh prod boot, rm m0_live_collect_v127.py in the\n"
       "# lane-commit cleanup; battery completion literal stays V1225 caps —\n"
       "# lineage rename carryover; gates_v1227_lane.sh adds the v127\n"
       "# presence/absence gates).\n")
t = t.replace("#!/bin/bash\n", "#!/bin/bash\n" + hdr, 1)

with open(DST, "w") as f:
    f.write(t)

print("wrote", DST)
print("v1227 count:", t.count("v1227"), " leftover v1226 count:", t.count("v1226"))
print("v1.2.27 count:", t.count("v1.2.27"), " leftover v1.2.26 count:", t.count("v1.2.26"))
print("battery literal V1225_VALIDATION_COMPLETE:", t.count("V1225_VALIDATION_COMPLETE"))
print("pre-commit refusal gates:", t.count("refuse to commit"))
print("m0live prod-boot gates:", t.count("m0live"))
