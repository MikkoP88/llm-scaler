#!/bin/bash
# watchdog_rewire_v1227.sh — v127 WS-A/phase A3: repoint lane-watchdog
# relaunch from the v126 chain (repro_bootV1226_prod.sh) to the v127 swift
# boot chain (boot_v1227_restore.sh) and wire the Layer-4 drift guard into
# the health loop (every 10 cycles beside the alive marker). Idempotent.
# Engine is untouched: only the watchdog bash loop is stopped/started.
set -e
cd /root/build
systemctl stop lane-watchdog
cp -n lane_watchdog.sh lane_watchdog.sh.pre_v1227
sed -i 's|lineage=repro_bootV1221.sh|lineage=boot_v1227_restore.sh (v127 swift)|' lane_watchdog.sh
sed -i 's|nohup bash repro_bootV1226_prod.sh WD_|nohup bash /root/build/v127_stage/boot_v1227_restore.sh WD_|' lane_watchdog.sh
# drift call beside the every-10-cycle alive marker in the code=200 branch
grep -q watchdog_v2_drift_check lane_watchdog.sh || \
  sed -i '/alive armed code=200 cycles=/a\    [ $((CYCLES % 10)) -eq 0 ] \&\& bash /root/build/v127_stage/watchdog_v2_drift_check.sh >> $LOG 2>\&1' lane_watchdog.sh
systemctl start lane-watchdog
sleep 2
echo "active=$(systemctl is-active lane-watchdog)"
tail -2 lane_watchdog.log
grep -n 'boot_v1227_restore\|drift_check' lane_watchdog.sh
echo WATCHDOG_REWIRED_V1227
