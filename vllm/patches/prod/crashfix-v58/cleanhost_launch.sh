#!/bin/bash
# cleanhost_launch.sh — post-reboot clean-state verification + image-lane
# boot for the CLEAN-HOST control rerun (user directive: rule out that
# event #10 was caused by residue from earlier crashes).
echo "boot_since: $(uptime -s)"
echo "resets_now: $(dmesg | grep -ac 'Engine reset')"
ls /root/build/lane_watchdog.paused >/dev/null 2>&1 && echo WATCHDOG-PAUSED || echo WATCHDOG-FLAG-MISSING
echo "watchdog_unit: $(systemctl is-active lane-watchdog)"
docker ps -a --format '{{.Names}} {{.Status}}'
for i in $(seq 1 30); do
  docker info >/dev/null 2>&1 && { echo "DOCKER-READY (after $((i*10))s)"; break; }
  sleep 10
done
nohup bash /root/build/boot_exp2618.sh CLEANHOST > /root/build/lce1/boot_EXPV1211C.out 2>&1 &
echo "BOOT-LAUNCHED-CLEANHOST $(date +%H:%M:%S)"
