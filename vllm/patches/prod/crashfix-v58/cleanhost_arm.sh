#!/bin/bash
# cleanhost_arm.sh — after HEALTH_OK on the CLEANHOST image lane: arm the q33
# crash watcher + launch the 18-round sustain battery (clean-host control).
cd /root/build
grep -E 'HEALTH_OK|BAKE-MARKER|BOOT_|live_resets' /root/build/lce1/boot_EXPV1211C.out | tail -6
code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health)
echo "health=$code resets=$(dmesg | grep -ac 'Engine reset')"
if [ "$code" != "200" ]; then echo NOT-HEALTHY-YET; exit 1; fi
nohup bash /root/build/q33_watch.sh > /root/build/lce1/expv1211c_watch.out 2>&1 &
echo "WATCHER-ARMED $(date +%H:%M:%S) pid=$!"
nohup bash /root/build/repro_sustain.sh 18 > /root/build/lce1/sustain_EXPV1211C.out 2>&1 &
echo "SUSTAIN-LAUNCHED-18 $(date +%H:%M:%S) pid=$!"
sleep 5
head -2 /root/build/lce1/expv1211c_watch.out
head -2 /root/build/lce1/sustain_EXPV1211C.out
