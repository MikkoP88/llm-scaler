#!/bin/bash
# armq8_helper.sh — arm Boot Q8 cleanly (BASE recount inside).
set -u
cd /root/build
R=$(dmesg | grep -ac 'Engine reset')
sed -i "s/^BASE=.*/BASE=$R/" poll_bootQ8.sh
echo "live_resets=$R"
nohup bash arm_bootQ8.sh > /root/build/lce1/bootQ8_arm.out 2>&1 &
sleep 3
tail -3 /root/build/lce1/bootQ8_arm.out
nohup bash poll_bootQ8.sh > /root/build/lce1/bootQ8_poll.out 2>&1 < /dev/null &
echo "ARMED_AND_POLLED $(date -u +%H:%M:%S)"
