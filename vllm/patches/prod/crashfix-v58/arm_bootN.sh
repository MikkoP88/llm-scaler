#!/bin/bash
# arm_bootN.sh — Boot N certification arm: watch5 + sustain 2 + loop
# 24 + loop 48 + loop 48 (K2-class ~2.5-3 h soak incl. loop workload
# class, per M6 lesson that loops crash where sustain passes).
set -u
pkill -f 'wedge_watch5.sh bootN' 2>/dev/null
pkill -f 'repro_sustain' 2>/dev/null
pkill -f 'repro_loo[p]' 2>/dev/null
sleep 1
nohup bash /root/build/wedge_watch5.sh bootN 8000 > /root/build/lce1/bootN_watch.out 2>&1 < /dev/null &
nohup bash -c 'bash /root/build/repro_sustain.sh 2; python3 -u /root/build/repro_loop.py 24' > /root/build/lce1/bootN_pressure.out 2>&1 < /dev/null &
CHAIN=$!
echo "$CHAIN" > /root/build/lce1/bootN_chain.pid
nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 20; done; python3 -u /root/build/repro_loop.py 48" > /root/build/lce1/bootN_loop2.out 2>&1 < /dev/null &
CHAIN2_PID_FILE=/root/build/lce1/bootN_loop2.pid
nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 20; done; while pgrep -f 'repro_loop.py 48' >/dev/null 2>&1; do sleep 30; done; python3 -u /root/build/repro_loop.py 48" > /root/build/lce1/bootN_loop3.out 2>&1 < /dev/null &
echo "ARMED chain=$CHAIN $(date -u)"
