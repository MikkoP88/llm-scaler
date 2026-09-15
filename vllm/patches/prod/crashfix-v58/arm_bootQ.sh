#!/bin/bash
# arm_bootQ.sh — Boot Q diag arm: watch5 + sustain 2 + loop 24 +
# wave2 48 + wave3 48 (same workload chain as Boot N; crash expected
# early — chain tail is just insurance).
set -u
pkill -f 'wedge_watch5.sh bootQ' 2>/dev/null
pkill -f 'repro_sustain' 2>/dev/null
pkill -f 'repro_loo[p]' 2>/dev/null
sleep 1
nohup bash /root/build/wedge_watch5.sh bootQ 8000 > /root/build/lce1/bootQ_watch.out 2>&1 < /dev/null &
nohup bash -c 'bash /root/build/repro_sustain.sh 2; python3 -u /root/build/repro_loop.py 24' > /root/build/lce1/bootQ_pressure.out 2>&1 < /dev/null &
CHAIN=$!
echo "$CHAIN" > /root/build/lce1/bootQ_chain.pid
nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 20; done; python3 -u /root/build/repro_loop.py 48" > /root/build/lce1/bootQ_loop2.out 2>&1 < /dev/null &
nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 20; done; while pgrep -f 'repro_loop.py 48' >/dev/null 2>&1; do sleep 30; done; python3 -u /root/build/repro_loop.py 48" > /root/build/lce1/bootQ_loop3.out 2>&1 < /dev/null &
echo "ARMED chain=$CHAIN $(date -u)"
