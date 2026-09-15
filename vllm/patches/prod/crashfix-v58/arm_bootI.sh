#!/bin/bash
# arm_bootI.sh — arm watch5 + pressure chain + wave2 extension on Boot I.
set -u
pkill -f 'wedge_watch5.sh bootI' 2>/dev/null
pkill -f 'repro_sustain' 2>/dev/null
pkill -f 'repro_loo[p]' 2>/dev/null
sleep 1
nohup bash /root/build/wedge_watch5.sh bootI 8000 > /root/build/lce1/bootI_watch.out 2>&1 < /dev/null &
nohup bash -c 'bash /root/build/repro_sustain.sh 4; python3 -u /root/build/repro_loop.py 24' > /root/build/lce1/bootI_pressure.out 2>&1 < /dev/null &
CHAIN=$!
echo "$CHAIN" > /root/build/lce1/bootI_chain.pid
nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 20; done; python3 -u /root/build/repro_loop.py 48" > /root/build/lce1/bootI_loop2.out 2>&1 < /dev/null &
echo "ARMED chain=$CHAIN $(date -u)"
