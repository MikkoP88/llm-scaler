#!/bin/bash
# teardown_k2_bootM.sh — after K2 wave2 completes: kill K2 pressure/watch/
# mem-logger, remove container, launch Boot M (user EXACT lane e5m2 @262144
# on baked v1.2.10, pidfd, v58diag, py-spy) + mem_growth_logger from t=0.
set -u
pkill -f 'repro_loo[p]' 2>/dev/null
pkill -f 'repro_sustai[n]' 2>/dev/null
pkill -f 'wedge_watch[5]' 2>/dev/null
pkill -f 'mem_growth_logge[r]' 2>/dev/null
sleep 2
docker rm -f lsv-test >/dev/null 2>&1
echo "K2 torn down $(date -u)"

nohup bash /root/build/repro_bootM.sh A > /root/build/lce1/bootM.log 2>&1 < /dev/null &
echo "bootM launcher pid $! $(date -u)"

nohup bash /root/build/mem_growth_logger.sh /root/build/lce1/bootM_memgrowth.log > /dev/null 2>&1 < /dev/null &
echo "mem logger from t0 pid $! $(date -u)"
echo TEARDOWN_BOOTM_LAUNCHED
