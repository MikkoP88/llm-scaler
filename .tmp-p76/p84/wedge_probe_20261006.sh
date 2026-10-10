#!/bin/bash
# wedge_probe_20261006.sh — P84 live-watch stall capturer (evidence-only).
# The restored async lane (v1.2.31 + ALLOW_ASYNC=1 + ZOMBIE_STRIKES=3) shows
# intermittent 2-6s /health stalls at engine idle (000 on 5s-timeout probes,
# /metrics unaffected, full recovery between windows). This probe loops a
# 0.5s health poll; on TWO consecutive non-200 it freeze-frames py-spy all
# engine processes + ps/dmesg/serve-tail into the bank dir and exits.
# It NEVER kills, restarts, or touches the lane.
DIR=/root/build/lce1/WD_wedge_live_20261006_220608
LPID=208; EC=481; W0=680; W1=686
stall=0
for i in $(seq 1 720); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://localhost:8000/health)
  if [ "$code" != "200" ]; then
    stall=$((stall+1))
    echo "$(date -u +%H:%M:%S) stall=$stall code=$code" >> $DIR/probe.log
    if [ "$stall" -ge 2 ]; then
      for p in $LPID $EC $W0 $W1; do
        timeout 25 docker exec lsv-test /opt/venv/bin/py-spy dump --pid $p > $DIR/pyspy_$p.txt 2>&1
      done
      docker exec lsv-test sh -c 'ps -eo pid,ppid,stat,etime,args | grep -E "VLLM|vllm serve" | grep -v grep' > $DIR/ps.txt 2>&1
      dmesg -T | tail -n 20 > $DIR/dmesg.txt 2>&1
      docker exec lsv-test sh -c 'tail -n 200 /root/serve_full.log' > $DIR/serve_tail.txt 2>&1
      xpu-smi dump --device 0 --metrics MEMORY --interval 1 --number 1 > $DIR/xpusmi0.txt 2>&1
      xpu-smi dump --device 1 --metrics MEMORY --interval 1 --number 1 > $DIR/xpusmi1.txt 2>&1
      echo "$(date -u +%H:%M:%S) CAPTURED" >> $DIR/probe.log
      exit 0
    fi
  else
    stall=0
  fi
  sleep 0.5
done
echo "$(date -u +%H:%M:%S) no-stall in ~6min window" >> $DIR/probe.log
