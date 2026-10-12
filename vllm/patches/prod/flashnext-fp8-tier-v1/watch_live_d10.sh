#!/bin/bash
# watch_live_d10.sh — live watch for fn8-d1 d10 (flash-next 125B FP8
# live testing, port 8321).  Runs ON THE HOST; serve log reachable via
# the /root/fn8_stage:/stage bind mount.  Appends one status line every
# 30s to /root/fn8_stage/watch_d10.log.  Kill with: pkill -f watch_live_d10
LOG=/root/fn8_stage/watch_d10.log
SL=/root/fn8_stage/serve_d10.log
echo "=== WATCH_D10 start $(date) ===" >> $LOG
while true; do
  TS=$(date +%H:%M:%S)
  HC=$(docker exec fn8-d1 curl -s -o /dev/null -w %{http_code} --max-time 5 http://127.0.0.1:8321/health 2>/dev/null || echo 000)
  NW=$(docker exec fn8-d1 sh -c 'pgrep -c -f VLLM::' 2>/dev/null || echo 0)
  NP=$(grep -c 'POST /v1' $SL 2>/dev/null || echo 0)
  N5=$(grep -c 'HTTP/1.1" 500' $SL 2>/dev/null || echo 0)
  NE=$(grep -c 'Indexing.h' $SL 2>/dev/null || echo 0)
  NT=$(grep -c 'Traceback' $SL 2>/dev/null || echo 0)
  NF=$(grep -c 'EngineCore failed' $SL 2>/dev/null || echo 0)
  NPRE=$(grep -c 'preempt' $SL 2>/dev/null || echo 0)
  NL=$(wc -l < $SL 2>/dev/null || echo 0)
  G0=$(xpu-smi dump --device 0 --metrics MEMORY --interval 1 --number 1 2>/dev/null | tail -1)
  G1=$(xpu-smi dump --device 1 --metrics MEMORY --interval 1 --number 1 2>/dev/null | tail -1)
  echo "$TS health=$HC procs=$NW posts=$NP http500=$N5 idx_oob=$NE tb=$NT ecore_fail=$NF preempt=$NPRE loglines=$NL gpu0=$G0 gpu1=$G1" >> $LOG
  sleep 30
done
