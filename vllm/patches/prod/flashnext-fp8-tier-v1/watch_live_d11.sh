#!/bin/bash
# watch_live_d11.sh — host-side 30s live watch for the d11 flash-next serve
# (fn8-d1 on HOST NETWORK, port 8321 direct). Appends one line per tick.
# Health curl keeps the docker exec form (works on any net mode).
LOG=/root/fn8_stage/watch_d11.log
SL=/root/fn8_stage/serve_d11.log
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
  N6P=$(grep -c 'FN8_06p' $SL 2>/dev/null || echo 0)
  NV63=$(grep -c 'TTFTFIX_ACTIVE\|INTERLEAVE_ACTIVE' $SL 2>/dev/null || echo 0)
  NPOI=$(grep -c 'POISONED-ROW' $SL 2>/dev/null || echo 0)
  NL=$(wc -l < $SL 2>/dev/null || echo 0)
  G0=$(xpu-smi dump --device 0 --metrics MEMORY --interval 1 --number 1 2>/dev/null | tail -1)
  G1=$(xpu-smi dump --device 1 --metrics MEMORY --interval 1 --number 1 2>/dev/null | tail -1)
  echo "$TS health=$HC procs=$NW posts=$NP http500=$N5 idx_oob=$NE tb=$NT ecore_fail=$NF preempt=$NPRE ple06p=$N6P v63v64=$NV63 poisoned=$NPOI loglines=$NL gpu0=$G0 gpu1=$G1" >> $LOG
  sleep 30
done
