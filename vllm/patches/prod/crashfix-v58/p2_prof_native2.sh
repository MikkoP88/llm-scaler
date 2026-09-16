#!/bin/bash
# p2_prof_native2.sh — steady-state native profile of Worker_TP0:
# fresh loadloop, 120s warm-up (past cold q17 + prefill), then 15s record.
# usage: p2_prof_native2.sh <OUT.json> <LOADLABEL>
OUT="${1:?out}"
LBL="${2:?label}"
pkill -f /root/build/loadloop.sh 2>/dev/null
sleep 1
nohup /root/build/loadloop.sh >/dev/null 2>&1 < /dev/null &
echo "P2PROFN2 $LBL loadloop restarted, warming 120s"
sleep 120
WPID=$(docker exec lsv-test sh -c 'pgrep -f Worker_TP0 | head -1')
echo "P2PROFN2 $LBL WPID=$WPID recording"
docker exec lsv-test /opt/venv/bin/py-spy record --native --pid "$WPID" \
  --duration 15 --rate 50 --format speedscope -o "$OUT" 2>&1 | tail -2
echo "P2PROFN2 $LBL DONE"
