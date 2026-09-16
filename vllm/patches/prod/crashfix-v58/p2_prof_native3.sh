#!/bin/bash
# p2_prof_native3.sh — native profile of Worker_TP0 with CONFIRMED active
# decode load (polls engine log for Running: N>0 before recording).
# usage: p2_prof_native3.sh <OUT.json> <LOADLABEL>
OUT="${1:?out}"
LBL="${2:?label}"
pkill -f /root/build/loadloop.sh 2>/dev/null
sleep 1
nohup /root/build/loadloop.sh >/dev/null 2>&1 < /dev/null &
echo "P2PROFN3 $LBL loadloop started; polling for active decode"
OK=""
for i in $(seq 1 24); do
  sleep 5
  R=$(docker exec lsv-test sh -c "grep -o 'Running: [0-9]*' /root/serve_full.log | tail -1")
  echo "  poll$i: ${R:-none}"
  case "$R" in *"Running: "[1-9]*) OK=1; break;; esac
done
if [ -z "$OK" ]; then
  echo "P2PROFN3 $LBL ABORT-NO-ACTIVE-LOAD"
  exit 1
fi
sleep 3
WPID=$(docker exec lsv-test sh -c 'pgrep -f Worker_TP0 | head -1')
echo "P2PROFN3 $LBL WPID=$WPID load-active, recording"
docker exec lsv-test /opt/venv/bin/py-spy record --native --pid "$WPID" \
  --duration 15 --rate 50 --format speedscope -o "$OUT" 2>&1 | tail -2
if ! python3 -c "import json,sys; d=json.load(open('$OUT')); sys.exit(0 if sum(len(p['samples']) for p in d['profiles'])>50 else 1)" 2>/dev/null; then
  echo "P2PROFN3 $LBL NATIVE-RECORD-EMPTY — dumping instead"
  docker exec lsv-test sh -c "for k in 1 2 3 4 5 6; do /opt/venv/bin/py-spy dump --native --pid $WPID; sleep 2; done" > "${OUT%.json}_dumps.txt" 2>&1
  tail -60 "${OUT%.json}_dumps.txt"
fi
echo "P2PROFN3 $LBL DONE"
