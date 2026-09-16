#!/bin/bash
# p2_prof_native.sh — capture py-spy speedscope WITH NATIVE FRAMES of Worker_TP0
# usage: p2_prof_native.sh <OUT.json> <LOADLABEL>
OUT="${1:?out}"
LBL="${2:?label}"
pgrep -f loadloop.sh >/dev/null || nohup /root/build/loadloop.sh >/dev/null 2>&1 < /dev/null &
sleep 6
WPID=$(docker exec lsv-test sh -c 'pgrep -f Worker_TP0 | head -1')
echo "P2PROFN $LBL WPID=$WPID"
docker exec lsv-test /opt/venv/bin/py-spy record --native --pid "$WPID" \
  --duration 15 --rate 50 --format speedscope -o "$OUT" 2>&1 | tail -2
echo "P2PROFN $LBL DONE"
