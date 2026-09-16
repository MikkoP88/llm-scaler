#!/bin/bash
# p2_prof.sh — capture py-spy speedscope of Worker_TP0 under load
# usage: p2_prof.sh <OUT.json> <LOADLABEL>
OUT="${1:?out}"
LBL="${2:?label}"
pgrep -f loadloop.sh >/dev/null || nohup /root/build/loadloop.sh >/dev/null 2>&1 < /dev/null &
sleep 6
WPID=$(docker exec lsv-test sh -c 'pgrep -f Worker_TP0 | head -1')
echo "P2PROF $LBL WPID=$WPID"
docker exec lsv-test /opt/venv/bin/py-spy record --pid "$WPID" \
  --duration 25 --rate 50 --format speedscope -o "$OUT" 2>&1 | tail -1
echo "P2PROF $LBL DONE"
