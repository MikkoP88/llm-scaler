#!/bin/bash
# logkeep.sh — v127-audit: snapshot lane boot logs out of lsv-test every 2 min
# (containers are rm'd per lane; raw logs needed for D10 stall-timestamp
# correlation and post-hoc forensics). Exits when lsv-test is gone.
mkdir -p /root/build/bench127/logs
while true; do
  for t in df7_tq4 mtp4_fp8 mtp4_tq4; do
    docker cp "lsv-test:/root/b_${t}.log" "/root/build/bench127/logs/b_${t}.log" >/dev/null 2>&1
  done
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    # one final pass after teardown window, then stop
    sleep 5
    for t in df7_tq4 mtp4_fp8 mtp4_tq4; do
      docker cp "lsv-test:/root/b_${t}.log" "/root/build/bench127/logs/b_${t}.log" >/dev/null 2>&1
    done
    exit 0
  fi
  sleep 120
done
