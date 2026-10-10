#!/bin/bash
# R1 restart diagnosis pt2: map container-namespace pids to cmdlines
# (setproctitle makes comm-pgrep blind; /proc cmdline scan is ground truth),
# and read the two 11:53 hf.py tracebacks that landed in the TERM window.
echo "=== named pids ==="
for p in 1 7 69 215 488; do
  docker exec lsv-test sh -c "if [ -d /proc/$p ]; then echo -n 'pid $p ALIVE: '; tr '\0' ' ' < /proc/$p/cmdline; echo; else echo 'pid $p GONE'; fi"
done
echo "=== full container proc map ==="
docker exec lsv-test sh -c 'for d in /proc/[0-9]*; do p=${d#/proc/}; printf "%s: " "$p"; tr "\0" " " < "$d/cmdline" 2>/dev/null | cut -c 1-120; echo; done' | sort -n
echo "=== traceback 1 (3300-3322) ==="
docker exec lsv-test sed -n '3300,3322p' /root/serve_full.log
echo "=== traceback 2 (3455-3477) ==="
docker exec lsv-test sed -n '3455,3477p' /root/serve_full.log
echo "=== requests around 11:5x ==="
docker exec lsv-test grep -n 'POST /v1' /root/serve_full.log | tail -n 8
