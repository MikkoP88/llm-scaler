#!/bin/bash
# q22_crash_capture.sh — §18 capture protocol, GENERIC pid discovery
# (Q22+): fr rings + f15b dumps docker-cp'd FIRST and VERIFIED
# non-empty before anything else; devcoredump saved immediately (1h
# TTL). Only fresh rings (mtime -240min) — image bakes stale Sep-6
# fr files into /tmp.
cd /root/build
D=lce1/Q22_crash
mkdir -p $D
TS=$(date +%H%M%S)

echo "--- STEP 1: fr rings + f15b dumps FIRST (verified):"
FRS=$(docker exec lsv-test sh -c "find /tmp -maxdepth 1 -name 'fr_*.log' -mmin -240 2>/dev/null")
for f in $FRS; do
  b=$(basename $f .log)
  docker cp "lsv-test:$f" "$D/${b}.log" 2>"$D/${b}.cperr" && \
    s=$(stat -c%s "$D/${b}.log" 2>/dev/null || echo 0) && \
    echo "copied $b size=$s" || echo "MISSING $b: $(cat $D/${b}.cperr 2>/dev/null)"
done
DUMPS=$(docker exec lsv-test sh -c "find /root /tmp -maxdepth 1 -name 'f15b_dump_*.log' -mmin -240 2>/dev/null")
for f in $DUMPS; do
  b=$(basename $f .log)
  docker cp "lsv-test:$f" "$D/${b}.log" 2>/dev/null && echo "copied $b size=$(stat -c%s $D/${b}.log)" || echo "no dump $b"
done
ls -la $D/

echo "--- STEP 2: devcoredump (TTL!):"
for c in $(ls /sys/class/drm/ | grep card); do
  if [ -d /sys/class/drm/$c/device/devcoredump ]; then
    cp /sys/class/drm/$c/device/devcoredump/data $D/devcoredump_${c}_Q22.bin 2>/dev/null && echo "saved $c size=$(stat -c%s $D/devcoredump_${c}_Q22.bin)"
    cat /sys/class/drm/$c/device/devcoredump/etime 2>/dev/null
  fi
done

echo "--- STEP 3: dmesg resets:"
dmesg -T | grep -E "reset|Reset|dump" | tail -10 | tee $D/dmesg_tail.txt

echo "--- STEP 4: first errors in serve log:"
docker exec lsv-test sh -c "grep -n 'EngineDeadError\|EngineCore encountered\|ASYNC-EVENT-STALL\|sample_tokens' /root/serve_full.log | head -10" > $D/first_errors.txt 2>&1
cat $D/first_errors.txt | cut -c1-200

echo "--- STEP 5: dump_input / fatal section:"
docker exec lsv-test sh -c "grep -n 'Dumping input data\|Dumping scheduler output\|fatal error' /root/serve_full.log | head -5" > $D/dump_lines.txt 2>&1
cat $D/dump_lines.txt
L=$(head -1 $D/dump_lines.txt | cut -d: -f1)
if [ -n "$L" ] && [ "$L" -gt 2 ] 2>/dev/null; then
  docker exec lsv-test sh -c "sed -n \"$((L-2)),$((L+8))p\" /root/serve_full.log" > $D/dump_section.txt 2>&1
  head -8 $D/dump_section.txt | cut -c1-400
fi

echo "--- STEP 6: worker ps:"
docker exec lsv-test sh -c "ps -o pid,stat,pcpu,etime,comm -C python3" > $D/ps.txt 2>&1
cat $D/ps.txt

echo "--- STEP 7: full serve log copy:"
docker cp lsv-test:/root/serve_full.log $D/serve_full_Q22.log && echo "serve log size=$(stat -c%s $D/serve_full_Q22.log)"

echo "--- STEP 8: teardown:"
docker exec lsv-test pkill -9 -f Worker_TP 2>/dev/null || true
sleep 2
docker rm -f lsv-test
ls -la $D/
echo "Q22_CAPTURE_AND_TEARDOWN_DONE $(date +%H:%M:%S)"
