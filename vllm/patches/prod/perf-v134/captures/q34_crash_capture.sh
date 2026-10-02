#!/bin/bash
# q34_crash_capture.sh — §18 capture protocol, v134 forensic upgrade
# (Q34): everything q22 saved (fr rings + f15b dumps FIRST and
# verified, devcoredump immediately, dmesg tail, first errors,
# dump_input section, worker ps, serve log) PLUS the perf-v134 P71
# gap-fills:
#   - fr ring TAILS (last 30 lines: the death-point op sequence)
#   - debugfs GPU client/context listings (which client held the
#     wedged context)
#   - xe module parameter snapshot (execlist/guc posture at crash)
#   - ALL dmesg Engine-reset + Timedout-job lines (every engine,
#     not just the tail-10 grep)
#   - journalctl kernel window (-15 min): pre-reset silence check
# devcoredump note: the sysfs data file is driver-capped (~500 KB,
# truncated mid-GuC-log on 1.1 MB logs) — the LRC/engine section is
# unobtainable; cp to EOF is already maximal.
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
  docker cp "lsv-test:$f" "$D/${b}.log" 2>/dev/null && echo "copied $b size=$(stat -c%s "$D/${b}.log")" || echo "no dump $b"
done
ls -la $D/

echo "--- STEP 2: devcoredump (TTL!):"
for c in $(ls /sys/class/drm/ | grep card); do
  if [ -d /sys/class/drm/$c/device/devcoredump ]; then
    cp /sys/class/drm/$c/device/devcoredump/data $D/devcoredump_${c}_Q22.bin 2>/dev/null && echo "saved $c size=$(stat -c%s $D/devcoredump_${c}_Q22.bin)"
    cat /sys/class/drm/$c/device/devcoredump/etime 2>/dev/null
  fi
done

echo "--- STEP 2b: fr ring TAILS (death-point sequences):"
for f in $D/fr_*.log; do
  [ -f "$f" ] && tail -30 "$f" > "$D/tail_$(basename $f)" && echo "tail $(basename $f)"
done

echo "--- STEP 2c: debugfs GPU client/context listings:"
for d in /sys/kernel/debug/dri/*/; do
  n=$(basename $d)
  cp "${d}clients" "$D/dri_clients_${n}" 2>/dev/null && echo "clients $n"
  cp "${d}internal_clients" "$D/dri_internal_${n}" 2>/dev/null
done

echo "--- STEP 2d: xe module parameter snapshot:"
: > $D/xe_params.txt
for p in $(ls /sys/module/xe/parameters/ 2>/dev/null); do
  v=$(cat /sys/module/xe/parameters/$p 2>/dev/null)
  echo "$p=$v" >> $D/xe_params.txt
done
echo "xe params banked ($(wc -l < $D/xe_params.txt) entries)"

echo "--- STEP 3: dmesg resets + ALL timedout-job lines:"
dmesg -T | grep -E "Engine reset|Timedout job|guc|reset queued|reset started|reset done|coredump" | tail -40 | tee $D/dmesg_tail.txt

echo "--- STEP 3b: journal kernel window (pre-reset silence check):"
S=$(date -d '-15 min' '+%Y-%m-%d %H:%M:%S')
journalctl -k --since "$S" --no-pager 2>/dev/null | grep -v audit | tail -120 > $D/journal_window.txt
echo "journal window $(wc -l < $D/journal_window.txt 2>/dev/null || echo 0) lines"

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
