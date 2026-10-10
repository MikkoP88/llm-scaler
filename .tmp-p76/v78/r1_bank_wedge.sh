#!/bin/bash
# Bank the R1 storm wedge artifacts to /root/build/lce1/WD_wedge_<HHMMSS>/
# (serve log, py-spy dumps, storm outputs, metrics snapshot) — BEFORE any teardown.
TS=$(date +%H%M%S)
D=/root/build/lce1/WD_wedge_$TS
mkdir -p "$D"

# full serve log out of the container (telemetry-fragility law)
docker cp lsv-test:/root/serve_full.log "$D/serve_full.log" 2>/dev/null && echo "serve_full.log banked"

# py-spy dumps of the hung workers + EngineCore (retry EC — py-spy exe quirk)
docker exec lsv-test sh -c '/opt/venv/bin/py-spy dump --pid 492' > "$D/pyspy_TP0_492.txt" 2>&1
docker exec lsv-test sh -c '/opt/venv/bin/py-spy dump --pid 498' > "$D/pyspy_TP1_498.txt" 2>&1
for p in 293 287 285; do
  docker exec lsv-test sh -c "/opt/venv/bin/py-spy dump --pid $p" > "$D/pyspy_EC_$p.txt" 2>&1
done
echo "py-spy banked: $(ls "$D" | grep pyspy | tr '\n' ' ')"

# storm outputs
cp -p /root/build/v130_stage/wsc_pressure_r1_leg1.out "$D/" 2>/dev/null
# split the contaminated jsonl at today's header (line 75): v134 original vs R1 rows
H=$(grep -n 'start 2026-10-03 12:47:32' /root/build/v130_stage/wsc_pressure_v134_leg5.jsonl | head -n 1 | cut -d: -f1)
if [ -n "$H" ]; then
  head -n $((H - 1)) /root/build/v130_stage/wsc_pressure_v134_leg5.jsonl > "$D/wsc_v134_leg5_original_tail.jsonl"
  tail -n +$H /root/build/v130_stage/wsc_pressure_v134_leg5.jsonl > "$D/wsc_r1_leg1_rows.jsonl"
  echo "jsonl split at line $H: v134=$((H - 1)) lines, r1=$(wc -l < "$D/wsc_r1_leg1_rows.jsonl") lines"
fi
cp -p /root/build/p77/r1_storm.log "$D/" 2>/dev/null

# metrics + engine state snapshot
curl -s --max-time 5 http://localhost:8000/metrics > "$D/metrics_wedge.txt" 2>&1
docker exec lsv-test sh -c 'tail -n 120 /root/serve_full.log' > "$D/serve_tail120.txt" 2>&1
dmesg -T | tail -n 60 > "$D/dmesg_tail.txt" 2>&1
docker exec lsv-test sh -c 'grep SPECTIMING /root/serve_full.log | tail -n 10' > "$D/spectiming_tail.txt" 2>&1

echo "=== banked at $D ==="
ls -la "$D"
md5sum "$D"/* 2>/dev/null
