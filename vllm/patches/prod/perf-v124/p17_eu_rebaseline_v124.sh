#!/bin/bash
# p17_eu_rebaseline_v124.sh — P17: xpu-smi EU re-baseline under ASYNC load.
# The 16%-active / 57%-stall / 287-305 GB/s verdict was measured on the SYNC
# lane; this re-measures on the async (v1.2.23) lane under sustained 4x1024
# load -> refreshed evidence for the T3-2 kernel spec (P19).
# 120 s of sampling @ 2 s interval, both GPUs, sustained bench loop underneath.
set -u
L=/root/build/lce1/p17_eu_rebaseline.log
{
echo "=== P17 EU RE-BASELINE under async load start $(date +%T) ==="
curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health || { echo ABORT_NOT_UP; exit 1; }

( cd /root/build; for i in $(seq 1 16); do python3 bench_genspeed.py 4 1024 1; sleep 1; done ) \
  > /root/build/lce1/p17_load.log 2>&1 &
LOADPID=$!
sleep 3
# NB: this host's xpu-smi rejects --device -1 for dump ("Device handle not
# found"); sample each device explicitly, in parallel, then merge CSVs.
xpu-smi dump --device 0 --metrics EU_ARRAY,MEMORY --interval 2 --number 60 \
  > /root/build/lce1/p17_xpu_d0.csv 2>&1 &
D0=$!
xpu-smi dump --device 1 --metrics EU_ARRAY,MEMORY --interval 2 --number 60 \
  > /root/build/lce1/p17_xpu_d1.csv 2>&1 &
D1=$!
wait $D0 $D1
( head -1 /root/build/lce1/p17_xpu_d0.csv
  cat /root/build/lce1/p17_xpu_d0.csv /root/build/lce1/p17_xpu_d1.csv \
    | grep -v '^Timestamp' ) > /root/build/lce1/p17_xpu_samples.csv
echo "raw_samples=$(wc -l < /root/build/lce1/p17_xpu_samples.csv)"
wait $LOADPID

echo "--- load results (last 4) ---"; grep GENSPEED /root/build/lce1/p17_load.log | tail -4
python3 - <<'PYEOF'
import csv, statistics
rows = []
with open('/root/build/lce1/p17_xpu_samples.csv') as f:
    for row in csv.DictReader(f):
        clean = {k.strip(): v for k, v in row.items() if k}
        if 'eu.active (%)' in clean:
            rows.append(clean)
def col(name):
    out = []
    for r in rows:
        try: out.append(float(r[name]))
        except (ValueError, TypeError, KeyError): pass
    return out
def stats(label, xs):
    if not xs: print(f"{label}: NO DATA"); return
    print(f"{label}: n={len(xs)} mean={statistics.mean(xs):.1f} p50={statistics.median(xs):.1f} max={max(xs):.1f}")
act, stall, idle = col('eu.active (%)'), col('eu.stall (%)'), col('eu.idle (%)')
rd = [v/1e6 for v in col('memory.read.bandwidth (kB/s)')]   # -> GB/s
wr = [v/1e6 for v in col('memory.write.bandwidth (kB/s)')]
print('--- ALL SAMPLES (both GPUs, incl. ramp gaps) ---')
stats('eu.active%', act); stats('eu.stall%', stall); stats('eu.idle%', idle)
stats('mem.read GB/s', rd); stats('mem.write GB/s', wr)
busy = [(a, s, i, r) for a, s, i, r in zip(act, stall, idle, rd) if a > 1]
if busy:
    print('--- BUSY SAMPLES (eu.active>1%) ---')
    stats('eu.active%', [b[0] for b in busy]); stats('eu.stall%', [b[1] for b in busy])
    stats('eu.idle%', [b[2] for b in busy]); stats('mem.read GB/s', [b[3] for b in busy])
if rd:
    p50 = statistics.median(rd)
    print(f"read vs ~573 GB/s peak: p50={p50:.0f} GB/s -> {100*p50/573:.0f}%")
PYEOF
echo "=== P17 DONE $(date +%T) ==="
} > "$L" 2>&1
tail -40 "$L"
echo P17_DONE
