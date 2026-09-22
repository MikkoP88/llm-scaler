#!/bin/bash
# v63 battery on patched lane (budget=2048): S1/L2/C4 + xgrammar + thinking + spec/async checks
R=/root/build/_v63_battery.txt
: > $R
B=http://127.0.0.1:8000
echo "== METRICS PRE ==" >> $R
curl -s -m 10 $B/metrics | grep -E 'spec_decode|num_preemptions|prefix_cache_hits_total' | tail -6 >> $R
echo "== V63 AB S1L2C4 ==" >> $R
cd /root/build && python3 v60_ab_probe.py >> $R 2>&1
echo "== XGRAMMAR x3 ==" >> $R
for i in 1 2 3; do python3 t2_xgrammar_probe.py >> $R 2>&1; done
echo "== THINKING A-D ==" >> $R
python3 t2_thinking_probe.py >> $R 2>&1
echo "== METRICS POST ==" >> $R
curl -s -m 10 $B/metrics | grep -E 'spec_decode|num_preemptions|prefix_cache_hits_total' | tail -6 >> $R
echo "== ASYNC CHECK ==" >> $R
docker exec lsv-test grep -c 'Asynchronous scheduling is enabled' /root/serve_full.log >> $R 2>&1
echo "== V63 ACTIVE CHECK ==" >> $R
docker exec lsv-test grep -m3 'V63_TTFTFIX_ACTIVE' /root/serve_full.log >> $R 2>&1
echo "BATTERY_V63_DONE" >> $R
