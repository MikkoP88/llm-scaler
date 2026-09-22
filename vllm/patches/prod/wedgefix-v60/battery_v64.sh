#!/bin/bash
# v64 battery on patched lane (K=2 interleave + budget=2048):
# S1/L2/C4 + xgrammar + thinking + spec/async checks + v64 markers
R=/root/build/_v64_battery.txt
: > $R
B=http://127.0.0.1:8000
echo "== METRICS PRE ==" >> $R
curl -s -m 10 $B/metrics | grep -E 'spec_decode|num_preemptions|prefix_cache_hits_total' | tail -6 >> $R
echo "== V64 AB S1L2C4 ==" >> $R
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
echo "== V64 ACTIVE CHECK ==" >> $R
docker exec lsv-test grep -m3 'V64_INTERLEAVE_ACTIVE' /root/serve_full.log >> $R 2>&1
echo "== V64 ENV ==" >> $R
docker exec lsv-test sh -c 'grep -E "^export VLLM_V6" /root/serve_user.sh || echo env-defaults' >> $R 2>&1
echo "BATTERY_V64_DONE" >> $R
