#!/bin/bash
# v134_b2std3.sh — CRASHFIX-76 Stage C1 (pre-bake, live posture): N=3
# standard-shape storms (72 reqs) on the NOSPEC engine. Banks the
# nospec-standard band that the v1.2.29 bake certifies against spec
# parity 1089/1029/1042 s. Terminal marker: B2STD3_DONE.
cd /root/build/v130_stage
: > /root/build/v134_b2std3.log
for i in 1 2 3; do
  echo "=== B2STD3 leg$i start $(date +%H:%M:%S)" >> /root/build/v134_b2std3.log
  python3 -u wsc_pressure_v134_leg1.py > wsc_pressure_v134_leg1.out 2>&1
  echo "=== B2STD3 leg$i rc=$? end $(date +%H:%M:%S)" >> /root/build/v134_b2std3.log
  grep PRESSUREB_DONE wsc_pressure_v134_leg1.out >> /root/build/v134_b2std3.log
done
echo "B2STD3_DONE $(date +%H:%M:%S)" >> /root/build/v134_b2std3.log
