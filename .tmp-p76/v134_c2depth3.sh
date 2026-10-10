#!/bin/bash
# v134_c2depth3.sh — CRASHFIX-76 Stage C2 (pre-bake, live nospec posture):
# N=3 depth-shape storms (18 convs x 6 turns = 108 reqs, the 3/3-death
# shape with spec) on the NOSPEC engine. Class-kill gate: 3/3 clean
# (108/108 each, zero resets, zero harvests). Terminal: C2DEPTH3_DONE.
cd /root/build/v130_stage
: > /root/build/v134_c2depth3.log
for i in 1 2 3; do
  echo "=== C2DEPTH3 leg$i start $(date +%H:%M:%S)" >> /root/build/v134_c2depth3.log
  python3 -u wsc_pressure_v134_leg7_depth.py > wsc_pressure_v134_leg7_depth.out 2>&1
  echo "=== C2DEPTH3 leg$i rc=$? end $(date +%H:%M:%S)" >> /root/build/v134_c2depth3.log
  grep PRESSUREB_DONE wsc_pressure_v134_leg7_depth.out >> /root/build/v134_c2depth3.log
done
echo "C2DEPTH3_DONE $(date +%H:%M:%S)" >> /root/build/v134_c2depth3.log
