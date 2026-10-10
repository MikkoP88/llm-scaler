#!/bin/bash
# v134_b2std.sh — CRASHFIX-76 Stage B2b: STANDARD-shape storm (the parity
# client verbatim: 18 convs x 4 turns = 72 reqs, mixed deep/SMALL cohorts)
# against the NOSPEC posture. Measures what spec MTPx4 actually buys on
# the certified standard shape: wall vs parity band 1089/1029/1042 s
# (spec, v1.2.28) decides the P74b bake direction:
#   in-band  -> nospec-default posture = zero-degradation crash-free bake
#   slower   -> spec required on standard -> adaptive path / P74c close
# Terminal marker: B2STD_DONE.
cd /root/build/v130_stage
: > /root/build/v134_b2std.log
echo "=== B2STD standard storm (spec OFF) start $(date +%H:%M:%S)" >> /root/build/v134_b2std.log
python3 -u wsc_pressure_v134_leg1.py > wsc_pressure_v134_leg1.out 2>&1
echo "=== B2STD rc=$? end $(date +%H:%M:%S)" >> /root/build/v134_b2std.log
echo "B2STD_DONE $(date +%H:%M:%S)" >> /root/build/v134_b2std.log
