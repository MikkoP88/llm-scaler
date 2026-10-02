#!/bin/bash
# v134_bleg1.sh — CRASHFIX-76 Stage A4 leg B1: depth-shape storm against the
# ESIMD-full-off posture (boot_v1227_bleg.sh). Client = leg7_depth verbatim
# (108 reqs, N_TURNS=6, the 2/2 death shape). Terminal marker: BLEG1_DONE.
cd /root/build/v130_stage
: > /root/build/v134_bleg1.log
echo "=== BLEG1 depth storm (ESIMD full-off) start $(date +%H:%M:%S)" >> /root/build/v134_bleg1.log
python3 -u wsc_pressure_v134_leg7_depth.py > wsc_pressure_v134_leg7_depth.out 2>&1
echo "=== BLEG1 rc=$? end $(date +%H:%M:%S)" >> /root/build/v134_bleg1.log
echo "BLEG1_DONE $(date +%H:%M:%S)" >> /root/build/v134_bleg1.log
