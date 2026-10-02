#!/bin/bash
# v134_b2nospec.sh — CRASHFIX-76 Stage B2: depth-shape storm against the
# nospec posture (boot_v1227_nospec.sh — MTP speculative decoding stripped,
# kernels stock). Client = leg7_depth verbatim (108 reqs, N_TURNS=6, the
# 2/2 death shape). Terminal marker: B2NOSPEC_DONE.
cd /root/build/v130_stage
: > /root/build/v134_b2nospec.log
echo "=== B2NOSPEC depth storm (spec OFF) start $(date +%H:%M:%S)" >> /root/build/v134_b2nospec.log
python3 -u wsc_pressure_v134_leg7_depth.py > wsc_pressure_v134_leg7_depth.out 2>&1
echo "=== B2NOSPEC rc=$? end $(date +%H:%M:%S)" >> /root/build/v134_b2nospec.log
echo "B2NOSPEC_DONE $(date +%H:%M:%S)" >> /root/build/v134_b2nospec.log
