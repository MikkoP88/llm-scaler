#!/bin/bash
# R1 standard-shape storm leg (wsc_pressure leg5 shape: 18 convs x 4 turns
# = 72 reqs) on the R1 sharded-view posture. Gate: ok=72/72, wall within or
# better than the spec-posture baseline band 865-1312 s (v1.2.27 spec N=3
# 1312/1095/865; v1.2.29 spec legs 1089/1029/1042). Terminal: R1_STORM_DONE.
cd /root/build/v130_stage
: > /root/build/p77/r1_storm.log
echo "=== r1_leg1 (leg5 shape) start $(date '+%F %T')" >> /root/build/p77/r1_storm.log
python3 -u wsc_pressure_v134_leg5.py > wsc_pressure_r1_leg1.out 2>&1
echo "=== r1_leg1 rc=$? end $(date '+%F %T')" >> /root/build/p77/r1_storm.log
echo "R1_STORM_DONE $(date '+%F %T')" >> /root/build/p77/r1_storm.log
