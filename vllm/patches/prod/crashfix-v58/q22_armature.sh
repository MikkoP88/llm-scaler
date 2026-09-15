#!/bin/bash
# q22_armature.sh — Q22a (NEO 26.27) battery certification: §20 F.
# 18-round repro_sustain (~3 h, PASS bar = >=3 battery-hours 0
# resets vs observed MTTF 14m-2h52m) + p1_coh_watch 200 min +
# auto-arm watcher (q22_watch.sh) + mid-load boundary sweeps at
# ~30 and ~90 min + post census.
cd /root/build
nohup bash q22_watch.sh > lce1/q22_watch.out 2>&1 < /dev/null &
echo "watcher armed $(date +%H:%M:%S)"
nohup bash repro_sustain.sh 18 > lce1/sustain_Q22.out 2>&1 < /dev/null &
echo "sustain(18) launched $(date +%H:%M:%S)"
sleep 90
nohup bash p1_coh_watch.sh 200 > lce1/p1_coh_Q22.out 2>&1 < /dev/null &
echo "coh_watch launched $(date +%H:%M:%S)"
sleep 1740
echo "=== mid-load boundary sweep 1 (~30 min) ($(date +%H:%M:%S))"
python3 -u p1_len5probe.py 2>&1 | tee lce1/p1_len5probe_Q22_load1.out
sleep 3600
echo "=== mid-load boundary sweep 2 (~90 min) ($(date +%H:%M:%S))"
python3 -u p1_len5probe.py 2>&1 | tee lce1/p1_len5probe_Q22_load2.out
wait
echo "=== battery chain done ($(date +%H:%M:%S))"
echo "--- resets since arm (dmesg):"
dmesg -T | grep -E "reset|Reset" | tail -6
echo "--- stall/fence census:"
docker exec lsv-test sh -c 'grep -c "ASYNC-EVENT-STALL" /root/serve_full.log' || echo 0
echo "--- health:"
curl -s -o /dev/null -w "%{http_code}\n" -m 4 http://localhost:8000/health
echo "--- guard fires total:"
docker exec lsv-test sh -c 'grep -c "v58 P1 uniform-decode prefill guard" /root/serve_full.log'
echo "--- container:"
docker ps --format "{{.Names}} {{.Status}}" | head -2
echo Q22_ARMATURE_DONE
