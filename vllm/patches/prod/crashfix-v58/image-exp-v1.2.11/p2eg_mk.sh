#!/bin/bash
# p2eg_mk.sh — create repro_bootP2EG.sh = repro_bootNV.sh + __pycache__ purge
# injected before serve start (isolates the one bake step overlay lanes lack).
sed '/docker cp \/root\/build\/serve_user.sh lsv-test/i docker exec lsv-test find /opt/venv/lib/python3.12/site-packages/vllm -name __pycache__ -type d -prune -exec rm -rf {} +' /root/build/repro_bootNV.sh > /root/build/repro_bootP2EG.sh
chmod +x /root/build/repro_bootP2EG.sh
grep -n 'pycache\|serve_user.sh lsv-test' /root/build/repro_bootP2EG.sh
