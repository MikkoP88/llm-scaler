#!/bin/bash
# v88a — PRE-FIX kernel repro: replay the captured faulting GDN prefill call
# (v86 ring, worker 548) against a standalone serve-geometry pool on the
# CURRENT (buggy) wheel. Expect: LO ok, HI(4173) device fault or misdirected
# write, exit 1. Engine is killed first (a fault would take it down anyway).
set -u
TS=$(date +%H%M%S)
L=/root/build/lce1/v88a_$TS.out
OD=/root/build/lce1/v88repro
echo "v88a pre-fix repro start $TS" | tee -a $L

mkdir -p $OD

docker start lsv-test >/dev/null 2>&1 || true
sleep 3

# inputs: repro script + the captured fault call
docker cp /root/build/repro_v88_int32.py lsv-test:/root/repro_v88_int32.py
docker cp /root/build/lce1/v86cap/gdn_capture_last_548.pt lsv-test:/root/gdn_cap_input.pt

# engine down so the repro owns the GPUs (a DEVICE_LOST would kill it anyway)
docker exec lsv-test pkill -f 'vllm serv[e]' || true
sleep 5

# pre-fix run (buggy wheel): expect exit 1 with HI fault/misdirect
docker exec lsv-test sh -c 'cd /root && /opt/venv/bin/python repro_v88_int32.py /root/gdn_cap_input.pt /root/v88out; echo REPRO_EXIT=$?' 2>&1 | tee -a $L

# preserve the in-range reference outputs for the post-fix bit-compare
docker cp lsv-test:/root/v88out/ref100.pt $OD/ref100.prefix.pt >/dev/null 2>&1 \
  && echo "saved ref100.prefix.pt" | tee -a $L || echo "no ref100 saved (LO failed?)" | tee -a $L

echo V88A_DONE | tee -a $L
