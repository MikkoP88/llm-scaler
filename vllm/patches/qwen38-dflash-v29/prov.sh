#!/bin/bash
# llm-scaler v29 P5/P6/P7: standard provocation chain for serve arms.
# fox battery (6x128 @65k) -> long_exp battery (3x2048 @65k) -> canonical
# barrage (10x4096 short ctx). This is the A5-control sequence that wedged a
# fresh v29 boot in ~15 min. RC=7 from any stage = WEDGE.
# $1 = tag; output /root/build/prov_<tag>.out
TAG="${1:-x}"
LOG=/root/build/prov_$TAG.out
{
echo "=== provocation $TAG start $(date +%H:%M:%S)"
echo "--- stage 1: fox 6x128 @65k"
python3 /root/build/deep_repro_fox.py 6 128
echo "FOX_RC=$?"
echo "--- stage 2: long_exp 3x2048 @65k $(date +%H:%M:%S)"
python3 /root/build/long_exp.py 3 2048 65k
echo "LONGEXP_RC=$?"
echo "--- stage 3: canonical 10x4096 $(date +%H:%M:%S)"
python3 /root/build/wedge_repro.py 10
echo "CANON_RC=$?"
echo "=== provocation $TAG end $(date +%H:%M:%S)"
} > $LOG 2>&1
echo "prov $TAG -> $LOG"
