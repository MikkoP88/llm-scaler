#!/bin/bash
# check_clocks.sh — inspect nv_screen clk dumps + live xpu-smi clock state
echo "== nvscreen clock-guard lines =="
grep -a -E 'clk|clock|Clock|2800|950' /root/build/lce1/nvscreen_RESTORE26R14.out | head -12
echo "== one raw clk dump (format learn) =="
head -25 "$(ls /root/build/lce1/nv_RESTORE26R14_clk_*.txt | head -1)"
echo "== LIVE dev0 =="
xpu-smi dump --device 0 --metrics all --number 1 2>/dev/null | grep -a -E 'Frequency|Clock|Utilization|Power|Temperature|throttle|Throttle' | head -12
echo "== LIVE dev1 =="
xpu-smi dump --device 1 --metrics all --number 1 2>/dev/null | grep -a -E 'Frequency|Clock|Utilization|Power|Temperature|throttle|Throttle' | head -12
