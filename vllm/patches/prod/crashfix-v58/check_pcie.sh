#!/bin/bash
# check_pcie.sh — parse pcie link gen/width + clocks from live xpu-smi dumps
# and compare with historical (passing) nv_screen clk dumps.
row () {  # row <device>
  xpu-smi dump --device "$1" --metrics all --number 1 2>/dev/null | tail -1
}
col () {  # col <row> <header-name>  (both from same dump)
  awk -F', ' -v h="$2" 'NR==1{for(i=1;i<=NF;i++) if($i==h) c=i} NR==2{print $c}' "$1" 2>/dev/null
}
for d in 0 1; do
  tmp=/tmp/pcie_dev$d.csv
  xpu-smi dump --device $d --metrics all --number 1 2>/dev/null > $tmp
  echo "dev$d: gen=$(col $tmp 'pcie.link.gen.current')/$(col $tmp 'pcie.link.gen.max') width=$(col $tmp 'pcie.link.width.current') clk=$(col $tmp 'clocks.current.graphics (MHz)') throttle=$(col $tmp 'clocks.throttle.reason') temp=$(col $tmp 'temperature.gpu (C)') power=$(col $tmp 'power.draw (W)')"
done
echo "== historical clk dumps present =="
ls -t /root/build/lce1/nv_*_clk_*.txt 2>/dev/null | head -8
echo "== historical samples (gen.current field) =="
for f in $(ls -t /root/build/lce1/nv_*_clk_*.txt 2>/dev/null | head -6); do
  awk -F', ' 'NR==1{for(i=1;i<=NF;i++) if($i=="pcie.link.gen.current") c=i} NR==2{print FILENAME": gen="$c}' "$f" 2>/dev/null
done
