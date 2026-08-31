#!/bin/bash
# llm-scaler v29 P1: passive platform telemetry (safe while serving).
# GPU<->GPU transport health: PCIe link state, NUMA, IOMMU, THP, governor,
# IRQ balance, xpu-smi static discovery. Output: /root/build/tele/platform.txt
OUT=/root/build/tele
mkdir -p $OUT
{
echo "===== date"; date
echo "===== xpu-smi discovery"
xpu-smi discovery 2>&1
echo "===== GPU BDFs via sysfs"
for d in /sys/bus/pci/devices/*; do
  cls=$(cat $d/class 2>/dev/null)
  case "$cls" in 0x028000|0x03*|0x12*) ;; esac
  if [ -e "$d/resource2" ] && echo "$cls" | grep -q '^0x03'; then
    echo "$d class=$cls numa=$(cat $d/numa_node 2>/dev/null) \
lnk=$(cat $d/current_link_speed 2>/dev/null)x$(cat $d/current_link_width 2>/dev/null) \
max=$(cat $d/max_link_speed 2>/dev/null)x$(cat $d/max_link_width 2>/dev/null)"
  fi
done
echo "===== all PCI display-class devices (alt)"
grep -l 0x03 /sys/bus/pci/devices/*/class 2>/dev/null | while read c; do
  d=$(dirname $c); echo "$d $(cat $c) numa=$(cat $d/numa_node 2>/dev/null)"
done
echo "===== CPU/NUMA"
lscpu | grep -E 'NUMA|Socket|Core|Model name|MHz' 2>/dev/null || cat /proc/cpuinfo | grep -m2 -E 'model name|MHz'
numactl --hardware 2>/dev/null | head -20 || echo "numactl not installed"
echo "===== IOMMU"
cat /proc/cmdline
dmesg 2>/dev/null | grep -i -m5 -E 'iommu|DMAR' || journalctl -k -b 2>/dev/null | grep -i -m5 -E 'iommu|DMAR'
ls /sys/class/iommu/ 2>/dev/null
for g in /sys/kernel/iommu_groups/*/devices/*; do
  echo "$g -> $(lspci 2>/dev/null)"; break
done 2>/dev/null
echo "===== THP"
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null
echo "===== governor"
for c in /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; do cat $c 2>/dev/null; done
echo "===== memory"
free -g
echo "===== numa meminfo"
numastat -m 2>/dev/null | head -15 || grep -E 'Node [0-9]' /proc/zoneinfo | head -0 || true
echo "===== worker processes affinity + cpuset"
for p in $(docker top lsv-test -eo pid,cmd 2>/dev/null | grep Worker_TP | awk '{print $1}'); do
  echo "pid $p affinity=$(taskset -cp $p 2>/dev/null | tail -1)"
  echo "  cpuset: $(cat /proc/$p/cpuset 2>/dev/null) mems=$(cat /proc/$p/status 2>/dev/null | grep -E 'Mems_allowed|Mems_allowed_list')"
done
echo "===== IRQ balance"
systemctl is-active irqbalance 2>/dev/null
echo "===== i915/drm devices"
ls -la /dev/dri/ 2>/dev/null
for c in /sys/class/drm/card*/device/uevent; do echo "$c: $(grep -E 'PCI_ID|PCI_SLOT_NAME' $c)"; done 2>/dev/null
echo "===== xpu-smi static"
xpu-smi static -d 0 2>&1 | head -30
xpu-smi static -d 1 2>&1 | head -30
echo "===== fabric/topology (if supported)"
xpu-smi topology -d 0 2>&1 | head -20
xpu-smi fabric -d 0 2>&1 | head -10
} > $OUT/platform.txt 2>&1
echo "written $OUT/platform.txt ($(wc -l < $OUT/platform.txt) lines)"
