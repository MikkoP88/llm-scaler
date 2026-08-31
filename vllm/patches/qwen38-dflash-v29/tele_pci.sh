#!/bin/bash
# llm-scaler v29 P1b: full PCI enumeration for the GPU BDFs — all functions,
# classes, drivers, link state. Resolves whether the Gen1x1 sysfs links belong
# to management functions vs the compute path.
OUT=/root/build/tele
mkdir -p $OUT
{
echo "===== all functions under 03: / b1: / da:"
for d in /sys/bus/pci/devices/0000:03:* /sys/bus/pci/devices/0000:b1:* /sys/bus/pci/devices/0000:da:*; do
  [ -d "$d" ] || continue
  echo "--- $(basename $d)"
  echo "  class=$(cat $d/class) vendor=$(cat $d/vendor) device=$(cat $d/device)"
  echo "  driver=$(basename $(readlink $d/driver 2>/dev/null) 2>/dev/null)"
  echo "  numa_node=$(cat $d/numa_node 2>/dev/null)"
  echo "  link: speed=$(cat $d/current_link_speed 2>/dev/null) width=$(cat $d/current_link_width 2>/dev/null)"
  echo "  max:   speed=$(cat $d/max_link_speed 2>/dev/null) width=$(cat $d/max_link_width 2>/dev/null)"
  echo "  bars: $(ls $d/resource* 2>/dev/null | wc -l) msi=$(cat $d/msi_bus 2>/dev/null)"
done
echo "===== accel/processing accelerators class 0x12 (XPUs often hide here)"
grep -l '^0x12' /sys/bus/pci/devices/*/class 2>/dev/null
echo "===== drm card -> device mapping"
for c in /sys/class/drm/card*; do
  p=$(readlink -f $c/device 2>/dev/null)
  echo "$c -> $p class=$(cat $p/class 2>/dev/null)"
done
echo "===== iommu groups of GPU bdfs"
for g in /sys/kernel/iommu_groups/*; do
  for dev in $g/devices/*; do
    b=$(basename $dev)
    case "$b" in
      0000:03:*|0000:b1:*|0000:da:*) echo "iommu_group $(basename $g): $b";;
    esac
  done
done
echo "===== DMAR / intel-iommu enabled?"
dmesg 2>/dev/null | grep -i -E 'iommu.*(enabled|disabled|mapping)|Intel-IOMMU' | head -8
journalctl -k -b 2>/dev/null | grep -i -E 'iommu.*(enabled|disabled)|DMAR:.*(IOMMU|context)' | head -8
echo "===== smon/p2p state if exposed"
ls /sys/module/dmabuf* /sys/module/i915/parameters/ 2>/dev/null | head -20
cat /sys/module/i915/parameters/enable_p2p 2>/dev/null || true
} > $OUT/pci.txt 2>&1
echo "written $OUT/pci.txt"
