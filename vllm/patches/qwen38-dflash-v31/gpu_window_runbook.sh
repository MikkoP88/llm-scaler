#!/bin/bash
# v31 GPU window arm 1: graphs + spec (bypass) + GDN-split + k-clamp
# Boot -> assert effective config -> watcher + provocation pass 1.
# NOT run while prod serves. Restore with restore_prod.sh afterwards.
#
# RESULTS (2026-09-01 window, see README.md for the full matrix):
#   arm 1  (this script): 65k WEDGE persists (GDN split REDUNDANT — already
#           default-split via _attention_ops); canonical 10/10 @17.4.
#   arm 1b (+moe fp8_block split): WEDGE @1 chunk — NEGATIVE.
#   arm 1c2(+ALL esimd/gemm splits, needs VLLM_DISABLE_COMPILE_CACHE=1):
#           WEDGE @1 chunk — all custom kernels exonerated.
#   arm D  (TORCH_COMPILE_DISABLE=1, no bypass change): ALL CLEAN @65k,
#           canonical 10/10 @17.7 — #11 CONVICTED to inductor-compiled
#           pieces x spec. D-k4: #12 corruption persists (capture-level).
#   CERT   v31.1 default posture (no bypass): ALL CLEAN, canonical @16.4.
# Fix shipped: llm-scaler-vllm-adv:v31.1 (gate_v311.patch + Dockerfile.v31_1).
set -u
echo "== stop watcher + old container =="
pkill -f 'frw5.sh' 2>/dev/null; pkill -f 'prov.sh' 2>/dev/null
sleep 1
docker rm -f lsv-test >/dev/null 2>&1

echo "== boot: v31 + bypass + spec k4 (auto-clamped to 3) + gdn splits =="
bash /root/build/serve_boot_var.sh \
  '{"method":"mtp","num_speculative_tokens":4}' \
  'VLLM_XPU_ENABLE_XPU_GRAPH=1 VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1' \
  v31arm1 '' 512 llm-scaler-vllm-adv:v31

echo "== health poll =="
code=000
for i in $(seq 1 54); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then echo "HEALTH OK after ~$((i*10))s"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER DIED"; docker cp lsv-test:/root/v31arm1.log /tmp/v31arm1_boot.log >/dev/null 2>&1
    tail -40 /tmp/v31arm1_boot.log; exit 7
  fi
done
[ "$code" = "200" ] || { echo "HEALTH TIMEOUT"; docker cp lsv-test:/root/v31arm1.log /tmp/v31arm1_boot.log >/dev/null 2>&1; tail -25 /tmp/v31arm1_boot.log; exit 8; }

echo "== assert effective config (clamp + splits fired, gate bypassed) =="
docker exec lsv-test bash -c "grep -c 'num_speculative_tokens 4 -> 3' /root/v31arm1.log" | grep -qx 1 || echo "WARN: clamp marker missing"
docker exec lsv-test bash -c "grep -c 'GDN kernels split out' /root/v31arm1.log" | grep -qx 1 || echo "WARN: split marker missing"
docker exec lsv-test bash -c "grep -c 'Continuing with fully eager execution' /root/v31arm1.log" | grep -qx 0 || echo "WARN: v30 gate fired despite bypass"
echo "== mapped ccl libs (expect 2021.15 on v31 lineage) =="
for p in $(docker exec lsv-test bash -c 'pgrep -f vllm | head -6'); do
  docker exec lsv-test bash -c "cat /proc/$p/maps 2>/dev/null | grep -oE '[^ ]*libccl.so[^ ]*'" 2>/dev/null
done | sort | uniq -c

echo "== coherence probe (k3 reference behavior) =="
bash /root/build/coh_probe.sh v31arm1 || echo "COH PROBE NONSTANDARD"

echo "== watcher + provocation pass 1 (detached, absolute paths) =="
setsid nohup bash /root/build/frw5.sh once v31arm1 > /root/build/frw5_v31arm1.out 2>&1 < /dev/null &
setsid nohup bash /root/build/prov.sh v31arm1p1 > /dev/null 2>&1 < /dev/null &
sleep 4
pgrep -af 'frw5.sh|prov.sh'
echo ARM1STARTED
# after pass 1: check /root/build/prov_v31arm1p1.out (RC=7 = wedge);
# if clean -> run pass 2 on SAME boot: bash /root/build/prov.sh v31arm1p2
# arm 2 (#12 lane): reboot with VLLM_XPU_ALLOW_K4_CAPTURE=1 (+ bypass) and
#   coh_probe P1 distinct-count vs eager reference.
