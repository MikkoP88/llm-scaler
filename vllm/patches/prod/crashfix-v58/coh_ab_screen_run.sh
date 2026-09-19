#!/bin/bash
# coh_ab_screen_run.sh — §24 Q: copy-on-hit A/B on extended bench3 (+128k).
# Lane CTRL (G85NSMB4KX): PC ON, no COH (repro_bootBS64_E4M3_G85NS_MB4K.sh)
#   -> measures the 65k + 128k warm dip on bench3x for the record.
# Lane COH (G85NSMB4KCOHX): PC ON + patch_v24pc copy-on-hit -> dip should
#   vanish (warm == cold within noise) at ~zero TTFT cost (copy vs prefill).
# Protocol per lane: pause watchdog -> teardown -> boot -> HEALTH_OK gate ->
# knob census -> 15min boosted settle (I6 f8ref-WARM rule) -> bs64_screen_x
# (f8ref e4m3 cert 52f598e7d38a + q17 + bench3x cold/warm incl ctx128k).
cd /root/build
set -u

run_lane() {
  LBL="$1"; BOOT="$2"
  echo "[COHAB] ($LBL) pause watchdog + teardown $(date +%H:%M:%S)"
  touch /root/build/lane_watchdog.paused
  docker rm -f lsv-test >/dev/null 2>&1 || true
  sleep 10
  echo "[COHAB] ($LBL) boot $BOOT $(date +%H:%M:%S)"
  bash "$BOOT" "$LBL" > "lce1/boot_$LBL.out" 2>&1 || true
  if ! grep -q HEALTH_OK "lce1/boot_$LBL.out"; then
    echo "BOOT_FAIL [$LBL]"; tail -25 "lce1/boot_$LBL.out"; return 1
  fi
  grep -m1 HEALTH_OK "lce1/boot_$LBL.out"
  docker exec lsv-test sh -c "grep -oE -- '--block-size [0-9]+|--gpu-memory-utilization [0-9.]+|--max-num-batched-tokens [0-9]+' /root/serve_user.sh; echo pc_count=\$(grep -c 'enable-prefix-caching' /root/serve_user.sh); echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh)"
  if [ "$LBL" = "G85NSMB4KCOHX" ]; then
    grep -E 'V24PC_INSTALLED|patched|installed' lce1/v24pc_apply_bootCOH.log
    docker exec lsv-test grep -c "llm-scaler v24pc" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py
  fi
  echo "[COHAB] ($LBL) warm-settle ~15min boosted $(date +%H:%M:%S)"
  for i in 1 2 3 4 5 6 7; do
    curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
    sleep 120
  done
  echo "[COHAB] ($LBL) settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"
  bash bs64_screen_x.sh "$LBL" 52f598e7d38a
  echo "LANE_DONE $LBL $(date +%H:%M:%S)"
}

run_lane G85NSMB4KX repro_bootBS64_E4M3_G85NS_MB4K.sh || echo "LANE_FAIL G85NSMB4KX"
sleep 20
run_lane G85NSMB4KCOHX repro_bootBS64_E4M3_G85NS_MB4K_COH.sh || echo "LANE_FAIL G85NSMB4KCOHX"
echo "COH_AB_DONE $(date +%H:%M:%S)"
