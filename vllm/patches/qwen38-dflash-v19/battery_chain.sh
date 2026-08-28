#!/bin/bash
# battery_chain.sh - sequential v17 battery arms (resumable).
#   START_FROM=3 ./battery_chain.sh   -> skip arms 1-2 (already done)
# Safety: battery_arm.sh aborts before relaunch if any engine reset exists
# this boot (#03 protocol); chain stops on first failure for manual triage
# (deep-probe xe faults are the known random family - reboot, resume).
set -u
LOG=/root/telemetry/battery.log
ts() { date '+%F %T'; }
ARMNO=0
skip() {  # returns 0 (true) when the stage should be skipped
  ARMNO=$((ARMNO+1))
  [ "$ARMNO" -lt "${START_FROM:-1}" ]
}

run_arm() { # name kv spec gmu
  echo "$(ts) CHAIN: arm $* starting" >> "$LOG"
  if /root/telemetry/battery_arm.sh "$@" >> "/root/telemetry/battery_$1.out" 2>&1; then
    echo "$(ts) CHAIN: arm $1 OK" >> "$LOG"
    return 0
  else
    rc=$?
    echo "$(ts) CHAIN: arm $1 FAILED rc=$rc - stopping chain for triage" >> "$LOG"
    exit "$rc"
  fi
}

deep() { # tag tokens
  echo "$(ts) CHAIN: deep probe $1 ($2 tok) starting" >> "$LOG"
  timeout 1800 python3 /root/deep_gen.py --tag "$1" --max-tokens "$2" --temperature 0.0 \
    >> "/root/bench/deep_$1.out" 2>&1
  rc=$?
  echo "$(ts) CHAIN: deep probe $1 rc=$rc : $(tail -1 /root/bench/deep_$1.out 2>/dev/null)" >> "$LOG"
  R=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); R=${R:-0}
  if [ "$R" -ne 0 ]; then
    echo "$(ts) CHAIN: $R engine resets after deep $1 - STOP (reboot, then START_FROM=<next>)" >> "$LOG"
    exit 3
  fi
}

longctx() { # tag  - 64k-token prompt + 1536 gen (KV-depth pressure, short exposure)
  echo "$(ts) CHAIN: longctx probe $1 starting" >> "$LOG"
  if [ -f /root/bench/prompt_64k.txt ]; then
    timeout 900 python3 /root/bench_gen.py --tag "$1" --model qwen3.8-27b-fp8 \
      --prompt-file /root/bench/prompt_64k.txt --max-tokens 1536 --depth-step 1536 \
      >> "/root/bench/longctx_$1.out" 2>&1
    echo "$(ts) CHAIN: longctx $1 : $(tail -1 /root/bench/longctx_$1.out 2>/dev/null)" >> "$LOG"
  else
    echo "$(ts) CHAIN: longctx $1 SKIPPED (no /root/bench/prompt_64k.txt)" >> "$LOG"
  fi
  R=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); R=${R:-0}
  [ "$R" -ne 0 ] && { echo "$(ts) CHAIN: resets after longctx $1 - STOP" >> "$LOG"; exit 3; }
}

# ---- arm 1 (chain): fp8 KV + dflash
if skip; then :; else
  run_arm v17a3_fp8_spec fp8 1 0.8
  longctx v17a3_fp8_longctx
fi

# ---- arm 2 (chain): fp8 KV target-only
if skip; then :; else
  run_arm v17a4_fp8_nospec fp8 0 0.8
  longctx v17a4_fp8_nospec_longctx
fi

# ---- arm 3 (chain): k8v4 + requested spec (expect #05b auto-disable warning)
if skip; then :; else
  run_arm v17a5_k8v4_spec turboquant_k8v4 1 0.8
  grep -aiE "turboquant.*(spec|disabled)|spec.*turboquant|auto-disable" /root/telemetry/serve_tel.log | head -2 >> "$LOG"
  longctx v17a5_k8v4_longctx
fi

# ---- arm 4 (chain): k8v4 target-only (+ one continuous deep5k: untested TQ decode)
# longctx DROPPED: a5 k8v4 longctx tripped the xe fault family at 60k depth
# (reset ~79s in, end-of-prefill; spec was auto-disabled = pure TQ target).
# deep5k retained: shallower TQ decode exposure, still untested.
if skip; then :; else
  run_arm v17a6_k8v4_nospec turboquant_k8v4 0 0.8
  deep v17a6_k8v4_deep5k 5000
fi

# ---- arm 5 (chain): 4bit_nc target-only (historically unusable; v17 warmup fix)
if skip; then :; else
  run_arm v17a7_tq4nc_nospec turboquant_4bit_nc 0 0.8
  longctx v17a7_tq4nc_longctx
fi

# ---- arm 6 (chain): memory-limited gmu 0.9 (default KV + spec)
if skip; then :; else
  run_arm v17a8_gmu09_spec "" 1 0.9
  longctx v17a8_gmu09_longctx
fi

# ---- arm 7 (chain): default recipe, temp sweep + concurrency
if skip; then :; else
  run_arm v17a9_default_spec "" 1 0.8
  longctx v17a9_default_longctx
  timeout 1800 python3 /root/deep_gen.py --tag v17a9_temp06_deep10k --max-tokens 10000 --temperature 0.6 >> /root/bench/deep_v17a9_temp06.out 2>&1
  echo "$(ts) CHAIN: temp0.6 deep10k: $(tail -1 /root/bench/deep_v17a9_temp06.out 2>/dev/null)" >> "$LOG"
  R=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); R=${R:-0}
  [ "$R" -ne 0 ] && { echo "$(ts) CHAIN: resets after temp0.6 - STOP" >> "$LOG"; exit 3; }
  timeout 1800 python3 /root/deep_gen.py --tag v17a9_temp10_deep10k --max-tokens 10000 --temperature 1.0 >> /root/bench/deep_v17a9_temp10.out 2>&1
  echo "$(ts) CHAIN: temp1.0 deep10k: $(tail -1 /root/bench/deep_v17a9_temp10.out 2>/dev/null)" >> "$LOG"
  R=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); R=${R:-0}
  [ "$R" -ne 0 ] && { echo "$(ts) CHAIN: resets after temp1.0 - STOP" >> "$LOG"; exit 3; }
  python3 /root/bench_gen.py --tag v17a9_conc2_512 --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 --concurrency 2 2>&1 | tail -1 >> "$LOG"
  python3 /root/bench_gen.py --tag v17a9_conc4_512 --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 --concurrency 4 2>&1 | tail -1 >> "$LOG"
fi

echo "$(ts) CHAIN: ALL ARMS COMPLETE" >> "$LOG"
