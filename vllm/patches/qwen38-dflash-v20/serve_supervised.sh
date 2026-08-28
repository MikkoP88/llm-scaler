#!/usr/bin/env bash
# llm-scaler supervised serve (KNOWN_ISSUES #03 companion, baked at /opt).
# The warmup watchdog (VLLM_XPU_WARMUP_TIMEOUT_S, gpu_worker.py) hard-exits
# a wedged warmup with a full traceback dump; this supervisor restarts the
# engine automatically instead of leaving a dead port in a healthy-looking
# container. Crashing exits restart; clean exits stop the loop.
#
# Env:
#   VLLM_SUPERVISE_MAX_RESTARTS  restart budget (default 3; 0 = unlimited)
#   VLLM_SUPERVISE_COOLDOWN_S    pause between restarts (default 30)
# All arguments are forwarded verbatim to `vllm serve`.
#
# Usage (docker --entrypoint /opt/serve_supervised.sh <image> -- serve args...):
#   serve_supervised.sh --model /models/target --tensor-parallel-size 2 ...
set -uo pipefail

MAX_RESTARTS="${VLLM_SUPERVISE_MAX_RESTARTS:-3}"
COOLDOWN_S="${VLLM_SUPERVISE_COOLDOWN_S:-30}"
restarts=0
rc=0

while true; do
    echo "[supervisor] $(date -Is) starting vllm serve (attempt $((restarts + 1)))"
    vllm serve "$@"
    rc=$?
    echo "[supervisor] $(date -Is) vllm serve exited rc=$rc"
    # 0 = clean stop, 130 = SIGINT, 143 = SIGTERM -> do not restart.
    if [[ $rc -eq 0 || $rc -eq 130 || $rc -eq 143 ]]; then
        echo "[supervisor] clean exit; supervisor stopping"
        exit "$rc"
    fi
    if [[ "$MAX_RESTARTS" != "0" && $restarts -ge "$MAX_RESTARTS" ]]; then
        echo "[supervisor] restart budget exhausted ($restarts); giving up (rc=$rc)" >&2
        exit "$rc"
    fi
    restarts=$((restarts + 1))
    echo "[supervisor] abnormal exit; cooldown ${COOLDOWN_S}s then restart $restarts/$MAX_RESTARTS"
    sleep "$COOLDOWN_S"
done
