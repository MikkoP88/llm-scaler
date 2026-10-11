#!/bin/bash
# fn8 P1 profiling-window cutover, phase 1 — run INSIDE fn8-g1
# (foreground; the serve + corpus launches happen AFTER this via
# separate `docker exec -d` calls, per the background-launch law).
# Precondition: P0 smoke battery complete (all LEG lines + BANKED in
# /stage/g1_smoke.log).  Stops the P0 serve gracefully, applies
# patch_01, verifies the patched md5.
set -u
F=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py

echo "== pre-state capturer md5 (must be pristine 36d913c3...):"
md5sum "$F"
echo "== pre-state engine tree:"
pgrep -af "VLLM::|EngineCore" | head -5 || true

echo "== graceful TERM:"
pkill -TERM -f vllm
exited=0
for i in $(seq 1 90); do
  sleep 2
  if ! pgrep -f "VLLM::|EngineCore" > /dev/null 2>&1; then
    echo "engine tree exited after $((i*2))s"
    exited=1
    break
  fi
done
if [ "$exited" = 0 ]; then
  echo "LEFTOVERS after 180s — escalating to -9"
  pkill -9 -f "VLLM::|EngineCore"
  sleep 3
fi
if pgrep -af "VLLM::|EngineCore"; then
  echo "STILL_ALIVE — ABORT (host-level check needed)"
  exit 1
fi
echo "CONFIRMED_EMPTY"

echo "== apply patch_01 (expect 36d913c3... -> ae55ade2...):"
python3 /stage_patch01.py || { echo APPLY_FAILED; exit 1; }
md5sum "$F"
echo "PHASE1_DONE — now launch via docker exec -d:"
echo "  docker exec -d fn8-g1 bash /stage_serve_g1p.sh"
echo "  docker exec -d fn8-g1 bash -lc 'python3 /stage_corpus.py > /stage/profile_run.log 2>&1'"
