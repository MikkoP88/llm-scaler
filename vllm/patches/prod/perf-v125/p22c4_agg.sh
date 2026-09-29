#!/bin/bash
# p22c4_agg.sh — C4 aggregate follow-up (reworked post-P22C5): re-boots
# each of the six C4 leg serves (scripts built + gate-verified by
# p22c4_legs.sh) and runs the 4x256 concurrent aggregate probe TWICE per
# leg. Needed because the legs chain's agg probe had an indexing bug
# (o[2] on a one-element list -> IndexError after solo#2) — solo/battery
# are banked in p22c4_legs.log; this chain adds the contention numbers,
# including the e5m2 87.45-agg slow-stream re-check. v125 P22C5 rework:
# applies opsall + P22C5 (static fp8 spec bridge) so the s4 fp8 legs
# boot instead of dying at the wheel's spec dtype guard; s4 legs gate on
# the C5 marker. Ends restoring the certified serve on the CLEAN venv
# (all four files) + re-arm.
set -u
L=/root/build/lce1/p22c4_agg.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
XPU=/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py
XO=/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py

restore_lane() {
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU 2>/dev/null; cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; cp $XO.pre_v125_c5 $XO 2>/dev/null; python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

agg() {  # $1 = tag
python3 - "$1" <<'PYEOF'
import json, sys, threading, time, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
URL = "http://127.0.0.1:8000/v1/chat/completions"
def call(mt, out):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Count from 1 to 1000."}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        n = r.get("usage", {}).get("completion_tokens", mt)
        out.append((n, time.perf_counter() - t0))
    except Exception as e:
        print(f"[{tag}] agg_FAIL {type(e).__name__}: {e}")
        out.append((0, -1.0))
for run in (1, 2):
    outs = [[] for _ in range(4)]
    ths = [threading.Thread(target=call, args=(256, o)) for o in outs]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    recs = [o[0] for o in outs if o]
    ok = [n for n, dt in recs if dt > 0]
    toks = sum(ok)
    per = " ".join(f"{n}tok/{dt:.1f}s" for n, dt in recs)
    print(f"[{tag}] agg4x256#{run} wall={wall:.2f}s aggregate={toks/wall if wall>0 else 0:.2f} tok/s ok_streams={len(ok)}/4 [{per}]")
PYEOF
}

leg() {  # $1 tag, $2 serve script, $3 serve log, $4 want_spec_lines, $5 want_c5 (0/1/''=info)
  echo "=== AGGLEG $1 $(date +%F' '%T) ==="
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
  PD=0
  for i in $(seq 1 18); do
    sleep 5
    if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; break; fi
  done
  [ "$PD" -eq 1 ] || { echo "ABORT_${1}_PORT_STILL_UP"; exit 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $3; cd /root && nohup bash $2 > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' $3 | tail -5"; exit 1; }
  sleep 15
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $3; true")
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $3; true")
  C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' $3; true")
  echo "[$1] spec_lines=$SPL (want $4) v125fix=$FX (want >=1) c5_static=$C5 (want ${5:-info})"
  [ "$SPL" -eq "$4" ] || { echo "ABORT_${1}_SPEC_LINES"; exit 1; }
  [ "$FX" -ge 1 ] || { echo "ABORT_${1}_NO_FIX_WARNING"; exit 1; }
  if [ -n "${5:-}" ]; then
    [ "$C5" -eq "$5" ] || { echo "ABORT_${1}_C5_MARKER"; exit 1; }
  fi
  agg "$1"
}

{
echo "=== P22C4 AGG start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'P22C4_LEGS_DONE' /root/build/lce1/p22c4_legs.log && echo "legs chain DONE evidence present" || { echo "ABORT_LEGS_NOT_DONE"; exit 1; }
docker exec lsv-test sh -c "for f in serve_p22c4_16ns.sh serve_p22c4_e4m3ns.sh serve_p22c4_e5m2ns.sh serve_p22c4_e4m3s4.sh serve_p22c4_e5m2s4.sh; do test -x /root/\$f || exit 1; done && echo LEG_SCRIPTS_PRESENT" || { echo "ABORT_NO_LEG_SCRIPTS"; exit 1; }

echo "--- apply opsall root fix + P22C5 bridge to the venv ---"
grep -q 'P22C5_FIXVAL_DONE' /root/build/lce1/p22c5_fixval.log || { echo "ABORT_FIXVAL_NOT_DONE"; exit 1; }
docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF"
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }

docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC"
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

leg fix16ns  /root/serve_p22c4_16ns.sh   /root/serve_p22c4_16ns.log   0 ''
leg e4m3ns   /root/serve_p22c4_e4m3ns.sh /root/serve_p22c4_e4m3ns.log 0 ''
leg e5m2ns   /root/serve_p22c4_e5m2ns.sh /root/serve_p22c4_e5m2ns.log 0 ''
leg cert16s4 /root/serve_user.sh         /root/serve_full.log         2 0
leg e4m3s4   /root/serve_p22c4_e4m3s4.sh /root/serve_p22c4_e4m3s4.log 2 1
leg e5m2s4   /root/serve_p22c4_e5m2s4.sh /root/serve_p22c4_e5m2s4.log 2 1

echo "--- teardown + clean-venv restore + certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && cp $XO.pre_v125_c5 $XO && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo REVERT_CLEAN_OK" || { echo "ABORT_REVERT"; exit 1; }
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log; true")
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1\|v125 P22C5' /root/serve_full.log; true")
echo "restored spec_lines=$SPL2 fix-marker-lines=$INSTR2 (want 2 / 0)"
[ "$SPL2" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$INSTR2" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C4 AGG DONE $(date +%F' '%T) ==="
echo "P22C4_AGG_DONE"
} > "$L" 2>&1
tail -90 "$L"
echo P22C4_AGG_SCRIPT_DONE
