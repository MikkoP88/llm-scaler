#!/bin/bash
# p22c4_legs.sh — C4 six-leg fp8/fp16 matrix on the v125 opsall ROOT FIX
# (Run 7i: stale custom_ops='none' -> 'all' at the xpu.py TP>1 gate).
#
# Posture: the fix is applied to the venv ONCE (patch_v125_opsall_fix.py)
# and every leg boots through the natural path — NO explicit custom_ops
# anywhere, NO C1/D3 instruments. ns legs = spec-off + cudagraph NONE
# (graphs are useless on ns serves; prefill is never graphed and C1
# excludes FULL for ns decode) built from serve_p22c1_ctrl.sh; s4 legs =
# certified serve_user.sh posture (spec-4 + FULL 85-size graphs).
#
#   fix16ns   fp16 mamba,   spec-off, cudagraph NONE   (baseline ns)
#   e4m3ns    fp8_e4m3 mamba, spec-off, cudagraph NONE (VLLM_XPU_GDN_FP8_NATIVE=1)
#   e5m2ns    fp8_e5m2 mamba, spec-off, cudagraph NONE (VLLM_XPU_GDN_FP8_NATIVE=1)
#   cert16s4  fp16 mamba,   spec-4 + FULL graphs       (certified control)
#   e4m3s4    fp8_e4m3 mamba, spec-4 + FULL graphs     (VLLM_XPU_GDN_FP8_NATIVE=1)
#   e5m2s4    fp8_e5m2 mamba, spec-4 + FULL graphs     (VLLM_XPU_GDN_FP8_NATIVE=1)
#
# Gates per leg: spec_lines == want (0 ns / 2 s4), capture lines == want
# (0 ns / 1 s4), v125 root fix warnings >= 1 (fix fired through the
# natural boot path), v60g == 0, battery + speed probes. Apples-to-apples
# on one host window. Ends: teardown, clean-venv restore (xpu.py), cert
# serve_user.sh reboot, marker check, watchdog re-arm.
set -u
L=/root/build/lce1/p22c4_legs.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
XPU=/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py

restore_lane() {
  # restores EVERY file this chain touches (Run 7i lesson: fix2's trap
  # omitted xpu.py); idempotent via the pause flag
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU 2>/dev/null; cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$XPU', doraise=True); py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True)\" && echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

battery() {  # $1 = tag
python3 - "$1" <<'PYEOF'
import json, sys, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label, think=False, lp=False):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if lp:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        ch = r["choices"][0]
        m = ch["message"]
        vis = m.get("content") or ""
        rsn = m.get("reasoning_content") or ""
        print(f"[{tag}] {label} content_len={len(vis)} reasoning_len={len(rsn)} finish={ch.get('finish_reason')}")
        print(f"[{tag}] {label} TEXT {((vis + ' ||| ' + rsn) if rsn else vis)[:300]!r}")
        if lp and ch.get("logprobs"):
            for i, s in enumerate(ch["logprobs"].get("content", [])[:6]):
                top = ",".join(f"{e['token']!r}:{e['logprob']:.2f}"
                               for e in s.get("top_logprobs", [])[:3])
                print(f"[{tag}] {label} pos{i} {s['token']!r} top: {top}")
        return vis
    except Exception as e:
        print(f"[{tag}] {label}_FAIL {type(e).__name__}: {e}")
        return ""
for mt in (8, 64):
    ask("What is 17*23? Answer with the number only.", mt, f"MT{mt}", lp=True)
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
d = ask("Write a short story (5 sentences) about a robot learning to garden.", 400, "STORY400")
print(f"[{tag}] VERDICT math391={'391' in a} list5={'5' in b} prose_len={len(c)} story_len={len(d)}")
ask("What is 17*23? Answer with the number only.", 600, "THINK600", think=True)
PYEOF
}

speed() {  # $1 = tag
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
        vis = r["choices"][0]["message"].get("content") or ""
        out.append((n, len(vis), time.perf_counter() - t0))
    except Exception as e:
        print(f"[{tag}] speed_FAIL {type(e).__name__}: {e}")
        out.append((0, 0, -1.0))
for run in (1, 2):
    out = []
    call(200, out)
    n, chars, dt = out[0]
    if dt > 0:
        print(f"[{tag}] solo#{run} 200tok wall={dt:.2f}s = {n/dt:.2f} tok/s chars={chars} completion={n}")
outs = [[] for _ in range(4)]
ths = [threading.Thread(target=call, args=(256, o)) for o in outs]
t0 = time.perf_counter()
for t in ths: t.start()
for t in ths: t.join()
wall = time.perf_counter() - t0
ok = [o[0] for o in outs if o and o[2] > 0]
toks = sum(ok)
print(f"[{tag}] agg4x256 wall={wall:.2f}s aggregate={toks/wall if wall>0 else 0:.2f} tok/s ok_streams={len(ok)}/4 completions={ok}")
PYEOF
}

leg() {  # $1 tag, $2 serve script, $3 serve log, $4 want_spec_lines, $5 want_capture
  echo "=== LEG $1 $(date +%F' '%T) ==="
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
  CAP=$(docker exec lsv-test sh -c "grep -c 'Capturing' $3; true")
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $3; true")
  echo "[$1] spec_lines=$SPL (want $4) capture=$CAP (want $5) v125fix=$FX (want >=1)"
  [ "$SPL" -eq "$4" ] || { echo "ABORT_${1}_SPEC_LINES"; exit 1; }
  [ "$CAP" -eq "$5" ] || { echo "ABORT_${1}_CAPTURE"; exit 1; }
  [ "$FX" -ge 1 ] || { echo "ABORT_${1}_NO_FIX_WARNING"; exit 1; }
  echo "--- [$1] pass-1 posture:"
  docker exec lsv-test sh -c "grep -m1 'Initializing a V1 LLM engine' $3 | grep -oE \"enforce_eager=[A-Za-z]+|'mode': <CompilationMode.[A-Z_]+: [0-9]+>|custom_ops': \\[[^]]*\\]\""
  battery "$1"
  speed "$1"
  V6=$(docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' $3; true")
  echo "[$1] v60g count=$V6 (want 0)"
  [ "$V6" -eq 0 ] || { echo "ABORT_${1}_V60G"; docker exec lsv-test sh -c "grep 'v60g NO-DRAFT' $3 | head -2"; exit 1; }
}

{
echo "=== P22C4 LEGS start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'P22C1_FIX2_DONE' /root/build/lce1/p22c1_fix2.log && echo "fixval P22C1_FIX2_DONE present" || { echo "ABORT_NO_FIXVAL_EVIDENCE"; exit 1; }

echo "--- 1. apply opsall root fix to the venv ---"
docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True)\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF"
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }

echo "--- 2. build the five leg serve scripts ---"
docker exec lsv-test sh -c "test -f /root/serve_p22c1_ctrl.sh && test -f /root/serve_user.sh && grep -q FULL_DECODE_ONLY /root/serve_p22c1_ctrl.sh && grep -q num_speculative_tokens /root/serve_user.sh && echo SOURCES_OK" || { echo "ABORT_NO_SOURCES"; exit 1; }
NOG='s|{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[^]]*]}|{"cudagraph_mode":"NONE"}|; s|serve_p22c1_ctrl|serve_p22c4_16ns|g'
docker exec lsv-test sh -c "sed '$NOG' /root/serve_p22c1_ctrl.sh > /root/serve_p22c4_16ns.sh" || { echo "ABORT_BUILD_16NS"; exit 1; }
for v in e4m3 e5m2; do
  docker exec lsv-test sh -c "sed '$NOG; s|--mamba-ssm-cache-dtype float16|--mamba-ssm-cache-dtype fp8_$v|; s|serve_p22c4_16ns|serve_p22c4_${v}ns|g' /root/serve_p22c1_ctrl.sh > /root/serve_p22c4_${v}ns.sh" || { echo "ABORT_BUILD_${v}NS"; exit 1; }
  docker exec lsv-test sh -c "sed -i '2i export VLLM_XPU_GDN_FP8_NATIVE=1' /root/serve_p22c4_${v}ns.sh" || { echo "ABORT_ENV_${v}NS"; exit 1; }
  docker exec lsv-test sh -c "sed 's|--mamba-ssm-cache-dtype float16|--mamba-ssm-cache-dtype fp8_$v|; s|serve_full.log|serve_p22c4_${v}s4.log|; s|serve_user.sh|serve_p22c4_${v}s4.sh|' /root/serve_user.sh > /root/serve_p22c4_${v}s4.sh" || { echo "ABORT_BUILD_${v}S4"; exit 1; }
  docker exec lsv-test sh -c "sed -i '2i export VLLM_XPU_GDN_FP8_NATIVE=1' /root/serve_p22c4_${v}s4.sh" || { echo "ABORT_ENV_${v}S4"; exit 1; }
done
docker exec lsv-test sh -c "chmod +x /root/serve_p22c4_*.sh"
echo "--- built script gates:"
docker exec lsv-test sh -c "grep -q '\"cudagraph_mode\":\"NONE\"' /root/serve_p22c4_16ns.sh && ! grep -q num_speculative_tokens /root/serve_p22c4_16ns.sh && echo 16ns_OK" || { echo "ABORT_VERIFY_16NS"; exit 1; }
for v in e4m3 e5m2; do
  docker exec lsv-test sh -c "grep -q '\"cudagraph_mode\":\"NONE\"' /root/serve_p22c4_${v}ns.sh && grep -q 'mamba-ssm-cache-dtype fp8_$v' /root/serve_p22c4_${v}ns.sh && grep -q VLLM_XPU_GDN_FP8_NATIVE /root/serve_p22c4_${v}ns.sh && ! grep -q num_speculative_tokens /root/serve_p22c4_${v}ns.sh && echo ${v}ns_OK" || { echo "ABORT_VERIFY_${v}NS"; exit 1; }
  docker exec lsv-test sh -c "grep -q num_speculative_tokens /root/serve_p22c4_${v}s4.sh && grep -q FULL_DECODE_ONLY /root/serve_p22c4_${v}s4.sh && grep -q 'mamba-ssm-cache-dtype fp8_$v' /root/serve_p22c4_${v}s4.sh && grep -q VLLM_XPU_GDN_FP8_NATIVE /root/serve_p22c4_${v}s4.sh && echo ${v}s4_OK" || { echo "ABORT_VERIFY_${v}S4"; exit 1; }
done
docker exec lsv-test sh -c "grep -hE 'compilation-config|mamba-ssm-cache-dtype|GDN_FP8' /root/serve_p22c4_16ns.sh /root/serve_p22c4_e4m3ns.sh /root/serve_p22c4_e5m2ns.sh /root/serve_p22c4_e4m3s4.sh /root/serve_p22c4_e5m2s4.sh | cut -c1-160"

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

leg fix16ns  /root/serve_p22c4_16ns.sh   /root/serve_p22c4_16ns.log   0 0
leg e4m3ns   /root/serve_p22c4_e4m3ns.sh /root/serve_p22c4_e4m3ns.log 0 0
leg e5m2ns   /root/serve_p22c4_e5m2ns.sh /root/serve_p22c4_e5m2ns.log 0 0
leg cert16s4 /root/serve_user.sh         /root/serve_full.log         2 1
leg e4m3s4   /root/serve_p22c4_e4m3s4.sh /root/serve_p22c4_e4m3s4.log 2 1
leg e5m2s4   /root/serve_p22c4_e5m2s4.sh /root/serve_p22c4_e5m2s4.log 2 1

echo "--- capacity echo per leg ---"
for lg in "serve_p22c4_16ns.log fix16ns" "serve_p22c4_e4m3ns.log e4m3ns" "serve_p22c4_e5m2ns.log e5m2ns" "serve_full.log cert16s4" "serve_p22c4_e4m3s4.log e4m3s4" "serve_p22c4_e5m2s4.log e5m2s4"; do
  set -- $lg
  docker exec lsv-test sh -c "grep -E 'GPU KV cache size|Maximum concurrency|Padding mamba page size|mamba.*pool|Mamba' /root/$1 | head -6" | sed "s/^/[$2] /"
done

echo "--- teardown + clean-venv restore + certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && python3 -c \"import py_compile; py_compile.compile('$XPU', doraise=True); py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True)\" && echo REVERT_CLEAN_OK" || { echo "ABORT_REVERT"; exit 1; }
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log; true")
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1' /root/serve_full.log; true")
echo "restored spec_lines=$SPL2 fix-marker-lines=$INSTR2 (want 2 / 0)"
[ "$SPL2" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$INSTR2" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C4 LEGS DONE $(date +%F' '%T) ==="
echo "P22C4_LEGS_DONE"
} > "$L" 2>&1
tail -160 "$L"
echo P22C4_LEGS_SCRIPT_DONE
