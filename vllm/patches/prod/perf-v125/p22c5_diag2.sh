#!/bin/bash
# p22c5_diag2.sh — v125 P22C5 defect mechanism discriminator.
#
# Convicted so far (p22c5_diag.sh, FULL graphs): static fp8 spec bridge
# corrupts a slot-stable subset of streams at n>1 (mt8 r2-r4 BAD=4,5,6
# every round; x8 0/10 clean; pair20 10/20; corruption = first token
# right then word-salad explosion), while ALL n=1 probes are clean — and
# the kernel/remap protocol demonstrably works at n=1.
#
# Two discriminators this chain runs in ONE eager boot:
#   1. EAGER vs CAPTURE — same fp8 spec-4 serve, --enforce-eager, no
#      FULL decode graphs. Clean eager => capture-replay interaction;
#      corrupt eager => bridge/kernel semantics proper.
#   2. IDENTICAL vs DISTINCT prompts — every corrupting probe so far used
#      identical prompts (prefix-cache slot-sharing territory); x8/pair
#      DISTINCT bursts falsify or confirm the sharing mechanism.
# Plus a once-only C5DUPD print (patch_v125_c5_dupdiag.py) at the first
# n>1 static gather: n, W, dup count, slot ids — fires eager-only.
#
# Ends: teardown + clean-venv restore (all four files) + certified serve
# + lane_watchdog re-arm. EXIT trap mirrors the restore.
set -u
L=/root/build/lce1/p22c5_diag2.log
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

probes() {
python3 - <<'PYEOF'
import json, threading, time, urllib.request
M = "qwen3.8-27b-fp8"
URL = "http://127.0.0.1:8000/v1/chat/completions"

def call(content, mt, out, idx):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        m = r["choices"][0]["message"]
        out[idx] = ((m.get("content") or "").strip(),
                    r["choices"][0].get("finish_reason"),
                    r.get("usage", {}).get("completion_tokens"))
    except Exception as e:
        out[idx] = (f"FAIL:{type(e).__name__}", "err", -1)

def burst(prompts, mt, tag, expect=None):
    # prompts: list of strings; expect: list of expected answers or None
    n = len(prompts)
    outs = [None] * n
    ths = [threading.Thread(target=call, args=(prompts[i], mt, outs, i)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    if expect is None:
        ok = sum(1 for o in outs if o[0] == "144")
        bad = [(i, o) for i, o in enumerate(outs) if o[0] != "144"]
        print(f"[{tag}] {ok}/{n} ok" + ("" if not bad else f" BAD={bad}"))
        return ok == n
    ok = sum(1 for i, o in enumerate(outs) if o[0] == expect[i])
    bad = [(i, o[0], expect[i]) for i, o in enumerate(outs) if o[0] != expect[i]]
    print(f"[{tag}] {ok}/{n} ok" + ("" if not bad else f" BAD={bad}"))
    return ok == n

ID8 = "What is 12*12? Reply with just the number."
IDP = "What is 7*8? Reply with just the number."

# 0) n=1 sanity (bridge untouched: ESIMD spec op at n=1)
o = [None]; call("What is 123+268? Reply with just the number.", 600, o, 0)
print(f"[n1-math] {o[0][0][:20]!r} -> {'OK' if '391' in o[0][0] else 'BAD'}")

# 1) mt8-identical x2 (reproduces the captured-leg corruption, now eager)
mt_i = all(burst([ID8]*8, 32, f"mt8id-r{r}") for r in (1, 2))
time.sleep(2)

# 2) x8-identical x6
xi = sum(1 for r in range(1, 7)
         if burst(["What is 9*11? Reply with just the number."]*8, 32,
                  f"x8id-r{r}", expect=["99"]*8))
print(f"[x8-identical] {xi}/6 rounds all-99")
time.sleep(2)

# 3) x8-distinct x6 — unique (a,b) per stream per round, no prefix share
xd = 0
for r in range(1, 7):
    a = [11 + ((r*8 + i) % 19) for i in range(8)]
    b = [3 + ((r*8 + i) % 7) for i in range(8)]
    pr = [f"What is {a[i]}*{b[i]}? Reply with just the number." for i in range(8)]
    if burst(pr, 32, f"x8di-r{r}", expect=[str(a[i]*b[i]) for i in range(8)]):
        xd += 1
print(f"[x8-distinct] {xd}/6 rounds all-correct")
time.sleep(2)

# 4) pair-identical x10
pi = sum(1 for r in range(1, 11)
         if burst([IDP]*2, 32, f"pid-r{r}", expect=["56", "56"]))
print(f"[pair-identical] {pi}/10 rounds both-56")
time.sleep(2)

# 5) pair-distinct x10
pd = 0
for r in range(1, 11):
    a = [13 + ((r*2 + i) % 17) for i in range(2)]
    b = [4 + ((r*2 + i) % 5) for i in range(2)]
    pr = [f"What is {a[i]}*{b[i]}? Reply with just the number." for i in range(2)]
    if burst(pr, 32, f"pdi-r{r}", expect=[str(a[i]*b[i]) for i in range(2)]):
        pd += 1
print(f"[pair-distinct] {pd}/10 rounds both-correct")

print(f"VERDICT-HINT eager: mt8id={mt_i} x8id={xi}/6 x8di={xd}/6 "
      f"pid={pi}/10 pdi={pd}/10 "
      f"(clean-eager => capture interaction; ident-only => prefix sharing; "
      f"all-corrupt => bridge/kernel semantics)")
PYEOF
}

{
echo "=== P22C5 DIAG2 start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- clean venv first (certified posture), then opsall + C5 + dupdiag ---"
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }

docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }
docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }
docker cp /root/build/patch_v125_c5_dupdiag.py lsv-test:/root/ || { echo "ABORT_CP_DUPD"; exit 1; }
PD=$(docker exec lsv-test python3 /root/patch_v125_c5_dupdiag.py)
echo "$PD" | grep -q 'V125_C5DUPD_OK\|V125_C5DUPD_ALREADY' || { echo "ABORT_DUPD_PATCH"; exit 1; }

echo "--- write eager serve script (e4m3 spec-4, --enforce-eager, no graphs) ---"
cat > /root/build/serve_p22c5_e4m3s4e.sh <<'SEOF'
#!/bin/bash
export VLLM_XPU_GDN_FP8_NATIVE=1
# serve_p22c5_e4m3s4e.sh — v125 P22C5 diag2 EAGER leg: identical to
# serve_p22c4_e4m3s4.sh except --enforce-eager replaces the
# FULL_DECODE_ONLY compilation config (isolates graph capture/replay
# as the discriminated variable). Runs INSIDE lsv-test.
cd /llm-scaler/vllm
exec vllm serve \
  --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --enforce-eager --async-scheduling --port 8000 \
  >> /root/serve_p22c5_e4m3s4e.log 2>&1
SEOF
docker cp /root/build/serve_p22c5_e4m3s4e.sh lsv-test:/root/ && docker exec lsv-test chmod +x /root/serve_p22c5_e4m3s4e.sh || { echo "ABORT_SERVE_SCRIPT"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

echo "--- boot e4m3s4 EAGER (spec-4 fp8 pool, static bridge, no graphs) ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
sleep 4
docker exec lsv-test sh -c ": > /root/serve_p22c5_e4m3s4e.log; cd /root && nohup bash serve_p22c5_e4m3s4e.sh > /dev/null 2>&1 & echo eager-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "eager health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c5_e4m3s4e.log | tail -8"; exit 1; }
sleep 15
# EAGER gate calibration (attempt-1 lesson): the v125fix/C5 markers print
# at FIRST INFERENCE CALL, not at boot — under captured legs they fired
# during capture (the dummy batch IS an inference); eager has no capture,
# so they CANNOT be present 15s after health. Boot-time gates are only
# the config ones; the marker gates run AFTER the probes.
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_p22c5_e4m3s4e.log; true")
E4M3=$(docker exec lsv-test sh -c "grep -c 'fp8_e4m3' /root/serve_p22c5_e4m3s4e.log; true")
CAP=$(docker exec lsv-test sh -c "grep -c 'Graph capturing' /root/serve_p22c5_e4m3s4e.log; true")
echo "[eager] spec_lines=$SPL (want 2) fp8_e4m3_cfg=$E4M3 (want >=1) capture_lines=$CAP (want 0)"
[ "$SPL" -eq 2 ] || { echo "ABORT_EAGER_SPEC_LINES"; exit 1; }
[ "$E4M3" -ge 1 ] || { echo "ABORT_EAGER_NO_FP8_CFG"; exit 1; }

probes

# POST-PROBE marker gates: by now the mt8id bursts ran n>=2 static-bridge
# batches — the C5 marker MUST have fired, else the bridge never engaged
# under eager (itself a dispatch finding worth aborting on).
C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' /root/serve_p22c5_e4m3s4e.log; true")
FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' /root/serve_p22c5_e4m3s4e.log; true")
echo "[eager post-probe] C5_static=$C5 (want >=1) v125fix=$FX (info)"
[ "$C5" -ge 1 ] || { echo "ABORT_EAGER_NO_C5_AFTER_PROBES"; exit 1; }

DUP=$(docker exec lsv-test sh -c "grep 'v125 C5DUPD' /root/serve_p22c5_e4m3s4e.log | head -3; true")
echo "C5DUPD lines: $DUP"

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
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1\|v125 P22C5\|v125 C5DUPD' /root/serve_full.log; true")
echo "restored spec_lines=$SPL2 fix-marker-lines=$INSTR2 (want 2 / 0)"
[ "$SPL2" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$INSTR2" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C5 DIAG2 DONE $(date +%F' '%T) ==="
echo "P22C5_DIAG2_DONE"
} > "$L" 2>&1
tail -60 "$L"
echo P22C5_DIAG2_SCRIPT_DONE
