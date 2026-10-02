#!/usr/bin/env python3
# warm_ext_v1219.py -- extended in-bake JIT warm for llm-scaler-exp:v1.2.19.
# The v63 warm (probe_jitwarm_v63.py) covers a solo big prefill + one
# single-session decode burst only; every fresh v1.2.18 boot still
# JIT-compiles on first contended / big-turn traffic (v65 Phase-0 census):
#   expand_kernel (2.1-2.2 s stall), eagle_prepare_inputs_padded_kernel,
#   eagle_prepare_next_token_padded_kernel,
#   eagle_step_slot_mapping_metadata_kernel, rejection_greedy_sample_kernel,
#   _zero_kv_blocks_kernel, _compute_slot_mapping_kernel,
#   batch_memcpy_kernel (v52f COW on a 2nd turn over the same prefix).
#
# Round 1 (no --verify), 4 concurrent sessions x 2 turns:
#   s1 ~24k-word prefill (~75k tok) + 128-tok decode, then a 2nd turn on
#      the same history -> prefix-cache COW + post-prefill decode (the CC
#      big-turn shape that spiked expand_kernel)
#   s2-s4 short prompts + 128-tok decodes concurrent with s1's chunks ->
#      v63/v64 contended interleave steps + padded multi-session spec
#      batches (eagle_prepare_*_padded, rejection sample at batch > 1)
#
# Round 2 (--verify): lighter same-shape replay with fresh seeds (cache
# MISS -> fresh alloc paths again). The bake records the serve-log offset
# before round 2 and gates ZERO jit warnings for the kernels above after
# it: a compile in round 2 = a shape round 1 failed to cover.
#
# usage: python3 warm_ext_v1219.py <base> <model> <seed> [--verify]
import json
import random
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 519
VERIFY = "--verify" in sys.argv

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega quantum "
         "vector tensor matrix kernel compiler runtime driver device memory "
         "bandwidth latency throughput tile lane group schedule barrier "
         "prefix cache chunk decode prefill draft accept reject padding").split()


def mk_text(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


def post(payload):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read())
    return time.time() - t0, d


def turn(tag, messages, max_tokens):
    dt, d = post({"model": MODEL, "max_tokens": max_tokens,
                  "temperature": 0.7, "messages": messages})
    ch = d["choices"][0]
    print("[%s] %.1fs finish=%s len=%d" % (
        tag, dt, ch.get("finish_reason"),
        len(ch["message"].get("content") or "")), flush=True)
    return ch["message"].get("content") or ""


def session(tag, n_words, seed, n_turns, tok0, tokN):
    msgs = [{"role": "user",
             "content": mk_text(n_words, seed) + "\nReply with a few words."}]
    for i in range(n_turns):
        c = turn("%s.t%d" % (tag, i + 1), msgs, tok0 if i == 0 else tokN)
        msgs.append({"role": "assistant", "content": c})
        msgs.append({"role": "user",
                     "content": mk_text(40, seed + 100 + i) + "\nContinue briefly."})


if VERIFY:
    plan = [("v1", 9000, SEED + 71), ("v2", 300, SEED + 72),
            ("v3", 300, SEED + 73)]
    turns, tok0, tokN = 2, 96, 64
else:
    plan = [("s1", 24000, SEED), ("s2", 400, SEED + 1),
            ("s3", 400, SEED + 2), ("s4", 400, SEED + 3)]
    turns, tok0, tokN = 2, 128, 96

ts = [threading.Thread(target=session, args=(t, n, s, turns, tok0, tokN))
      for (t, n, s) in plan]
w0 = time.time()
for t in ts:
    t.start()
for t in ts:
    t.join()
print("WARM_EXT_V1219_DONE verify=%s wall=%.1fs" % (VERIFY, time.time() - w0))
