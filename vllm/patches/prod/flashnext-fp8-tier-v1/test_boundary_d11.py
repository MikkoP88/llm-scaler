#!/usr/bin/env python3
"""d11 boundary + template A/B + loop checks (run in-container).

python3 /stage/test_boundary_d11.py  [stage2]

User report: via Open WebUI all responses end mid-response.
Hypothesis (from the 16:45:59 crash): requests run to the 8192
max_model_len cap -> finish_reason=length (truncation looks like
'ends in the middle'), and the boundary spec step (T<L -> S=0) then
killed the engine (06q now skips it).  Proves:
  1. boundary: long prompt with max_tokens = remaining budget decodes
     INTO the cap -> finish_reason=length, engine SURVIVES, next
     request works.  (v1 of this script 400'd: prompt+max_tokens must
     fit under 8192 — this build rejects instead of clamping, which is
     ALSO what Open WebUI users see as a hard error if their UI sends
     a big max_tokens.)
  2. template A/B: same boundary shape with the NATIVE default
     template (per-request chat_template if supported, else
     tokenizer-rendered completions prompt) -> same length-cap
     behavior (cap is template-independent).
  3. loop check: long free generation -> no degenerate repetition and
     the request finishes.
"""
import sys
import time

import requests

BASE = "http://127.0.0.1:8321/v1"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-d"}
MODEL = "flash-next-fp8"
CAP = 8192

STAGE2 = "stage2" in sys.argv  # skip engine-healthy re-checks

results = []


def rec(name, ok, detail=""):
    results.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail[:260] if detail else ""), flush=True)


def rep_ratio(text, tail=800, sh=30):
    t = text[-tail:]
    n = max(1, len(t) - sh + 1)
    shingles = [t[i:i + sh] for i in range(0, n, sh // 2)]
    return len(set(shingles)) / max(1, len(shingles))


def full_text(msg):
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/models/flash-next-fp8")


def budget_for(user_content, reserve=6):
    """prompt token count via the SERVER's template + remaining budget.

    Robust across transformers return shapes: render to STRING
    (always a str) then tok.encode — never trust apply_chat_template(
    tokenize=True) length here (v1 did len(dict)==2 -> mt=8184 -> 400;
    v2's dict probe still mis-sized on this build)."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        chat_template=open("/stage/chat_template_flashnext_cc.jinja").read(),
        tokenize=False, add_generation_prompt=True)
    pt = len(tok.encode(rendered))
    return pt, max(16, CAP - pt - reserve)


filler = ("The quick brown fox jumps over the lazy dog near the "
          "river bank at dawn. ") * 420  # ~5200 tokens; leaves ~2900 gen

ESSAY = "\nWrite a detailed essay about rivers. Think as long as needed."

# ---- 1. boundary test: CC template (server default), decode INTO cap ----
try:
    content = filler + ESSAY
    pt, mt = budget_for(content)
    r = requests.post(BASE + "/chat/completions", headers=HDR, json={
        "model": MODEL, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_tokens": mt}, timeout=900)
    if r.status_code != 200:
        rec("boundary_cc_length_cap", False,
            f"HTTP{r.status_code} body={r.text[:180]}")
    else:
        j = r.json()
        ch = j["choices"][0]
        fr = ch.get("finish_reason")
        ct = j["usage"]["completion_tokens"]
        pt2 = j["usage"]["prompt_tokens"]
        txt = full_text(ch["message"])
        ratio = rep_ratio(txt)
        ok = fr == "length" and pt2 + ct <= CAP + 8 and ratio > 0.5
        rec("boundary_cc_length_cap", ok,
            f"finish={fr} pt={pt2}/{pt} ct={ct} pt+ct={pt2+ct} "
            f"rep_ratio={ratio:.2f} tail={txt[-110:]!r}")
except Exception as e:
    rec("boundary_cc_length_cap", False, f"{type(e).__name__}: {e}")

# engine survived + clean follow-up?
try:
    r = requests.post(BASE + "/chat/completions", headers=HDR, json={
        "model": MODEL, "messages": [{"role": "user", "content":
        "What is the capital of Germany? One word."}],
        "temperature": 0, "max_tokens": 1024}, timeout=300)
    c = full_text(r.json()["choices"][0]["message"])
    rec("post_boundary_engine_alive", r.status_code == 200 and "Berlin" in c,
        repr(c[:100]))
except Exception as e:
    rec("post_boundary_engine_alive", False, f"{type(e).__name__}: {e}")

# ---- 2. native default template boundary (per-request chat_template?) ----
try:
    native = tok.chat_template
    assert native and "tool_call" in native, "no native template"
    probe = requests.post(BASE + "/chat/completions", headers=HDR, json={
        "model": MODEL, "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 8, "temperature": 0, "chat_template": native}, timeout=120)
    content = filler + "\nWrite a detailed essay about rivers."
    if probe.status_code == 200:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": content}], chat_template=native,
            tokenize=True, add_generation_prompt=True)
        ptn = len(enc["input_ids"]) if isinstance(enc, dict) else len(enc)
        mtn = max(16, CAP - ptn - 6)
        r = requests.post(BASE + "/chat/completions", headers=HDR, json={
            "model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": mtn, "chat_template": native},
            timeout=900)
        j = r.json()
        ch = j["choices"][0]
        fr = ch.get("finish_reason")
        ct = j["usage"]["completion_tokens"]
        pt2 = j["usage"]["prompt_tokens"]
        txt = full_text(ch["message"])
        ratio = rep_ratio(txt)
        ok = (r.status_code == 200 and fr == "length"
              and pt2 + ct <= CAP + 8 and ratio > 0.5)
        rec("boundary_native_template", ok,
            f"per-req OK finish={fr} pt={pt2} ct={ct} "
            f"rep_ratio={ratio:.2f} tail={txt[-100:]!r}")
    else:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": content}], chat_template=native,
            tokenize=False, add_generation_prompt=True)
        ptn = len(tok.encode(prompt))
        mtn = max(16, CAP - ptn - 6)
        r = requests.post(BASE + "/completions", headers=HDR, json={
            "model": MODEL, "prompt": prompt, "max_tokens": mtn,
            "temperature": 0}, timeout=900)
        j = r.json()
        ch = j["choices"][0]
        fr = ch.get("finish_reason")
        ct = j["usage"]["completion_tokens"]
        pt2 = j["usage"]["prompt_tokens"]
        txt = ch.get("text") or ""
        ratio = rep_ratio(txt)
        ok = (r.status_code == 200 and fr == "length"
              and pt2 + ct <= CAP + 8 and ratio > 0.5)
        rec("boundary_native_template", ok,
            f"via completions probe={probe.status_code} finish={fr} "
            f"pt={pt2} ct={ct} rep_ratio={ratio:.2f} tail={txt[-100:]!r}")
except Exception as e:
    rec("boundary_native_template", False, f"{type(e).__name__}: {e}")

# ---- 3. loop check: long free generation under the cap ----
try:
    t0 = time.time()
    r = requests.post(BASE + "/chat/completions", headers=HDR, json={
        "model": MODEL, "messages": [{"role": "user", "content":
        "Count from 1 to 300, one number per line, then write DONE."}],
        "temperature": 0, "max_tokens": 1600}, timeout=900)
    j = r.json()
    ch = j["choices"][0]
    fr = ch.get("finish_reason")
    ct = j["usage"]["completion_tokens"]
    txt = full_text(ch["message"])
    ratio = rep_ratio(txt)
    ok = (r.status_code == 200 and ct >= 200 and ratio > 0.6
          and fr in ("stop", "length"))
    rec("no_degenerate_loop", ok,
        f"finish={fr} ct={ct} rep_ratio={ratio:.2f} {time.time()-t0:.0f}s "
        f"tail={txt[-100:]!r}")
except Exception as e:
    rec("no_degenerate_loop", False, f"{type(e).__name__}: {e}")

# ---- 4. log health: no NEW tracebacks since boot ----
try:
    log = open("/stage/serve_d11.log", "r", errors="replace").read()
    boot_n = log.find("Application startup complete")
    tail = log[boot_n:] if boot_n >= 0 else log
    tb = tail.count("Traceback")
    n6q = tail.count("FN8_06q")
    n6p = tail.count("FN8_06p")
    dead = tail.count("EngineCore failed") + tail.count("EngineDead")
    rec("serve_log_clean_no_tb", tb == 0 and dead == 0,
        f"since-boot tb={tb} dead={dead} 06q={n6q} 06p={n6p}")
    print(f"INFO 06q_hits={n6q} 06p_hits={n6p}")
except Exception as e:
    rec("serve_log_clean_no_tb", False, f"{type(e).__name__}: {e}")

bad = [n for n, ok in results if not ok]
print("RESULT", "ALL_PASS" if not bad else f"FAILED={bad}", flush=True)
