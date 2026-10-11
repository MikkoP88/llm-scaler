#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P1 — routing-profile corpus runner (fn8-g1p :8319).

Drives a code+prose+agentic corpus through the profiling serve
(fn8_serve_g1p.sh) so patch_01's V1X_ROUTING_DUMP tap banks a routing
histogram.  Fat prompts (prefill tokens are cheap per token at P0 speed:
one 104 GiB copy per chunk amortizes) + short decodes (expensive per
token).  Banks per-leg latencies to /stage/bank/profile_run.json.

Afterwards: python3 /stage/fn8_expert_profile.py /stage/bank/routing_hist.json
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8319"
BANK = "/stage/bank/profile_run.json"

PY_FIB = '''Complete this Python module. It must provide fib(n) iteratively with
memoization, a CLI entry point, and doctests.

```python
import sys
from functools import lru_cache


def fib(n: int) -> int:
    """Return the n-th Fibonacci number (fib(0)=0, fib(1)=1).
    >>> [fib(i) for i in range(10)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
'''

PY_SORT = '''Finish this implementation of quicksort with a three-way partition and
a small-array insertion-sort cutoff, then add a property test.

```python
def quicksort(a: list, lo: int = 0, hi: int | None = None) -> list:
    if hi is None:
        hi = len(a) - 1
    while lo < hi:
        if hi - lo < 16:
            _insertion(a, lo, hi)
            return a
        p = _median3(a, lo, hi)
'''

SH_DEPLOY = '''Write a bash deployment script that: pulls the current git tag, builds a
docker image with a pinned base, runs the test suite first and aborts on
failure, then swaps the running container behind a health check loop with
a rollback if the new container does not answer /health within 60 seconds.
Include logging to syslog and a lock file so two deploys cannot race.'''

SQL_ANALYTICS = '''Given tables orders(order_id, customer_id, created_at, total_cents,
status) and customers(customer_id, signup_at, tier), write a SQL query
returning for each customer tier the monthly cohort retention over the
last 12 months: month, tier, customers_active, orders, revenue_cents,
plus a 3-month rolling average. Explain the index you would add.'''

EXPLAIN_MOE = '''Explain in detail how mixture-of-experts layers work in modern LLMs:
the router, top-k gating, load balancing losses during training, expert
parallelism versus tensor parallelism at serving time, and why sparse
activation keeps FLOPs low while parameter counts grow. Then compare it
to dense scaling.'''

HIST_ESSAY = '''Write an essay on the history of virtual memory: from atlas paging and
multics segments, through demand paging and copy-on-write fork, to NUMA
interleave policies and modern huge pages. Cover why the abstraction won
over manual overlays and what broke when RAM hierarchies got deep.'''

AGENTIC_TRACE = '''You are an agent with tools: search(query), read_file(path),
write_file(path, content), run_tests(). Task: the CI log shows a flaky
test in test_cache.py that fails about one run in eight with a KeyError.
Describe your step-by-step investigation, the tool calls you would make,
what you would look for in the cache implementation, and how you would
fix and verify it. Be concrete about each step.'''

CORPUS = [
    ("py_fib", PY_FIB, 32),
    ("py_sort", PY_SORT, 32),
    ("sh_deploy", SH_DEPLOY, 32),
    ("sql_analytics", SQL_ANALYTICS, 32),
    ("explain_moe", EXPLAIN_MOE, 48),
    ("hist_essay", HIST_ESSAY, 48),
    ("agentic_trace", AGENTIC_TRACE, 48),
    # phrasing-shifted repeat of leg 1: routing-stability eyeball
    ("py_fib_v2", PY_FIB.replace("Complete this", "Here is the start of").replace(
        "It must provide", "please provide"), 32),
]


def req(path, payload, timeout=14400):
    r = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_health(deadline_s=14400):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    print("HEALTH_200 after", round(time.time() - t0, 1), "s",
                          flush=True)
                    return
        except Exception:
            pass
        time.sleep(10)
    raise SystemExit("health deadline exceeded")


def main():
    wait_health()
    bank = {"ts": time.time(), "legs": []}
    for name, prompt, max_tok in CORPUS:
        t0 = time.time()
        out = req("/v1/completions", {
            "model": "flash-next-fp8",
            "prompt": prompt,
            "max_tokens": max_tok,
            "temperature": 0.0,
        })
        dt = time.time() - t0
        usage = out.get("usage", {})
        text = out["choices"][0]["text"]
        leg = {"name": name, "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "latency_s": round(dt, 2), "text_head": text[:100]}
        bank["legs"].append(leg)
        print(f"LEG {name} p={leg['prompt_tokens']} "
              f"c={leg['completion_tokens']} {dt:.1f}s :: {text[:60]!r}",
              flush=True)
        with open(BANK, "w") as f:
            json.dump(bank, f, indent=1)
    print("PROFILE_RUN_DONE", BANK, flush=True)


if __name__ == "__main__":
    main()
