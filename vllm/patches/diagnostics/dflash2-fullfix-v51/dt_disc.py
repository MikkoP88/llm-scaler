# dt_disc.py — llm-scaler v51 Phase 4a (F9d): client-disconnect abort test.
#
# Crash-3 chain under test: client hard-closes mid-stream -> serving
# consumer task cancelled -> merge_async_iterators fast-path aclose (F9
# utils patch) -> async_llm generate() finally -> fire-and-forget
# abort(request_id, internal=True) -> engine frees the slot.
# Pre-F9 the abandoned generator was only closed by GC, so the request
# kept a running slot for a nondeterministic window (zombie leak).
#
# Method: open a streaming request on BOTH endpoints (completions + chat),
# read ~10 SSE events, hard-close the socket, then poll /metrics
# vllm:num_requests_running / waiting every 0.5 s with NO other traffic
# (so GC has no allocation pressure) for up to 60 s. time-to-zero <= a
# few seconds == abort fired at disconnect (F9 works); a request still
# running after tens of idle seconds == zombie (F9 gap).
import http.client
import json
import time
import urllib.request

HOST = "127.0.0.1"
PORT = 8000
MODEL = "qwen3.8-27b-fp8"


def metrics_running_waiting() -> tuple[int, int]:
    try:
        with urllib.request.urlopen(
            f"http://{HOST}:{PORT}/metrics", timeout=5
        ) as r:
            body = r.read().decode()
    except Exception as e:
        print(f"DISC metrics-error {e}")
        return (-1, -1)
    run = wait = -1
    for line in body.splitlines():
        if line.startswith("vllm:num_requests_running"):
            run = int(float(line.rsplit(" ", 1)[1]))
        elif line.startswith("vllm:num_requests_waiting"):
            wait = int(float(line.rsplit(" ", 1)[1]))
    return (run, wait)


def short(tag: str) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 8,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.loads(r.read())
    dt = time.time() - t0
    txt = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    print(f"DISC {tag} short-control {dt:.1f}s text={txt[:40]!r}")


def disconnect_probe(endpoint: str) -> None:
    if endpoint == "completions":
        body = {
            "model": MODEL,
            "prompt": "Write a html car game.",
            "max_tokens": 512,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }
        path = "/v1/completions"
    else:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Write a html car game."}],
            "max_tokens": 512,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }
        path = "/v1/chat/completions"
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    t0 = time.time()
    conn.request(
        "POST", path, body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        print(f"DISC {endpoint} FATAL status={resp.status}")
        conn.close()
        return
    events = 0
    first_event = None
    while events < 10:
        line = resp.readline()
        if not line:
            break
        s = line.decode(errors="replace").strip()
        if s.startswith("data:") and s != "data: [DONE]":
            events += 1
            if first_event is None:
                first_event = time.time()
        if time.time() - t0 > 20:
            break
    t_disc = time.time()
    conn.close()  # hard socket close mid-stream
    print(f"DISC {endpoint} disconnected after {t_disc - t0:.2f}s "
          f"({events} events, first@{0 if first_event is None else first_event - t0:.2f}s)")
    # Idle poll: no traffic -> no GC pressure. Zombie stays; F9 abort clears.
    t_zero = None
    while time.time() - t_disc < 60:
        run, wait = metrics_running_waiting()
        if run == 0 and wait == 0:
            t_zero = time.time()
            break
        time.sleep(0.5)
    if t_zero is None:
        run, wait = metrics_running_waiting()
        print(f"DISC {endpoint} VERDICT ZOMBIE: still running={run} "
              f"waiting={wait} after 60s idle")
    else:
        print(f"DISC {endpoint} VERDICT ABORTED-AT-DISCONNECT: "
              f"slots zero {t_zero - t_disc:.2f}s after close")
    time.sleep(2)


def main() -> None:
    print("DISC start (F9 client-disconnect abort test)")
    run, wait = metrics_running_waiting()
    print(f"DISC baseline running={run} waiting={wait}")
    short("pre")
    disconnect_probe("completions")
    disconnect_probe("chat")
    short("post")
    run, wait = metrics_running_waiting()
    print(f"DISC final running={run} waiting={wait}")
    print("DISC_DONE")


if __name__ == "__main__":
    main()
