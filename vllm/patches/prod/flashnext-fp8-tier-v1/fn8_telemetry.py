#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 telemetry — comprehensive window snapshot (host-side).

Runs on 10.20.3.65 as root; one JSON per invocation banked to
/root/fn8_stage/telemetry/snap_<ts>.json.  Captures every issue class the
series tracks:

  env         container flags (V1X_*/VLLM_*/DISABLE_*/...), serve args line,
              content-gate md5s + patch markers
  parameters  model config (expert/layer/shape fields), engine shape
              (tp/ep/dtype/max_len/offload/MNBT), pool-relevant geometry
  memory      xpu-smi per-card VRAM/power, docker stats cgroup, host
              meminfo + swap, stage disk free
  performance corpus legs + latencies, engine throughput tail, routing
              histogram header + LIVE coverage preview (activations-weighted,
              global and ep2 even/odd rank-local views — the P1 gate number
              mid-corpus)
  health      dmesg xe/oops/MCE/EDAC watch tail, uptime, loadavg

Zero engine-process contact: reads logs/files/proc/sysfs/smapi only — safe
to run against the busy profiling window at any cadence.  Pure stdlib.
"""
import json
import os
import re
import subprocess
import time

CONTAINER = os.environ.get("FN8_TELE_CONTAINER", "fn8-g1")
OUT_DIR = "/root/fn8_stage/telemetry"
CAPTURER = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
            "layers/fused_moe/routed_experts_capturer.py")
GATE_MD5S = {
    # content gates that must hold for the window (PHASES Round 6/8)
    "capturer_patched": "ae55ade2",
}


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001 — telemetry never hard-fails
        return f"<error {e}>"


def dex(cmd, timeout=120):
    return sh(f"docker exec {CONTAINER} {cmd}", timeout)


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def collect_env():
    env_raw = dex("env | grep -E '^(V1X_|VLLM_|DISABLE_|ENABLE_|PYTHONHASHSEED|"
                  "OMP|ZE_|SYCL|ONEAPI|NUMBA|NCCL|CCL)' | sort")
    args_line = ""
    for ln in dex("grep -m1 'non-default args' /stage/serve_g1b.log"
                  ).splitlines():
        args_line = ln
    gates = {
        "capturer_md5": dex(f"md5sum {CAPTURER}").split()[0]
        if dex(f"md5sum {CAPTURER}") else None,
        "capturer_gate_ok": None,
        "fn8_01_markers": dex(f"grep -c FN8_01 {CAPTURER}"),
        "serve_script_md5": dex("md5sum /stage_serve_g1b.sh").split()[0]
        if dex("md5sum /stage_serve_g1b.sh") else None,
        "engine_tree": dex("pgrep -af 'VLLM::|EngineCore' | head -3"),
    }
    if gates["capturer_md5"]:
        gates["capturer_gate_ok"] = gates["capturer_md5"].startswith(
            GATE_MD5S["capturer_patched"])
    return {"container": CONTAINER, "flags": env_raw, "serve_args": args_line,
            "gates": gates}


def collect_params():
    cfg_raw = dex("cat /models/flash-next-fp8/config.json")
    keep = ("num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "shared_expert_intermediate_size", "hidden_size",
            "num_hidden_layers", "num_routed_experts", "ple_layer_ids",
            "ple_context_len", "ple_state_len", "torch_dtype",
            "norm_topk_prob", "mtp_num_hidden_layers")
    cfg = {}
    try:
        full = json.loads(cfg_raw)
        for k in keep:
            if k in full:
                cfg[k] = full[k]
    except Exception as e:  # noqa: BLE001
        cfg = {"parse_error": str(e), "raw_head": cfg_raw[:200]}
    engine = {}
    m = re.search(r"Initializing a V1 LLM engine .*?with config: (.*)",
                  dex("grep -m1 'Initializing a V1 LLM engine' "
                      "/stage/serve_g1b.log"))
    if m:
        line = m.group(1)
        for field in ("tensor_parallel_size", "pipeline_parallel_size",
                      "data_parallel_size", "dtype", "max_seq_len",
                      "quantization=", "kv_cache_dtype", "seed",
                      "enable_prefix_caching", "enable_chunked_prefill"):
            mm = re.search(rf"{re.escape(field)}[=(]\S+", line)
            if mm:
                engine[mm.group(0)] = True
    na = dex("grep -m1 'non-default args' /stage/serve_g1b.log")
    if "'cpu_offload_gb':" in na:
        engine["cpu_offload_gb"] = re.search(
            r"'cpu_offload_gb': ([0-9.]+)", na).group(1)
    if "'max_num_batched_tokens':" in na:
        engine["max_num_batched_tokens"] = re.search(
            r"'max_num_batched_tokens': ([0-9]+)", na).group(1)
    if "'max_num_seqs':" in na:
        engine["max_num_seqs"] = re.search(
            r"'max_num_seqs': ([0-9]+)", na).group(1)
    return {"model_config": cfg, "engine_shape": engine}


def collect_memory():
    snap = {
        "xpu_smi": sh("xpu-smi dump 2>/dev/null | "
                      "grep -iE 'device|power|memory|utilization' | head -24"),
        "docker_stats": sh(f"docker stats --no-stream --format "
                           f"'{{{{.Name}}}} {{{{.MemUsage}}}} "
                           f"{{{{.CPUPerc}}}}' {CONTAINER}"),
        "host_free": sh("free -g | head -3"),
        "stage_disk": dex("df -h /stage | tail -1"),
        "cgroup_mem": dex("cat /sys/fs/cgroup/memory.current 2>/dev/null"
                          " || cat /sys/fs/cgroup/memory/memory.usage_in_bytes"
                          " 2>/dev/null"),
    }
    return snap


def collect_perf():
    perf = {"corpus_log": dex("grep -E 'LEG|PROFILE_RUN_DONE|HEALTH' "
                              "/stage/profile_run.log | tail -12")}
    legs = []
    try:
        pr = json.loads(dex("cat /stage/bank/profile_run.json"))
        for l in pr.get("legs", []):
            legs.append({k: l.get(k) for k in
                         ("name", "prompt_tokens", "completion_tokens",
                          "latency_s")})
    except Exception:  # noqa: BLE001 — absent file = corpus not started
        pass
    perf["legs"] = legs
    perf["throughput_tail"] = dex("grep 'loggers.py' /stage/serve_g1b.log "
                                  "| tail -4")
    # routing histogram header + live coverage preview
    hist = {}
    try:
        h = json.loads(dex("cat /stage/bank/routing_hist.json"))
        counts = h["counts"]  # [L][E]
        flat = []
        for li, row in enumerate(counts):
            for ei, c in enumerate(row):
                if c:
                    flat.append((c, li, ei))
        flat.sort(reverse=True)
        total = sum(c for c, _, _ in flat)
        hist = {
            "scatters": h.get("scatters"),
            "tapped_tokens": h.get("tapped_tokens"),
            "num_layers": h.get("num_layers"),
            "num_experts": h.get("num_experts"),
            "nonzero_pairs": len(flat),
            "total_activations": total,
        }
        for budget in (8192, 12288, 16384):
            cov = sum(c for c, _, _ in flat[:budget]) / max(total, 1)
            hist[f"coverage_within_{budget}"] = round(cov, 4)
        # ep2 rank-local views: even/odd global expert ids (interleaved
        # placement preview) + contiguous halves
        for split, pred in (
            ("rank0_even", lambda e: e % 2 == 0),
            ("rank1_odd", lambda e: e % 2 == 1),
            ("rank0_lo", lambda e: e < len(counts[0]) // 2),
            ("rank1_hi", lambda e: e >= len(counts[0]) // 2),
        ):
            sub = [(c, li, ei) for c, li, ei in flat if pred(ei)]
            subt = sum(c for c, _, _ in sub)
            # rank-local pool of 8192 global budget -> ~4096 local slots
            hist[f"{split}_share"] = round(subt / max(total, 1), 4)
            hist[f"{split}_cov_at_4096slots"] = round(
                sum(c for c, _, _ in sub[:4096]) / max(subt, 1), 4)
    except Exception as e:  # noqa: BLE001
        hist = {"status": f"not-yet: {e}"} if not dex(
            "ls /stage/bank/routing_hist.json") else {"error": str(e)}
    perf["routing_hist"] = hist
    return perf


def collect_health():
    return {
        "date": sh("date"),
        "uptime": sh("uptime"),
        "dmesg_watch": sh("dmesg -T 2>/dev/null | grep -iE "
                          "'xe |oops|bug:|MCE|EDAC|Hardware Error' "
                          "| tail -12"),
        "mce_count": sh("journalctl -k --since '-8 hours' 2>/dev/null | "
                        "grep -ic mce"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    snap = {
        "ts": ts,
        "env": collect_env(),
        "parameters": collect_params(),
        "memory": collect_memory(),
        "performance": collect_perf(),
        "health": collect_health(),
    }
    path = os.path.join(OUT_DIR, f"snap_{ts}.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=1)
    # compact one-screen summary
    p, h = snap["performance"], snap["performance"]["routing_hist"]
    print(f"TELEMETRY {path}")
    print(f"  legs={len(p['legs'])} "
          f"scatters={h.get('scatters')} "
          f"tok={h.get('tapped_tokens')} "
          f"cov8192={h.get('coverage_within_8192')} "
          f"cov12288={h.get('coverage_within_12288')}")
    g = snap["env"]["gates"]
    print(f"  gate_ok={g['capturer_gate_ok']} "
          f"engine_tree={'ALIVE' if g['engine_tree'] else 'EMPTY'}")
    watch = snap["health"]["dmesg_watch"]
    print(f"  dmesg_watch_lines={len(watch.splitlines()) if watch else 0}")


if __name__ == "__main__":
    main()
