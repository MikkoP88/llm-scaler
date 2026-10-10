# Fleet context — v78 R1b validation (TP1-replicated MTP head)

## What is under test
Lane `lsv-test` on host 10.20.3.65 runs vLLM serve with:
- Target: Qwen3.5-style 27B, TP2 (2 XPU cards)
- Speculative decode: MTP (method=mtp, num_speculative_tokens=3)
- v78 R1b (NEW, under test): `VLLM_XPU_MTP_TP1=1` — the MTP drafter head is
  loaded REPLICATED (full weights per rank, TP1 view) so it issues ZERO
  oneCCL collectives per step. Expected effects in SPECTIMING:
  propose device-time ~12ms → ~5-6ms; greedy device ~2.2ms → ~0.5ms.
- Async scheduling ON (`VLLM_XPU_ALLOW_ASYNC=1`), barrier 0, v52k timers on.

## Endpoints (from this Windows host, direct)
- Base: `http://10.20.3.65:8000`
- Health: `GET /health` → expect 200
- Models: `GET /v1/models` → discover model id
- Chat: `POST /v1/chat/completions` (OpenAI shape). Temperature ≤ 0.5
  (temp ≥ 0.6 has a known sampling attractor — avoid contaminating results).

## Host access (READ-ONLY + curl test traffic ONLY)
```
plink -batch -pw <HOSTPW> root@10.20.3.65 "<command>"
```
- Serve log: `docker exec lsv-test tail -N /root/serve_full.log`
- SPECTIMING lines appear as `SPECTIMING steps=200 | step_wall_ms=...`
- Stall fingerprint lines (user-reported stall signature): `v52f PRE-COPY`
  and `v52f DEF-PP` (Worker_TP0/TP1 lines, mamba_utils.py).
- plink traps: escape `\$` for shell vars inside double quotes; NEVER
  background a process inside plink; NEVER nest quotes deeply.

## HARD SAFETY RULES (no exceptions)
1. NEVER reboot the host, NEVER `pkill`/`kill` anything, NEVER
   `docker rm/stop/restart/exec -d` — the coordinator owns lifecycle.
2. NEVER touch litellm (port 4000, container litellm-proxy) or read
   /root/litellm_config.yaml.
3. NEVER modify files under /opt/venv inside the container or /root/build
   on the host. Log harvesting = `tail`/`grep` reads only.
4. On engine death (health 000 / EngineCore died / DEVICE_LOST / ccs reset
   in log): STOP generating load immediately, bank evidence
   (`tail -300` of serve_full.log into a local file), mark verdict CRASH
   in your report, and exit. Do NOT attempt recovery.
5. Work ONLY inside your own directory. Report file names are fixed in
   your mission.

## Stall definitions
- TTFT > 60s on a standard-size request, or total wall > 300s standard /
  > 900s deep, or any single request with no completion for > 300s.
- Any `v52f PRE-COPY` / `v52f DEF-PP` line in the serve log during your
  leg (timestamp it).
