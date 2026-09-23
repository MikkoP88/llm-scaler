# ccthink-fix — Claude Code thinking display + reasoning_effort "high" (2026-09-23)

Fixes the "CC shows no thinking text" regression (litellm Responses-API
bridge dropping vLLM reasoning on /v1/messages) and the engine-side
`reasoning_effort high` 400 (stock template accepts xhigh/medium/low only).
No image change — engine v1.2.20 untouched. Full RCA: vllm/KNOWN_ISSUES.md
#26; narrative: ../wedgefix-v60/README.md tail section.

## What is deployed where (host 10.20.3.65)

| Artifact | Live path | Role |
|---|---|---|
| litellm_config.yaml | /root/litellm_config.yaml (mounted → litellm-proxy:/app/config.yaml) | `litellm_settings.use_chat_completions_url_for_anthropic_messages: true` (line ~195) routes /v1/messages via the chat/completions adapter instead of the Responses bridge |
| chat_template_qwen38_high.jinja | /root/build/chat_template_qwen38_high.jinja | accepts reasoning_effort `high`; served via `--chat-template` |
| repro_bootV1221.sh | /root/build/repro_bootV1221.sh | boot lineage (from V1212): template mount + boot-time flag injection + gates TEMPLATE_MISSING=11 / TEMPLATE_FLAG_MISSING=12 |
| lane_watchdog.sh | /root/build/lane_watchdog.sh (backup .pre_v1221) | :75 relaunches via repro_bootV1221.sh WD_* |
| probe_cc_think_battery.py | was /root/build/thk42.py | full battery: stream/nonstream thinking, E2/E2B, CC-shape, tools, history replay |
| probe_cc_400_bisect.py | was /root/build/thk36.py | 400-body capture + CC-shape reproduction |

## litellm recurrence guards (from #25/#26)

- Any re-create of litellm-proxy MUST keep `-e LITELLM_MASTER_KEY=sk-dummy`
  (db-less master-key mode; plain recreate → 400 "No connected db." on every
  route).
- Any config edit/removal of `use_chat_completions_url_for_anthropic_messages`
  re-breaks CC thinking silently — engine and logs stay clean.
- Env-var equivalent: `LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES`.

## Boot-script edit lesson

sed-inserted `docker run` continuation lines can lose their trailing `\`
(invisible to `bash -n`; docker run would truncate before the image name).
After any edit to repro_bootV*.sh: `diff` against the predecessor and verify
the mount block ends `ro \` with `sed -n 'Np' file | cat -A`.
