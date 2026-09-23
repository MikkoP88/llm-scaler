# ccthink-fix — Claude Code thinking display + reasoning_effort "high" (2026-09-23)

Fixes the "CC shows no thinking text" regression (litellm Responses-API
bridge dropping vLLM reasoning on /v1/messages) and the engine-side
`reasoning_effort high` 400 (stock template accepts xhigh/medium/low only).
No image change — engine v1.2.20 untouched. Full RCA: vllm/KNOWN_ISSUES.md
#26; narrative: ../wedgefix-v60/README.md tail section.

## What is deployed where (host 10.20.3.65)

| Artifact | Live path | Role |
|---|---|---|
| litellm_config.yaml | /root/litellm_config.yaml (mounted → litellm-proxy:/app/config.yaml) | `litellm_settings.use_chat_completions_url_for_anthropic_messages: true` (line ~195) routes /v1/messages via the chat/completions adapter instead of the Responses bridge; `callbacks: [custom_callbacks.custom_callback]` loads the tool-args repair |
| custom_callbacks.py | /root/build/custom_callbacks.py → mounted to litellm site-packages | un-stringifies nested tool arguments (`{"questions": "[{...}]"}` → native array) so CC's AskUserQuestion dropdown and any array-in-array tool schema validate; fires on both stream (raw-SSE reframe) and non-stream paths |
| recreate_litellm.sh | /root/build/recreate_litellm.sh | disaster recovery: exact docker run (master-key env + config mount + module mount + pinned image 114aca7726c3). Run after ANY `docker rm` of litellm-proxy |
| chat_template_qwen38_high.jinja | /root/build/chat_template_qwen38_high.jinja | accepts reasoning_effort `high`; served via `--chat-template` |
| repro_bootV1221.sh | /root/build/repro_bootV1221.sh | boot lineage (from V1212): template mount + boot-time flag injection + gates TEMPLATE_MISSING=11 / TEMPLATE_FLAG_MISSING=12 |
| lane_watchdog.sh | /root/build/lane_watchdog.sh (backup .pre_v1221) | :75 relaunches via repro_bootV1221.sh WD_* |
| probe_cc_think_battery.py | was /root/build/thk42.py | full battery: stream/nonstream thinking, E2/E2B, CC-shape, tools, history replay |
| probe_cc_400_bisect.py | was /root/build/thk36.py | 400-body capture + CC-shape reproduction |

## Tool-args stringification defect (why the callback exists)

The model emits deeply-nested tool arguments as JSON-in-string
(`AskUserQuestion` → `{"questions": "\n[{...}]\n"}`). Verified at the engine
directly (:8000) — model emission, not litellm. `TodoWrite` (flat
array-of-objects) is emitted natively; only array-in-array nesting
stringifies. System-prompt nudge failed 3/3. The callback repairs it
server-side: buffers each tool_use block's input_json_delta, walks the
parsed args, replaces any string that strips to `[`/`{` and json-parses to a
container with its native value, re-emits one delta before
content_block_stop. Verified: streaming + all 3 non-stream tiers VALID=True;
battery (thinking/plain/CC-shape/nonthink) unchanged; callback log
`/tmp/custom_cb.log` records each repair. litellm-version-specific hook
contracts (learned the hard way): streaming hook must be an async GENERATOR
function (called as `hook(response=...)`, never awaited —
`_wrap_streaming_iterator_with_enrichment`); success hook receives `data=`
not `request_data=`.

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
