#!/bin/bash
# cc_fleet_launch.sh — v127 W1/W2 controlled CC fleet load (user directive
# 2026-09-30: "start own Claude CLI instances with multiple agents and
# subagents, use qwen3.8-27b-fp8-opus-max and qwen3.8-27b-fp8-sonnet").
# Runs on the LOCAL Windows machine (Git Bash); local ~/.claude/settings.json
# already routes ANTHROPIC_BASE_URL -> 10.100.8.6:4004 (master key sk-dummy)
# with CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-27b-fp8-sonnet, so:
#   --model qwen3.8-27b-fp8-opus-max  -> main agent on opus-max
#   --model qwen3.8-27b-fp8-sonnet    -> main agent on sonnet
#   subagents (Task tool)             -> sonnet (settings default)
# Tasks are READ-ONLY analyses of this repo (long multi-turn context growth +
# subagent fan-out = the real CC storm regime); reports go to .tmp-ccload/.
# usage: bash cc_fleet_launch.sh <N_OPUSMAX> <N_SONNET>   (default 2 4)
set -u
cd /c/Projects/llm-scaler
N_OP="${1:-2}"; N_SO="${2:-4}"
LOGD=/c/Projects/llm-scaler/.tmp-ccload
mkdir -p "$LOGD"
TS=$(date +%H%M%S)

RO='STRICT: read-only for this repository — do NOT edit, create, or delete any file outside C:\Projects\llm-scaler\.tmp-ccload\. Use subagents for exploration per CLAUDE.md.'

declare -a TASKS=(
  "Deep-read vllm/patches/prod/perf-v127/FIX_AND_TEST_PLAN.md and vllm/patches/prod/perf-v127/STALL_ROOT_CAUSE.md. Use subagents to verify every workstream WS-A..WS-G names its telemetry capture points and gates. Write a gap list with file:line evidence to $LOGD\\report_A.md. $RO"
  "Use subagents to map the vLLM scheduler patch stack: find all 'llm-scaler v6x' markers under vllm/patches (v63/v64/v66 and their patch scripts) and explain the contended-prefill budget + decode interleave + fairness-bypass machinery and every env knob. Write to $LOGD\\report_B.md. $RO"
  "Analyze vllm/patches/prod/perf-v126/PHASES.md end-to-end: build a timeline table of phases P27..P32 with verdicts, and list every open diagnostic debt item still unresolved. Write to $LOGD\\report_C.md. $RO"
  "Use subagents to audit the boot chain scripts under vllm/patches/prod (repro_boot*, boot_v1227*, watchdog*): list every gate each script asserts, with exit codes, and flag any script that could relaunch the lane with a stale posture. Write to $LOGD\\report_D.md. $RO"
  "Study vllm/patches/prod/perf-v127/cc_fleet_replay.py and metrics_recorder.py: propose (as text only) two additional per-request metrics that would expose re-prefill storms earlier, with exact metric names and where they would come from. Write to $LOGD\\report_E.md. $RO"
  "Read vllm/patches/prod/crashfix-v55/ and wedgefix-v75/ histories via subagents and write a one-page root-cause taxonomy of every wedge/stall class this lane has ever hit, each with its fix and the artifact that proves it. Write to $LOGD\\report_F.md. $RO"
)

launch() { # $1=model $2=idx $3=task
  # --model and ANTHROPIC_MODEL both reject custom gateway aliases
  # (unrecognized_model). Supported passthrough = alias + remap env:
  # --model opus   + ANTHROPIC_DEFAULT_OPUS_MODEL=<lane model>
  # --model sonnet + ANTHROPIC_DEFAULT_SONNET_MODEL=<lane model>
  local log="$LOGD/inst_$2_$TS.log" alias envv
  case "$1" in
    *opus*)   alias=opus;   envv="ANTHROPIC_DEFAULT_OPUS_MODEL=$1" ;;
    *sonnet*) alias=sonnet; envv="ANTHROPIC_DEFAULT_SONNET_MODEL=$1" ;;
    *)        alias=sonnet; envv="ANTHROPIC_MODEL=$1" ;;
  esac
  echo "launch inst=$2 model=$1 (alias=$alias) log=$log"
  ( env "$envv" claude -p "$3" --model "$alias" > "$log" 2>&1 ; echo "EXIT inst=$2 rc=$? $(date +%H:%M:%S)" >> "$LOGD/exits_$TS.log" ) &
}

i=0
for k in $(seq 1 "$N_OP"); do
  launch qwen3.8-27b-fp8-opus-max "opus$k" "${TASKS[$((i % 6))]}"; i=$((i+1))
done
for k in $(seq 1 "$N_SO"); do
  launch qwen3.8-27b-fp8-sonnet "son$k" "${TASKS[$((i % 6))]}"; i=$((i+1))
done
echo "CC_FLEET_LAUNCHED opusmax=$N_OP sonnet=$N_SO ts=$TS"
wait
echo "CC_FLEET_ALL_EXITED ts=$TS"
