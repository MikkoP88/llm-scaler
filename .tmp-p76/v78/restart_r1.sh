#!/bin/bash
# v78 R1 graceful restart v2 — TERM the serve CLI pid (found via /proc
# cmdline scan, setproctitle- and self-match-safe), verify FULL tree exit by
# comm scan (python3 + VLLM:: covers CLI / EngineCore / Workers / tracker),
# TERM a lingering resource_tracker if it is all that remains, then redeploy.
# v1 lesson: pgrep by comm is blind to setproctitle'd VLLM:: processes and
# can latch onto stale pids -> false "exited" + false FATAL.
set -e

SERVE_PID=$(docker exec lsv-test sh -c 'grep -l "bin/v[l]lm" /proc/[0-9]*/cmdline 2>/dev/null | head -n 1 | cut -d/ -f3' || true)
if [ -z "$SERVE_PID" ]; then
  echo "no serve process found - container may already be down; redeploying"
else
  echo "TERM to serve CLI pid $SERVE_PID"
  docker exec lsv-test kill -TERM "$SERVE_PID"
fi

count_tree() {
  docker exec lsv-test sh -c 'grep -l -E "^(python3|VLLM::)" /proc/[0-9]*/comm 2>/dev/null | wc -l'
}

for i in $(seq 1 75); do
  N=$(count_tree)
  if [ "$N" = "0" ]; then
    echo "container python/vllm tree fully exited after $((i * 2))s"
    break
  fi
  sleep 2
done

N=$(count_tree)
if [ "$N" != "0" ]; then
  echo "tree still alive (n=$N) - checking whether only a resource_tracker lingers"
  TRACKERS=$(docker exec lsv-test sh -c 'grep -l "resource[_]tracker" /proc/[0-9]*/cmdline 2>/dev/null | cut -d/ -f3' | tr "\n" " " || true)
  OTHERS=$(docker exec lsv-test sh -c 'for d in /proc/[0-9]*/comm; do c=$(cat "$d"); case "$c" in python3*|VLLM::*) p=${d%/comm}; p=${p#/proc/}; cmd=$(tr "\0" " " < "${p}/cmdline" 2>/dev/null); case "$cmd" in *resource_tracker*) ;; *) echo "$p $c $cmd";; esac;; esac; done' || true)
  if [ -n "$TRACKERS" ] && [ -z "$OTHERS" ]; then
    echo "only resource_tracker remains - TERM $TRACKERS"
    docker exec lsv-test sh -c "kill -TERM $TRACKERS" || true
    sleep 10
  fi
  N=$(count_tree)
fi

if [ "$N" != "0" ]; then
  echo "FATAL: tree still alive after 180s (n=$N):"
  docker exec lsv-test sh -c 'for d in /proc/[0-9]*/comm; do c=$(cat "$d"); case "$c" in python3*|VLLM::*) p=${d%/comm}; p=${p#/proc/}; printf "%s %s: " "$p" "$c"; tr "\0" " " < "${p}/cmdline" 2>/dev/null | cut -c 1-100; echo;; esac; done' || true
  echo "ABORT - lane untouched, investigate before any rm -f (orphan law)"
  exit 1
fi

sleep 2
bash /root/build/p77/run_v78.sh
