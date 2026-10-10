# Mission: WATCHER — continuous lane telemetry harvester

Read `../CONTEXT.md` first. You watch, you never generate load. Work in
this directory only. Runtime budget: exit no later than 100 minutes after
start. Produce `WATCH_LOG.md` + `spectiming.log` + `fingerprints.log` +
`errors.log`.

## Loop (every 20 seconds, for up to 100 minutes)
0. BOOT GRACE: for the first 12 minutes, health 000 is EXPECTED (engine
   booting). Log it, take no action, do not run crash-evidence.
1. Health: `curl -s -o /dev/null -w "%{http_code}" http://10.20.3.65:8000/health`
   → append `ts code` to health.log (create it). If code is 000 twice in a
   row AFTER the boot grace → run the crash-evidence step once, mark
   DEATH_TIME, keep looping (post-mortem harvesting is valuable) but stop
   calling the endpoint more than once per 60s.
2. Serve log tail via plink (read-only):
   `docker exec lsv-test tail -400 /root/serve_full.log` then locally:
   - lines containing `SPECTIMING` → append to spectiming.log (dedupe).
   - lines containing `v52f PRE-COPY` or `v52f DEF-PP` → append with ts
     to fingerprints.log.
   - lines containing `ERROR`, `Traceback`, `ccs`, `reset`, `DEVICE_LOST`,
     `EngineCore died` → append to errors.log.
3. Never write to the remote. Never restart anything.

## Crash-evidence step (once, on first 000×2)
`docker exec lsv-test tail -300 /root/serve_full.log` → save locally as
crash_tail.txt. Note exact timestamp.

## Exit report (WATCH_LOG.md)
- Timeline: boot-healthy window, any stall fingerprint bursts (counts per
  minute), error taxonomy with first-seen timestamps.
- spectiming.log line count + 3 representative lines.
- Verdict line: `WATCH: CLEAN | FINGERPRINTS | ERRORS | DEATH`.
