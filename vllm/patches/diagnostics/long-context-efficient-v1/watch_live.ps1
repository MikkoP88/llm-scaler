param([int]$IntervalSeconds = 60, [int]$MaxHours = 48)
$ErrorActionPreference = 'Stop'
$watchDir = Join-Path $PSScriptRoot 'evidence/live-watch'
New-Item -ItemType Directory -Path $watchDir -Force | Out-Null
$deadline = (Get-Date).AddHours($MaxHours)
# Read-only host observations; no inference requests, restarts or test edits.
$remoteProbe = @'
date -u
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
pgrep -af '^bash master_lce1.sh|^python3 /root/build/nlp_suite.py' || true
tail -5 /root/build/lce1/master.log
curl -s --max-time 8 -o /dev/null -w 'HEALTH_HTTP=%{http_code}\n' localhost:8000/health
curl -s --max-time 8 localhost:8000/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)' || true
docker exec lsv-test sh -c 'for f in /root/b_lce1_*.log; do test -f "$f" && tail -10 "$f"; done' 2>&1
'@
$remoteProbe = $remoteProbe.Replace("`r", '')
while ((Get-Date) -lt $deadline) {
    $stamp = (Get-Date).ToUniversalTime().ToString('o')
    try {
        $lines = & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 root@10.20.3.65 $remoteProbe 2>&1
        $exitCode = $LASTEXITCODE
        $body = $lines -join "`n"
        $record = [ordered]@{utc=$stamp; ssh_exit=$exitCode; observation=$body}
        ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath (Join-Path $watchDir 'snapshots.jsonl') -Encoding utf8
        @("Observed UTC: $stamp", "SSH exit: $exitCode", '', $body) | Set-Content -LiteralPath (Join-Path $watchDir 'LATEST.txt') -Encoding utf8
        if ($body -match 'MASTER_LCE1_DONE') { break }
    } catch {
        "$stamp $($_.Exception.Message)" | Add-Content -LiteralPath (Join-Path $watchDir 'watch-errors.txt')
    }
    Start-Sleep -Seconds $IntervalSeconds
}
