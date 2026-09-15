# KB build watchdog: keep build_kb_fast.py alive until the target record count is reached.
# Restarts on death; also restarts a hung builder whose output has been stale for > StaleSec.
param(
    [string]$Root = "D:\Python\MMDIT",
    [int]$Target = 3546,
    [int]$PollSec = 60,
    [int]$StaleSec = 1800
)

$kb      = Join-Path $Root "experiment\2026_9_14_EXP_4_CULTURE_KB"
$out     = Join-Path $kb   "kb\kb_all.jsonl"
$log     = Join-Path $kb   "logs\kb_build.out.log"
$err     = Join-Path $kb   "logs\kb_build.err.log"
$wlog    = Join-Path $kb   "logs\watchdog_kb.out.log"
$pidfile = Join-Path $kb   "logs\kb_build.pid"

function WLog($m) { "$(Get-Date -Format s) $m" | Add-Content -Path $wlog -Encoding utf8 }
function CountLines() { if (Test-Path $out) { (Get-Content $out | Measure-Object).Count } else { 0 } }
function BuilderProcs() {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*build_kb_fast.py*" }
}
function StartBuilder() {
    $env:PYTHONIOENCODING = "utf-8"
    $p = Start-Process -FilePath "python" -ArgumentList @(
        "code\kb\build_kb_fast.py",
        "--concepts", "experiment\2026_9_14_EXP_5_CULTURE_COMP\benchmark\concepts_all.jsonl",
        "--out", "experiment\2026_9_14_EXP_4_CULTURE_KB\kb\kb_all.jsonl",
        "--model", "gpt-4o-mini", "--fallback-model", "gpt-4o"
    ) -WorkingDirectory $Root -RedirectStandardOutput $log -RedirectStandardError $err `
      -WindowStyle Hidden -PassThru
    if ($p) { $p.Id | Set-Content -Path $pidfile -Encoding ascii; WLog "started builder pid=$($p.Id)" }
}

WLog "watchdog start target=$Target poll=${PollSec}s stale=${StaleSec}s"
while ($true) {
    $n = CountLines
    if ($n -ge $Target) { WLog "ALL DONE lines=$n"; break }

    $procs = @(BuilderProcs)
    if ($procs.Count -eq 0) {
        WLog "builder not running (lines=$n); restarting"
        StartBuilder
    } else {
        $files = @()
        foreach ($f in @($out, $log)) { if (Test-Path $f) { $files += (Get-Item $f).LastWriteTime } }
        $last = if ($files.Count) { ($files | Measure-Object -Maximum).Maximum } else { (Get-Date).AddHours(-2) }
        if (((Get-Date) - $last).TotalSeconds -gt $StaleSec) {
            WLog "stale > ${StaleSec}s (lines=$n); killing and restarting"
            $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 3
            StartBuilder
        } else {
            WLog "ok lines=$n pid=$($procs[0].ProcessId)"
        }
    }
    Start-Sleep -Seconds $PollSec
}
WLog "watchdog exit"
