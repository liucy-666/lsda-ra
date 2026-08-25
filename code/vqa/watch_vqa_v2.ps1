param(
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'
$env:TEST_API_KEY = [Environment]::GetEnvironmentVariable('TEST_API_KEY', 'User')
if (-not $env:TEST_API_KEY) { throw 'TEST_API_KEY is not available in the user environment.' }
$Root = 'D:\Python\MMDIT\experiment\2026_8_25_EXP_1\cultural100_records\experiment_4500\binary_vqa_v2'
$Worker = 'D:\Python\MMDIT\code\lsda\rate_binary_vqa_v2.py'
$StateDir = Join-Path $Root 'watchdog'
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

$configs = @(
    @{ Rater='GEMINI'; Model='gemini-3.5-flash-lite'; Chunks=2 },
    @{ Rater='QWEN'; Model='qwen3-vl-235b-a22b-instruct'; Chunks=6 }
)

function Get-ValidCount([string]$Rater) {
    $seen = @{}
    $dir = Join-Path (Join-Path $Root 'ratings') $Rater
    Get-ChildItem -LiteralPath $dir -Filter 'ratings_*.jsonl' -File -ErrorAction SilentlyContinue | ForEach-Object {
        Get-Content -LiteralPath $_.FullName -Encoding UTF8 | ForEach-Object {
            try {
                $row = $_ | ConvertFrom-Json
                if ($row.eval_id) { $seen[$row.eval_id] = $true }
            } catch {}
        }
    }
    return $seen.Count
}

function Test-Worker([string]$Rater, [int]$Chunk) {
    $pidFile = Join-Path $StateDir ("worker_{0}_{1:00}.pid" -f $Rater,$Chunk)
    if (-not (Test-Path -LiteralPath $pidFile)) { return $false }
    $workerPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    return $null -ne (Get-Process -Id $workerPid -ErrorAction SilentlyContinue)
}

function Start-Worker($Config, [int]$Chunk) {
    $rater = $Config.Rater
    $logDir = Join-Path (Join-Path $Root 'logs') $rater
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $out = Join-Path $logDir ("watch_{0}_{1:00}_{2}.out.log" -f $rater,$Chunk,$stamp)
    $err = Join-Path $logDir ("watch_{0}_{1:00}_{2}.err.log" -f $rater,$Chunk,$stamp)
    $args = @(
        $Worker, '--rater', $rater, '--model', $Config.Model,
        '--chunk', $Chunk, '--nchunks', $Config.Chunks
    )
    $proc = Start-Process -FilePath 'python' -ArgumentList $args -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $pidFile = Join-Path $StateDir ("worker_{0}_{1:00}.pid" -f $rater,$Chunk)
    Set-Content -LiteralPath $pidFile -Value $proc.Id -Encoding ASCII
}

while ($true) {
    $snapshot = [ordered]@{ timestamp=(Get-Date).ToString('o') }
    $allDone = $true
    foreach ($config in $configs) {
        $count = Get-ValidCount $config.Rater
        $snapshot[$config.Rater] = $count
        if ($count -lt 1800) {
            $allDone = $false
            for ($chunk=0; $chunk -lt $config.Chunks; $chunk++) {
                if (-not (Test-Worker $config.Rater $chunk)) {
                    Start-Worker $config $chunk
                }
            }
        }
    }
    $snapshot | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $StateDir 'status.json') -Encoding UTF8
    if ($allDone) { break }
    Start-Sleep -Seconds $PollSeconds
}
