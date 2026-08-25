$root = 'D:\Python\MMDIT\AAA_Experiment\experiment_4500'
$base = Join-Path $root 'ra_vlm'
$worker = Join-Path $root 'rate_base_vlm.ps1'
$task = Join-Path $root 'method_vlm_task.json'
$logs = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$jobs = @(
    @{Rater='QWEN'; Model='qwen3-vl-235b-a22b-instruct'},
    @{Rater='GEMINI'; Model='gemini-3-pro-preview'}
)
foreach ($job in $jobs) {
    $stdout = Join-Path $logs ("stable_" + $job.Rater + '.out.log')
    $stderr = Join-Path $logs ("stable_" + $job.Rater + '.err.log')
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker,
        '-RaterId', $job.Rater, '-Model', $job.Model,
        '-ChunkIdx', '0', '-NChunks', '1', '-BaseDir', $base,
        '-RunTag', 'a_stable1', '-TaskFile', $task
    )
    $process = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
        -ArgumentList $arguments -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $logs ("stable_" + $job.Rater + '.pid')) `
        -Value $process.Id -Encoding ascii
    Write-Host ($job.Rater + ' pid=' + $process.Id)
}
