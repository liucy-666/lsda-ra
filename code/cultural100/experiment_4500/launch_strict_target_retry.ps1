$root = 'D:\Python\MMDIT\AAA_Experiment\experiment_4500'
$worker = Join-Path $root 'rate_base_vlm.ps1'
$task = Join-Path $root 'method_vlm_task.json'
$jobs = @(
    @{Dataset='original_vlm'; Rater='QWEN'; Model='qwen3-vl-235b-a22b-instruct'},
    @{Dataset='original_vlm'; Rater='GEMINI'; Model='gemini-3-pro-preview'},
    @{Dataset='ra_vlm'; Rater='QWEN'; Model='qwen3-vl-235b-a22b-instruct'},
    @{Dataset='ra_vlm'; Rater='GEMINI'; Model='gemini-3-pro-preview'}
)
foreach ($job in $jobs) {
    $base = Join-Path $root $job.Dataset
    $logs = Join-Path $base 'logs'
    $order = Join-Path $base ("strict_target_retry_" + $job.Rater + '.json')
    $stdout = Join-Path $logs ("strict_target_" + $job.Rater + '.out.log')
    $stderr = Join-Path $logs ("strict_target_" + $job.Rater + '.err.log')
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker,
        '-RaterId', $job.Rater, '-Model', $job.Model,
        '-ChunkIdx', '0', '-NChunks', '1', '-BaseDir', $base,
        '-RunTag', 'a_strict_target', '-TaskFile', $task, '-OrderFile', $order
    )
    $process = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
        -ArgumentList $arguments -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $logs ("strict_target_" + $job.Rater + '.pid')) `
        -Value $process.Id -Encoding ascii
    Write-Host ($job.Dataset + ' ' + $job.Rater + ' pid=' + $process.Id)
}
