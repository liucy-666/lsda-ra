$root = 'D:\Python\MMDIT\AAA_Experiment\experiment_4500'
$worker = Join-Path $root 'rate_base_vlm.ps1'
$jobs = @(
    @{Dataset='base_vlm';     Rater='QWEN';   Model='qwen3-vl-235b-a22b-instruct'; Task='base_vlm_task.json'},
    @{Dataset='base_vlm';     Rater='GEMINI'; Model='gemini-3-pro-preview';         Task='base_vlm_task.json'},
    @{Dataset='original_vlm'; Rater='QWEN';   Model='qwen3-vl-235b-a22b-instruct'; Task='method_vlm_task.json'},
    @{Dataset='original_vlm'; Rater='GEMINI'; Model='gemini-3-pro-preview';         Task='method_vlm_task.json'},
    @{Dataset='ra_vlm';       Rater='QWEN';   Model='qwen3-vl-235b-a22b-instruct'; Task='method_vlm_task.json'},
    @{Dataset='ra_vlm';       Rater='GEMINI'; Model='gemini-3-pro-preview';         Task='method_vlm_task.json'}
)

foreach ($job in $jobs) {
    $base = Join-Path $root $job.Dataset
    $logs = Join-Path $base 'logs'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $order = Join-Path $base ("retry_order_" + $job.Rater + '.json')
    $task = Join-Path $root $job.Task
    $tag = 'a_binary_full'
    $stdout = Join-Path $logs ("binary_full_" + $job.Rater + '.out.log')
    $stderr = Join-Path $logs ("binary_full_" + $job.Rater + '.err.log')
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker,
        '-RaterId', $job.Rater, '-Model', $job.Model,
        '-ChunkIdx', '0', '-NChunks', '1', '-BaseDir', $base,
        '-RunTag', $tag, '-TaskFile', $task, '-OrderFile', $order
    )
    $process = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
        -ArgumentList $arguments -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $logs ("binary_full_" + $job.Rater + '.pid')) `
        -Value $process.Id -Encoding ascii
    Write-Host ($job.Dataset + ' ' + $job.Rater + ' pid=' + $process.Id)
}
