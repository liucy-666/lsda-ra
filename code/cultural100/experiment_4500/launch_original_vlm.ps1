$root = 'D:\Python\MMDIT\AAA_Experiment\experiment_4500'
$base = Join-Path $root 'original_vlm'
$worker = Join-Path $root 'rate_base_vlm.ps1'
$taskFile = Join-Path $root 'method_vlm_task.json'
$logs = Join-Path $base 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$configs = @(
    @{Id='QWEN'; Model='qwen3-vl-235b-a22b-instruct'},
    @{Id='GEMINI'; Model='gemini-3-pro-preview'}
)
$nchunks = 6
foreach ($config in $configs) {
    foreach ($chunk in 0..($nchunks - 1)) {
        $stdout = Join-Path $logs "$($config.Id)_chunk_${chunk}.out.log"
        $stderr = Join-Path $logs "$($config.Id)_chunk_${chunk}.err.log"
        $arguments = @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker,
            '-RaterId', $config.Id, '-Model', $config.Model,
            '-ChunkIdx', "$chunk", '-NChunks', "$nchunks",
            '-BaseDir', $base, '-RunTag', 'full', '-TaskFile', $taskFile
        )
        $process = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
            -ArgumentList $arguments -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr -PassThru
        Set-Content -LiteralPath (Join-Path $logs "$($config.Id)_chunk_${chunk}.pid") `
            -Value $process.Id -Encoding ascii
    }
}
Write-Host 'launched 12 OpenLux blind-rating workers for original LSDA'
