$root = 'D:\Python\MMDIT\AAA_Experiment\experiment_4500'
$base = Join-Path $root 'base_vlm'
$worker = Join-Path $root 'rate_base_vlm.ps1'
$logs = Join-Path $base 'logs'
$configs = @(
    @{Id='QWEN'; Model='qwen3-vl-235b-a22b-instruct'},
    @{Id='GEMINI'; Model='gemini-3-pro-preview'}
)
$nchunks = 6
foreach ($config in $configs) {
    foreach ($chunk in 1..5) {
        $stdout = Join-Path $logs "$($config.Id)_chunk_${chunk}.out.log"
        $stderr = Join-Path $logs "$($config.Id)_chunk_${chunk}.err.log"
        $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`" " +
            "-RaterId $($config.Id) -Model $($config.Model) " +
            "-ChunkIdx $chunk -NChunks $nchunks -BaseDir `"$base`" -RunTag full"
        $process = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
            -ArgumentList $argLine -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr -PassThru
        Set-Content -LiteralPath (Join-Path $logs "$($config.Id)_chunk_${chunk}.pid") `
            -Value $process.Id -Encoding ascii
    }
}
Write-Host 'launched remaining 10 OpenLux workers'
