$root = 'D:\Python\MMDIT\AAA_Experiment\experiment_4500'
$base = Join-Path $root 'base_vlm'
$worker = Join-Path $root 'rate_base_vlm.ps1'
$logs = Join-Path $base 'logs'
$configs = @(
    @{Id='QWEN'; Model='qwen3-vl-235b-a22b-instruct'},
    @{Id='GEMINI'; Model='gemini-3-pro-preview'}
)
foreach ($config in $configs) {
    $oldPidFile = Join-Path $logs "$($config.Id)_chunk_0.pid"
    if (Test-Path $oldPidFile) {
        $oldProcessId = [int](Get-Content $oldPidFile)
        Stop-Process -Id $oldProcessId -Force -ErrorAction SilentlyContinue
    }
    $stdout = Join-Path $logs "$($config.Id)_chunk_0.out.log"
    $stderr = Join-Path $logs "$($config.Id)_chunk_0.err.log"
    $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`" " +
        "-RaterId $($config.Id) -Model $($config.Model) " +
        "-ChunkIdx 0 -NChunks 6 -BaseDir `"$base`" -RunTag full"
    $process = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden `
        -ArgumentList $argLine -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath $oldPidFile -Value $process.Id -Encoding ascii
}
Write-Host 'restarted corrected chunk 0 workers'
