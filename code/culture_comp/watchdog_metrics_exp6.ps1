# Watchdog for the EXP_6 objective-metrics chain: restart if it dies before completion.
$exp = "D:\Python\MMDIT\experiment\2026_9_15_EXP_6_KNOW_INJECT"
$log = "$exp\logs\metrics_watchdog.log"
while ($true) {
    $txt = ""
    if (Test-Path "$exp\logs\metrics_chain.log") { $txt = Get-Content "$exp\logs\metrics_chain.log" -Raw }
    if ($txt -match "ALL METRICS DONE") { Add-Content $log "$(Get-Date -Format s) DONE"; break }
    $metricProc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "metric_texture|metric_dino|score_siglip" }).Count
    $chainProc = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*run_metrics_exp6.ps1*" }).Count
    if ($metricProc -eq 0 -and $chainProc -eq 0) {
        Add-Content $log "$(Get-Date -Format s) restart metrics chain"
        Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", "code\culture_comp\run_metrics_exp6.ps1" -WorkingDirectory "D:\Python\MMDIT" `
            -RedirectStandardOutput "$exp\logs\metrics_chain.console.log" `
            -RedirectStandardError "$exp\logs\metrics_chain.err.log" -WindowStyle Hidden | Out-Null
    } else {
        Add-Content $log "$(Get-Date -Format s) ok metricProcs=$metricProc chain=$chainProc"
    }
    Start-Sleep -Seconds 600
}
