# Local monitor for EXP_6: pull the server watchdog log line every 15 min into a local file.
param(
    [string]$Root = "D:\Python\MMDIT",
    [string]$SshHost = "A100",
    [int]$EverySec = 900
)
$exp = Join-Path $Root "experiment\2026_9_15_EXP_6_KNOW_INJECT"
$log = Join-Path $exp "logs\monitor.md"
$rlog = "/science/wx/pry/MMDIT/experiments/2026_9_15_EXP_6_KNOW_INJECT/watchdog.log"
while ($true) {
    $line = ""
    try { $line = (ssh -o ConnectTimeout=15 $SshHost "tail -1 $rlog" 2>$null | Select-Object -Last 1) } catch { $line = "ssh_error" }
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm")
    Add-Content -Path $log -Value "## $stamp`n- $line`n" -Encoding utf8
    if ("$line" -match "LL=300/300" -and "$line" -match "LSDA=250/250") {
        Add-Content -Path $log -Value "ALL DONE`n" -Encoding utf8
        break
    }
    Start-Sleep -Seconds $EverySec
}
