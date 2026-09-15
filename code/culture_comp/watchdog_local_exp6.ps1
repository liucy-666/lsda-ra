# Local watchdog for EXP_6 scoring: restart score_images.py workers if they die before completing.
param(
    [string]$Root = "D:\Python\MMDIT",
    [int]$EverySec = 300
)
$exp = Join-Path $Root "experiment\2026_9_15_EXP_6_KNOW_INJECT"
$log = Join-Path $exp "logs\local_watchdog.log"
$py = "python"

function CountLines($p) { if (Test-Path $p) { (Get-Content $p | Measure-Object).Count } else { 0 } }
function RunningScore($manifest) {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -like "*score_images.py*" -and $_.CommandLine -like "*$manifest*" }).Count -gt 0
}
function StartScore($manifest, $out, $tag) {
    Start-Process -FilePath $py -ArgumentList @(
        "code\mmdit_causal\score_images.py", "--manifest", $manifest, "--out", $out,
        "--raters", "GPT54,GEMINI35", "--resize", "768", "--sleep", "1.5"
    ) -WorkingDirectory $Root -RedirectStandardOutput "$exp\logs\score_$tag.out.log" `
      -RedirectStandardError "$exp\logs\score_$tag.err.log" -WindowStyle Hidden | Out-Null
}

while ($true) {
    $t = (Get-Date).ToString("yyyy-MM-dd HH:mm")
    $nll = CountLines "$exp\ratings\ll_scores.jsonl"
    $nls = CountLines "$exp\ratings\lsdalong_scores.jsonl"
    $mll = "$exp\manifests\ll_score_manifest.jsonl"
    $mls = "$exp\manifests\lsdalong_score_manifest.jsonl"
    Add-Content $log "$t LL=$nll/300 LSDA=$nls/249" -Encoding utf8
    if ($nll -lt 300 -and -not (RunningScore "ll_score_manifest.jsonl")) {
        Add-Content $log "$t restart LL scorer ($nll/300)" -Encoding utf8
        StartScore $mll "$exp\ratings\ll_scores.jsonl" "ll"
    }
    if ($nls -lt 249 -and -not (RunningScore "lsdalong_score_manifest.jsonl")) {
        Add-Content $log "$t restart LSDA scorer ($nls/249)" -Encoding utf8
        StartScore $mls "$exp\ratings\lsdalong_scores.jsonl" "lsdalong"
    }
    if ($nll -ge 300 -and $nls -ge 249) { Add-Content $log "$t SCORING DONE" -Encoding utf8; break }
    Start-Sleep -Seconds $EverySec
}
