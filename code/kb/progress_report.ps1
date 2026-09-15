# Progress reporter: every 30 min append a concise KB-build summary to logs/progress.md
param(
    [string]$Root = "D:\Python\MMDIT",
    [int]$Target = 3546,
    [int]$EverySec = 1800
)

$kb  = Join-Path $Root "experiment\2026_9_14_EXP_4_CULTURE_KB"
$out = Join-Path $kb   "kb\kb_all.jsonl"
$plog = Join-Path $kb  "logs\progress.md"
$blog = Join-Path $kb  "logs\kb_build.out.log"
$wpid = Join-Path $kb  "logs\watchdog_kb.pid"

function Report() {
    $lines = if (Test-Path $out) { Get-Content $out -ErrorAction SilentlyContinue } else { @() }
    $n = @($lines).Count
    $wiki = 0; $ai = 0; $err = 0
    foreach ($l in $lines) {
        if (-not $l.Trim()) { continue }
        try { $r = $l | ConvertFrom-Json } catch { continue }
        if ($r.source -eq "wikidata") { $wiki++ } elseif ($r.source -eq "llm_fallback") { $ai++ }
        if ($r.error) { $err++ }
    }
    $rate = if ($n -gt 0) { [math]::Round(100.0 * $wiki / $n, 1) } else { 0 }
    $bp = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*build_kb_fast.py*" })
    $wp = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*watchdog_kb.ps1*" })
    $lastlog = if (Test-Path $blog) { (Get-Content $blog -Tail 1 -ErrorAction SilentlyContinue) } else { "" }
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $status = if ($n -ge $Target) { "DONE" } elseif ($bp.Count -gt 0) { "building" } else { "STOPPED" }
    $md = "`n## $stamp  [$status]`n- built: $n / $Target`n- wikidata: $wiki   llm_fallback: $ai   errors: $err   wiki_rate: $rate%`n- builder: $(if($bp.Count){'alive pid='+$bp[0].ProcessId}else{'DEAD'})   watchdog: $(if($wp.Count){'alive'}else{'DEAD'})`n- last: $lastlog`n"
    Add-Content -Path $plog -Value $md -Encoding utf8
}

Report
while ($true) {
    Start-Sleep -Seconds $EverySec
    Report
}
