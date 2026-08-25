param(
    [Parameter(Mandatory=$true)][string]$RaterId,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][int]$ChunkIdx,
    [Parameter(Mandatory=$true)][int]$NChunks,
    [Parameter(Mandatory=$true)][string]$BaseDir,
    [string]$RunTag = 'main',
    [string]$TaskFile = '',
    [string]$OrderFile = ''
)
$ErrorActionPreference = 'Continue'
$scriptRootResolved = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $TaskFile) { $TaskFile = Join-Path $scriptRootResolved 'base_vlm_task.json' }
$key = [Environment]::GetEnvironmentVariable('TEST_API_KEY','User')
if (-not $key) { Write-Host 'NO KEY'; exit 1 }

$task = Get-Content $TaskFile -Raw | ConvertFrom-Json
$mapRows = Get-Content (Join-Path $BaseDir 'key\blind_map.json') -Raw | ConvertFrom-Json
$map = @{}
foreach ($row in $mapRows) { $map[$row.sample_id] = $row }
if (-not $OrderFile) { $OrderFile = Join-Path $BaseDir "order_$RaterId.json" }
$parsedIds = Get-Content $OrderFile -Raw | ConvertFrom-Json
$ids = @()
foreach ($parsedId in $parsedIds) { $ids += [string]$parsedId }
$chunk = @()
for ($i=0; $i -lt $ids.Count; $i++) { if ($i % $NChunks -eq $ChunkIdx) { $chunk += $ids[$i] } }
$outDir = Join-Path $BaseDir "ratings\$RaterId"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir "ratings_${RaterId}_${RunTag}_chunk_${ChunkIdx}.jsonl.raw"
$schema = $task.output_schema | ConvertTo-Json -Depth 10 -Compress
$ok = 0; $fail = 0; $seen = 0

foreach ($sampleId in $chunk) {
    $seen++
    if (Test-Path $out) {
        if (Select-String -Path $out -SimpleMatch -Pattern ('"sample_id":"' + $sampleId + '"')) { continue }
    }
    $row = $map[$sampleId]
    $prompt = $task.instructions + "`n`nSAMPLE ID: " + $sampleId +
        "`nTARGET A: " + $row.entity_A +
        "`nA DIAGNOSTIC: " + $row.entity_A_diagnostic +
        "`nTARGET B: " + $row.entity_B +
        "`nB DIAGNOSTIC: " + $row.entity_B_diagnostic +
        "`n`nRequired JSON schema: " + $schema
    $jpg = Join-Path $BaseDir "blind\$sampleId.jpg"
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($jpg))
    $content = @(
        [ordered]@{type='text'; text=$prompt},
        [ordered]@{type='image_url'; image_url=[ordered]@{url='data:image/jpeg;base64,' + $b64}}
    )
    $body = [ordered]@{
        model=$Model; temperature=0.1; max_tokens=3500
        response_format=[ordered]@{type='json_object'}
        messages=@([ordered]@{role='user'; content=$content})
    } | ConvertTo-Json -Depth 12 -Compress
    $txt = $null; $status = ''
    for ($attempt=1; $attempt -le 3; $attempt++) {
        try {
            $resp = Invoke-RestMethod -Uri 'https://api.openlux.ai/v1/chat/completions' -Method Post `
                -Headers @{Authorization="Bearer $key"} -ContentType 'application/json' `
                -Body $body -TimeoutSec 300
            $txt = $resp.choices[0].message.content
            if ($txt -is [array]) { $txt = $txt -join ' ' }
            break
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            if (-not $status) { $status = $_.Exception.GetType().Name }
            if ($attempt -lt 3) { Start-Sleep -Seconds (5 * $attempt * $attempt) }
        }
    }
    if ($txt) { $ok++ } else { $fail++; $txt = 'MISSING' }
    $record = [ordered]@{
        sample_id=$sampleId; rater_id=$RaterId; rater_model=$Model
        raw=$txt; request_status=$status
    } | ConvertTo-Json -Compress
    [IO.File]::AppendAllText($out, $record + "`n", [Text.UTF8Encoding]::new($false))
    if ($seen % 20 -eq 0) { Write-Host "[$seen/$($chunk.Count)] ok=$ok fail=$fail" }
}
Write-Host "DONE $RaterId chunk=$ChunkIdx ok=$ok fail=$fail out=$out"
