# EXP_6 objective metrics chain (texture/dino/siglip) for the new arms, with skip-if-done.
$ErrorActionPreference = "Continue"
Set-Location "D:\Python\MMDIT"
$cls = "experiment\2026_9_12_EXP_3_KA_ME\analysis\classification.json"
$base = "data\KNOW_INJECT\2026_9_15_EXP_6"
$refs = "$base\refs_only"
$empty = "$base\empty_lsda"
$r = "experiment\2026_9_15_EXP_6_KNOW_INJECT\ratings"
$log = "experiment\2026_9_15_EXP_6_KNOW_INJECT\logs\metrics_chain.log"

function CountLines($p) { if (Test-Path $p) { (Get-Content $p | Measure-Object).Count } else { 0 } }
function Log($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) $m" | Out-File -Append -Encoding utf8 $log }

foreach ($arm in @("LSDA_Long", "LL")) {
    $dir = "$base\arm_dirs\$arm"
    $expected = if ($arm -eq "LSDA_Long") { 249 } else { 300 }

    if ((CountLines "$r\metric_texture_$arm.jsonl") -lt $expected) {
        Log "texture $arm start"
        python code\metrics\metric_texture.py --classification $cls --census-dir $refs --lsda-dir $empty --binding-dir $dir --out "$r\metric_texture_$arm.jsonl" *>> $log
        Log "texture $arm end ($(CountLines "$r\metric_texture_$arm.jsonl"))"
    }
    if ((CountLines "$r\metric_dino_$arm.jsonl") -lt $expected) {
        Log "dino $arm start"
        python code\metrics\metric_dino.py --classification $cls --census-dir $refs --lsda-dir $empty --binding-dir $dir --arms Binding --out "$r\metric_dino_$arm.jsonl" *>> $log
        Log "dino $arm end ($(CountLines "$r\metric_dino_$arm.jsonl"))"
    }
    if ((CountLines "$r\metric_siglip_$arm.jsonl") -lt $expected) {
        Log "siglip $arm start"
        python code\metrics\score_siglip_arms.py --classification $cls --census-dir $refs --lsda-dir $empty --binding-dir $dir --arms Binding --dtype bfloat16 --out "$r\metric_siglip_$arm.jsonl" *>> $log
        Log "siglip $arm end ($(CountLines "$r\metric_siglip_$arm.jsonl"))"
    }
}
Log "ALL METRICS DONE"
