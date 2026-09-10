# ============================================================
# Run the automated LSDA evaluation pipeline (after setup_env.ps1).
# Keep this file pure ASCII (Windows PowerShell 5.1 ANSI parsing).
# Steps: SAM masks pilot -> full -> binding scores -> analysis.
# Run with: powershell -ExecutionPolicy Bypass -File run_eval.ps1
# ============================================================
$ErrorActionPreference = 'Continue'
$root = 'D:\Python\MMDIT'
$pyexe = 'D:\Python3.13\python.exe'   # Python 3.13 has torch 2.11.0+cu128
$wt = "$root\.venv\_tmp"
New-Item -ItemType Directory -Force -Path $wt | Out-Null
$env:TMP = $wt; $env:TEMP = $wt
$env:HF_HOME = "$root\.hf_cache"
$env:TORCH_HOME = "$root\.torch_cache"
$env:PYTHONPATH = "$root\pyenv"

function Step { param([string]$m) Write-Output ("=== " + $m + " ===") }

Step '1/5 SAM masks pilot (20 images per condition)'
& $pyexe -X utf8 "$root\code\metrics\sam_masks.py" --limit 20

Step '2/5 (MANUAL QC) review QC overlays, then rerun without --limit for full masks'
Write-Output 'Inspect: D:\Python\MMDIT\experiment\2026_8_31_EXP_2\segmentation\masks_*\*_qc.png'

Step '3/5 full SAM masks'
& $pyexe -X utf8 "$root\code\metrics\sam_masks.py"

Step '4/5 binding scores (SigLIP2 masked-maxcos + CLIP-I/T + DINO + BLIP-VQA)'
& $pyexe -X utf8 "$root\code\metrics\score_binding.py"

Step '5/5 unified analysis + report'
& $pyexe -X utf8 "$root\code\metrics\analyze_unified.py"

Write-Output 'DONE. Report: D:\Python\MMDIT\experiment\2026_8_31_EXP_2\report\UNIFIED_REPORT.md'
