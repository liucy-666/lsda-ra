# ============================================================
# MMDIT local evaluation environment setup.
# Run from a normal PowerShell (NOT sandboxed):
#   powershell -ExecutionPolicy Bypass -File D:\Python\MMDIT\code\metrics\setup_env.ps1
# Log: experiment\2026_8_31_EXP_2\logs\setup_env.log
# NOTE: keep this file pure ASCII (Windows PowerShell 5.1 parses
# BOM-less .ps1 as ANSI and Chinese text breaks string parsing).
# ============================================================
$ErrorActionPreference = 'Continue'
$log = 'D:\Python\MMDIT\experiment\2026_8_31_EXP_2\logs\setup_env.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log { param([string]$m) $ts = Get-Date -Format 'HH:mm:ss'; ("[$ts] " + $m) | Tee-Object -FilePath $log -Append }

Log '=== [1/3] pip install torch+torchvision (cu128 for RTX 5060 Blackwell) ==='
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) {
    Log 'torch direct failed (hash/network) -> retry with Aliyun mirror'
    python -m pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128 2>&1 | Tee-Object -FilePath $log -Append
}

Log '=== [2/3] pip install other deps (transformers/timm/open_clip/SAM/hf_hub) ==='
python -m pip install transformers timm open_clip_torch sentencepiece safetensors scikit-image opencv-python-headless segment-anything huggingface_hub requests 2>&1 | Tee-Object -FilePath $log -Append

Log '=== [3/3] download model weights to D:\Python\MMDIT\models (~9GB, auto hf-mirror fallback) ==='
python 'D:\Python\MMDIT\code\metrics\download_models.py' --out 'D:\Python\MMDIT\models' 2>&1 | Tee-Object -FilePath $log -Append

Log '=== verify ==='
python -c "import torch, transformers, huggingface_hub, timm; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('transformers', transformers.__version__)" 2>&1 | Tee-Object -FilePath $log -Append
Get-ChildItem 'D:\Python\MMDIT\models' -Directory -ErrorAction SilentlyContinue | ForEach-Object { Log ("model dir: " + $_.Name) }
Log '=== DONE. Send me the last 20 lines of setup_env.log ==='
