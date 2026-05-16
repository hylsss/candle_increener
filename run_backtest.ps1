# 本地一键回测脚本
# 用法：
#   .\run_backtest.ps1                          # 默认 500 票 × 120 日（~30-50 分钟）
#   .\run_backtest.ps1 -Sample 200 -Days 60     # 快速版 ~10 分钟
#   .\run_backtest.ps1 -Sample 1000 -Days 250   # 完整版 ~2 小时

param(
    [int]$Sample = 500,
    [int]$Days = 120,
    [string]$HoldDays = "5,10,20",
    [int]$Workers = 8
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$env:HTTP_PROXY  = ""
$env:HTTPS_PROXY = ""
$env:http_proxy  = ""
$env:https_proxy = ""
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:BT_SAMPLE_SIZE   = "$Sample"
$env:BT_LOOKBACK_DAYS = "$Days"
$env:BT_HOLD_DAYS     = "$HoldDays"
$env:BT_WORKERS       = "$Workers"

python backtest.py
