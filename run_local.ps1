# 本地一键扫描脚本
# 用法：
#   .\run_local.ps1                # 跑全市场（~4 分钟）
#   .\run_local.ps1 -Limit 100     # 仅扫前 100 只（快速验证）
#   .\run_local.ps1 -Stock 600519  # 单股诊断

param(
    [int]$Limit = 0,
    [string]$Stock = "",
    [string]$Name = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 关代理 + 强制 UTF-8（emoji 输出需要）
$env:HTTP_PROXY  = ""
$env:HTTPS_PROXY = ""
$env:http_proxy  = ""
$env:https_proxy = ""
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if ($Stock) {
    if ($Name) {
        python analyze_one.py $Stock --name $Name
    } else {
        python analyze_one.py $Stock
    }
} else {
    if ($Limit -gt 0) { $env:LIMIT = "$Limit" } else { $env:LIMIT = "" }
    $env:WORKERS = "10"
    python run_scan.py
}
