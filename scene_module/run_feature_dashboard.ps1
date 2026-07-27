$ErrorActionPreference = "Stop"
$moduleDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "正在启动遥感场景特征分析台..."
Write-Host "页面地址：http://127.0.0.1:8501"
python (Join-Path $moduleDir "feature_web_app.py")
