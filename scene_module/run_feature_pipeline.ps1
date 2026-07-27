param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

python "$Root\analyze_and_prepare.py" --dataset $Dataset --output "$Root\artifacts"
python "$Root\feature_engineering.py" extract --index "$Root\artifacts\scene_index.csv" --output "$Root\artifacts\scene_features.csv"
python "$Root\feature_engineering.py" evaluate --features "$Root\artifacts\scene_features.csv" --output "$Root\runs\feature_baseline"

Write-Host "Completed. Read: $Root\特征工程实测报告.md"
