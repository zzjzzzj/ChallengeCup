param(
    [int]$Epochs = 12,
    [int]$BatchSize = 32,
    [string]$Augmentation = "none",
    [string]$RunName = "resnet18_target_baseline"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $PSScriptRoot "artifacts\target_crops\manifest.csv"
$runDir = Join-Path $PSScriptRoot "runs\$RunName"

Push-Location $projectRoot
try {
    if (-not (Test-Path $manifest)) {
        python -m target_classifier_module.prepare_crops
    }
    python -m target_classifier_module.train_classifier `
        --manifest $manifest `
        --output $runDir `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --augmentation $Augmentation
}
finally {
    Pop-Location
}
