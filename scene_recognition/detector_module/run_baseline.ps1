param(
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [string]$RunName = "yolov8n_baseline_v1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $projectRoot

try {
    python -m scene_recognition.detector_module.prepare_detection_dataset
    python -m scene_recognition.detector_module.create_incremental_protocol
    python -m scene_recognition.detector_module.train_detector `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --name $RunName `
        --exist-ok

    $runDir = Join-Path $projectRoot "scene_recognition\detector_module\runs\$RunName"
    python -m scene_recognition.detector_module.select_baseline_checkpoint --run $runDir
    $selectedModel = Join-Path $runDir "weights\submission_map50.pt"
    python -m scene_recognition.detector_module.evaluate_detector --model $selectedModel
    python -m scene_recognition.detector_module.export_detector --model $selectedModel
}
finally {
    Pop-Location
}
