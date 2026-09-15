param(
    [string]$ProjectRoot = "",
    [string]$CondaEnv = "zzjDEREnv",
    [string]$Python = "python",
    [string]$DataRoot = "",
    [string]$DataYaml = "",
    [string]$Classes = "",
    [string]$Workspace = "",
    [string]$InitialModel = "",
    [string]$RunName = "base_before_r1",
    [int]$Epochs = 30,
    [int]$Patience = 10,
    [int]$ImageSize = 640,
    [int]$BatchSize = 8,
    [int]$Workers = 4,
    [string]$Device = "0",
    [int]$Seed = 42,
    [int]$ExportImageSize = 640,
    [int]$ExportOpset = 12,
    [switch]$ReuseAugmented,
    [switch]$RebuildAugmented,
    [switch]$BuiltinAug,
    [switch]$NoAmp,
    [switch]$NoPlots,
    [switch]$NoExportOnnx,
    [switch]$ExistOk,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"

function FullPathUnderProject([string]$PathText, [string]$Root) {
    if ([string]::IsNullOrWhiteSpace($PathText)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return [System.IO.Path]::GetFullPath($PathText)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $PathText))
}

function Invoke-Step([string]$Exe, [string[]]$ArgsList) {
    Write-Host ("[INFO] " + $Exe + " " + ($ArgsList -join " "))
    & $Exe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $LASTEXITCODE, $Exe)
    }
}

function Add-ExistingPython([System.Collections.Generic.List[string]]$List, [string]$Candidate) {
    if (-not [string]::IsNullOrWhiteSpace($Candidate) -and (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        $full = [System.IO.Path]::GetFullPath($Candidate)
        if (-not $List.Contains($full)) {
            $List.Add($full) | Out-Null
        }
    }
}

function Resolve-CondaPython([string]$EnvName) {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        Add-ExistingPython $candidates (Join-Path $env:CONDA_PREFIX "python.exe")
    }

    $json = $null
    try {
        $json = (& conda --no-plugins env list --json 2>$null) -join "`n"
    } catch {
        $json = $null
    }
    if (-not [string]::IsNullOrWhiteSpace($json)) {
        try {
            $envList = $json | ConvertFrom-Json
            foreach ($envPath in $envList.envs) {
                if ((Split-Path -Path $envPath -Leaf) -eq $EnvName) {
                    Add-ExistingPython $candidates (Join-Path $envPath "python.exe")
                }
            }
        } catch {
        }
    }

    Add-ExistingPython $candidates "D:\000zzjTools\AnacondaEnvs\envs\$EnvName\python.exe"
    Add-ExistingPython $candidates "D:\000zzjTools\Anaconda\envs\$EnvName\python.exe"
    Add-ExistingPython $candidates (Join-Path $HOME ".conda\envs\$EnvName\python.exe")
    Add-ExistingPython $candidates (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    Add-ExistingPython $candidates (Join-Path $ProjectRoot ".venv-yolo\Scripts\python.exe")
    Add-ExistingPython $candidates (Join-Path $ProjectRoot "数据集（不上传git）.venv-yolo\Scripts\python.exe")
    Add-ExistingPython $candidates (Join-Path $ProjectRoot "数据集（不上传git）\.venv-yolo\Scripts\python.exe")

    foreach ($candidate in $candidates) {
        return $candidate
    }
    return ""
}

function Test-TrainingPython([string]$Exe) {
    $probe = @"
import importlib.util
import sys

missing = [name for name in ("torch", "ultralytics", "yaml") if importlib.util.find_spec(name) is None]
print(sys.executable)
if missing:
    print("MISSING=" + ",".join(missing))
    raise SystemExit(3)
import torch
import ultralytics
print("torch=" + torch.__version__)
print("ultralytics=" + ultralytics.__version__)
"@
    & $Exe -c $probe
    return $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
} else {
    $ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
}
$ScriptDir = [System.IO.Path]::GetFullPath($PSScriptRoot)

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $ProjectRoot "data\datasets_r1_base_train"
} else {
    $DataRoot = FullPathUnderProject $DataRoot $ProjectRoot
}
if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = Join-Path $ProjectRoot "outputs\ascend310bpro_before_pc"
} else {
    $Workspace = FullPathUnderProject $Workspace $ProjectRoot
}
if ([string]::IsNullOrWhiteSpace($InitialModel)) {
    $InitialModel = Join-Path $ScriptDir "models\yolo26n.pt"
} else {
    $InitialModel = FullPathUnderProject $InitialModel $ProjectRoot
}
if ([string]::IsNullOrWhiteSpace($Classes)) {
    $candidate = Join-Path $DataRoot "classes.txt"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $Classes = $candidate
    } else {
        $Classes = Join-Path $ScriptDir "classes_base4.txt"
    }
} else {
    $Classes = FullPathUnderProject $Classes $ProjectRoot
}
if (-not [string]::IsNullOrWhiteSpace($DataYaml)) {
    $DataYaml = FullPathUnderProject $DataYaml $ProjectRoot
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root not found: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $InitialModel -PathType Leaf)) {
    throw "Initial model not found: $InitialModel"
}
if (-not (Test-Path -LiteralPath $Classes -PathType Leaf)) {
    throw "Classes file not found: $Classes"
}
if ([string]::IsNullOrWhiteSpace($DataYaml) -and -not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Base dataset root not found: $DataRoot"
}
if ($SmokeTest) {
    $Epochs = 1
    $Patience = 1
}

Set-Location $ProjectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda command not found. Open an Anaconda/Miniconda PowerShell first."
}
Write-Host "[INFO] Activating conda env: $CondaEnv"
conda activate $CondaEnv
if ($LASTEXITCODE -ne 0) {
    throw "conda activate failed: $CondaEnv"
}

if ($Python -eq "python") {
    $resolvedPython = Resolve-CondaPython $CondaEnv
    if (-not [string]::IsNullOrWhiteSpace($resolvedPython)) {
        $Python = $resolvedPython
    }
}
Write-Host "[INFO] Python executable: $Python"

$probeExit = Test-TrainingPython $Python
if ($probeExit -ne 0) {
    throw @"
Selected Python cannot import the training dependencies.

Conda env requested : $CondaEnv
Selected Python     : $Python

Fix one of these:
  1. Install python/torch/ultralytics/yaml into $CondaEnv.
  2. Pass -Python "C:\path\to\env\python.exe" for an environment that already has them.

For example:
  .\deployment\ascend310bpro\train_before_model_pc.ps1 -Python "I:\项目\ChallengeCup\数据集（不上传git）.venv-yolo\Scripts\python.exe" -SmokeTest
"@
}

New-Item -ItemType Directory -Force -Path $Workspace | Out-Null

if ([string]::IsNullOrWhiteSpace($DataYaml)) {
    $AugDir = Join-Path $Workspace "base_before_augmented"
    $DataYaml = Join-Path $AugDir "data.yaml"
    if ((Test-Path -LiteralPath $DataYaml -PathType Leaf) -and $ReuseAugmented -and -not $RebuildAugmented) {
        Write-Host "[INFO] Reuse augmented dataset: $DataYaml"
    } else {
        if ((Test-Path -LiteralPath $AugDir) -and -not $RebuildAugmented -and -not $ReuseAugmented) {
            throw "Augmented directory already exists: $AugDir. Pass -ReuseAugmented or -RebuildAugmented."
        }
        $augmentArgs = @(
            "train.py", "ascend310b-augment",
            "--dataset-root", $DataRoot,
            "--split", "train",
            "--output", $AugDir,
            "--classes", $Classes,
            "--include-original"
        )
        if ($RebuildAugmented) {
            $augmentArgs += "--force"
        }
        Invoke-Step $Python $augmentArgs
    }
}

if (-not (Test-Path -LiteralPath $DataYaml -PathType Leaf)) {
    throw "Training data.yaml not found: $DataYaml"
}

$RunRoot = Join-Path $Workspace "runs"
$trainArgs = @(
    "train.py", "yolo",
    "--data", $DataYaml,
    "--model", $InitialModel,
    "--epochs", [string]$Epochs,
    "--patience", [string]$Patience,
    "--image-size", [string]$ImageSize,
    "--batch-size", [string]$BatchSize,
    "--workers", [string]$Workers,
    "--device", $Device,
    "--seed", [string]$Seed,
    "--project", $RunRoot,
    "--name", $RunName,
    "--eval-split", "val"
)
if (-not $BuiltinAug) {
    $trainArgs += "--no-builtin-aug"
}
if ($NoAmp) {
    $trainArgs += "--no-amp"
}
if ($NoPlots) {
    $trainArgs += "--no-plots"
}
if ($ExistOk) {
    $trainArgs += "--exist-ok"
}

Invoke-Step $Python $trainArgs

$BestPt = Join-Path $RunRoot (Join-Path $RunName "weights\best.pt")
if (-not (Test-Path -LiteralPath $BestPt -PathType Leaf)) {
    throw "best.pt not found after training: $BestPt"
}

$ExportedOnnx = $null
if (-not $NoExportOnnx) {
    $ExportDir = Join-Path $Workspace "exports"
    New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null
    $exportArgs = @(
        "scene_recognition\detector_module\export_detector.py",
        "--model", $BestPt,
        "--data", $DataYaml,
        "--output", $ExportDir,
        "--image-size", [string]$ExportImageSize,
        "--opset", [string]$ExportOpset,
        "--device", $Device,
        "--skip-validation"
    )
    Invoke-Step $Python $exportArgs
    $RawOnnx = Join-Path $ExportDir "detector_yolov8n_bs1.onnx"
    if (-not (Test-Path -LiteralPath $RawOnnx -PathType Leaf)) {
        throw "ONNX export finished but expected file was not found: $RawOnnx"
    }
    $ExportedOnnx = Join-Path $ExportDir ("before_increment_4class_{0}x{0}.onnx" -f $ExportImageSize)
    Copy-Item -LiteralPath $RawOnnx -Destination $ExportedOnnx -Force
}

$MetaPath = Join-Path $Workspace "before_model_paths.json"
$meta = [ordered]@{
    project_root = $ProjectRoot
    conda_env = $CondaEnv
    data_yaml = $DataYaml
    classes = $Classes
    initial_model = $InitialModel
    before_model_pt = $BestPt
    before_model_onnx = $ExportedOnnx
    run_dir = (Join-Path $RunRoot $RunName)
    note = "Use before_model_pt as the initial checkpoint for board-side incremental training. Use before_model_onnx only for NPU inference/ATC conversion."
}
$meta | ConvertTo-Json -Depth 6 | Out-File -FilePath $MetaPath -Encoding utf8

Write-Host "[INFO] Before-model checkpoint: $BestPt"
if ($ExportedOnnx) {
    Write-Host "[INFO] Before-model ONNX      : $ExportedOnnx"
}
Write-Host "[INFO] Metadata               : $MetaPath"
