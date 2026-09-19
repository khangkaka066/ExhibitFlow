param(
    [string]$OnnxPath = "models/yolox_s_mot17_640.onnx",
    [string]$EnginePath = "models/yolox_s_mot17_640_fp16.engine",
    [int]$WorkspaceMiB = 4096
)

$ErrorActionPreference = "Stop"

$trtexec = Get-Command trtexec.exe -ErrorAction SilentlyContinue
if (-not $trtexec) {
    $trtexec = Get-Command trtexec -ErrorAction SilentlyContinue
}
if (-not $trtexec) {
    throw "trtexec was not found. Install NVIDIA TensorRT and add its bin directory to PATH."
}

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
}
if (-not $nvidiaSmi) {
    throw "nvidia-smi was not found. Install an NVIDIA driver before building TensorRT."
}

$gpuInfo = & $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0 -or -not $gpuInfo) {
    throw "No usable NVIDIA GPU was detected by nvidia-smi."
}

if (-not (Test-Path -LiteralPath $OnnxPath)) {
    throw "ONNX model not found: $OnnxPath"
}

$engineParent = Split-Path -Parent $EnginePath
if ($engineParent -and -not (Test-Path -LiteralPath $engineParent)) {
    New-Item -ItemType Directory -Path $engineParent | Out-Null
}

Write-Host "Detected NVIDIA GPU(s): $gpuInfo"
Write-Host "Building TensorRT FP16 engine from $OnnxPath"
& $trtexec.Source `
    "--onnx=$OnnxPath" `
    "--saveEngine=$EnginePath" `
    "--fp16" `
    "--memPoolSize=workspace:$WorkspaceMiB" `
    "--skipInference"

if ($LASTEXITCODE -ne 0) {
    throw "TensorRT engine build failed with exit code $LASTEXITCODE."
}

Write-Host "TensorRT engine created: $EnginePath"
