param(
    [Parameter(Mandatory = $true)][string]$PatchRoot,
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$PatchRoot = (Resolve-Path $PatchRoot).Path
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
if ($PatchRoot.TrimEnd('\') -eq $ProjectRoot.TrimEnd('\')) {
    throw "PatchRoot and ProjectRoot must be different directories."
}

$Files = @(
    "modeling\pring_modeling\live_prediction.py",
    "modeling\pring_modeling\prediction_service.py",
    "modeling\pring_modeling\prediction_store.py",
    "modeling\scripts\build_stage3_fallback_bundle.py",
    "artifacts\models\production\production_stage3_fallback.joblib",
    "artifacts\models\production\stage3_fallback_manifest.json",
    "docker-compose.yml",
    "docker-compose.production.yml",
    "parity_stage1_fallback.env.snippet",
    "diagnose_parity_fix.ps1",
    "README_PARITY_STAGE1_FALLBACK_FIX.md"
)

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $ProjectRoot "patch_backups\parity_stage1_fallback_$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

foreach ($RelativePath in $Files) {
    $Source = Join-Path $PatchRoot $RelativePath
    $Destination = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $Source)) { throw "Missing patch file: $Source" }
    if (Test-Path $Destination) {
        $Backup = Join-Path $BackupRoot $RelativePath
        New-Item -ItemType Directory -Force -Path (Split-Path $Backup -Parent) | Out-Null
        Copy-Item $Destination $Backup -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
    Copy-Item $Source $Destination -Force
    Write-Host "Replaced $RelativePath"
}

Write-Host "Backup created at $BackupRoot"
Write-Host "Merge parity_stage1_fallback.env.snippet into .env, then rebuild predictor."
