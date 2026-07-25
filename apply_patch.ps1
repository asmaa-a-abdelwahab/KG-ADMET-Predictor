param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $false)]
    [string]$PatchRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$PatchRoot = (Resolve-Path $PatchRoot).Path

$RelativeFiles = @(
    "modeling\Dockerfile",
    "modeling\requirements.txt",
    "modeling\pring_modeling\live_prediction.py",
    "modeling\pring_modeling\prediction_service.py",
    "modeling\pring_modeling\prediction_api.py",
    "modeling\pring_modeling\pyg_runtime.py",
    "streamlit\app.py",
    "streamlit\utils\ui_utils.py",
    "streamlit\utils\prediction_ui.py",
    "streamlit\utils\prediction_client.py",
    "docker-compose.yml",
    "docker-compose.production.yml"
)

if ($ProjectRoot.TrimEnd('\') -eq $PatchRoot.TrimEnd('\')) {
    Write-Host "PatchRoot and ProjectRoot are the same directory."
    Write-Host "The patch was extracted directly into the repository, so no copy operation is required."
    Write-Host "Verify the files and merge hybrid_prediction.env.snippet into .env."
    exit 0
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $ProjectRoot "patch_backups\hybrid_prediction_$Timestamp"
New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null

foreach ($RelativePath in $RelativeFiles) {
    $Source = Join-Path $PatchRoot $RelativePath
    $Destination = Join-Path $ProjectRoot $RelativePath
    $Backup = Join-Path $BackupRoot $RelativePath

    if (-not (Test-Path $Source)) {
        throw "Patch file is missing: $Source"
    }

    if (Test-Path $Destination) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Backup) -Force | Out-Null
        Copy-Item $Destination $Backup -Force
    }

    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item $Source $Destination -Force
    Write-Host "Replaced $RelativePath"
}

Write-Host ""
Write-Host "Backup created at: $BackupRoot"
Write-Host "Merge hybrid_prediction.env.snippet into the existing .env before rebuilding."
