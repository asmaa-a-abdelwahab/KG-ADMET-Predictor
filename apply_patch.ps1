param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $ProjectRoot "patch_backups\oom_shared_graph_$Timestamp"

$Files = @(
    "modeling\pring_modeling\prediction_service.py",
    "modeling\requirements.txt",
    "docker-compose.yml"
)

New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null

foreach ($RelativePath in $Files) {
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
Write-Host "Merge memory.env.snippet into your existing .env, then rebuild predictor and recreate Neo4j."
