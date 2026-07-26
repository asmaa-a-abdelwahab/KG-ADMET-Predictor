$ErrorActionPreference = "Continue"

Write-Host "=== Service status ==="
docker compose ps predictor streamlit

Write-Host "`n=== Predictor health ==="
try {
    Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 30
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
}

Write-Host "`n=== Live inference parity ==="
try {
    Invoke-RestMethod -Method Post -Uri "http://localhost:8000/validate-live-parity?force=true" |
        ConvertTo-Json -Depth 30
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
}

Write-Host "`n=== Prediction result files ==="
Get-Item `
  ".\artifacts\results\production\finalized_training_frame.csv", `
  ".\artifacts\results\production\production_prediction_cache.csv" `
  -ErrorAction SilentlyContinue

Write-Host "`n=== Predictor state ==="
docker inspect kg-admet-predictor --format "OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} Status={{.State.Status}} RestartCount={{.RestartCount}}"

Write-Host "`n=== Recent predictor logs ==="
docker compose logs --tail=150 predictor
