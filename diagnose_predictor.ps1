$ErrorActionPreference = "Continue"

Write-Host "=== Compose status ==="
docker compose ps predictor streamlit

Write-Host "`n=== Predictor liveness ==="
try {
    Invoke-RestMethod -Uri "http://localhost:8000/live" |
        ConvertTo-Json -Depth 10
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message
    }
}

Write-Host "`n=== Predictor diagnostic health ==="
try {
    Invoke-RestMethod -Uri "http://localhost:8000/health" |
        ConvertTo-Json -Depth 20
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message
    }
}

Write-Host "`n=== Predictor readiness body ==="
try {
    Invoke-RestMethod -Uri "http://localhost:8000/ready" |
        ConvertTo-Json -Depth 20
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message
    }
}

Write-Host "`n=== Mounted production assets ==="
docker compose exec predictor sh -lc @'
find /models/production -maxdepth 1 -type f -printf "%f %s bytes\n" 2>/dev/null || true
find /results/production -maxdepth 1 -type f -printf "%f %s bytes\n" 2>/dev/null || true
'@

Write-Host "`n=== Asset validation ==="
docker compose exec predictor python /opt/kg/validate_prediction_assets.py

Write-Host "`n=== Recent predictor logs ==="
docker compose logs --tail=100 predictor
