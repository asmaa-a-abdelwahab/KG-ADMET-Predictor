$ErrorActionPreference = "Continue"

Write-Host "=== Predictor health ==="
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 30

Write-Host "`n=== Force live parity validation ==="
try {
    Invoke-RestMethod -Method Post `
      -Uri "http://localhost:8000/validate-live-parity?force=true" |
      ConvertTo-Json -Depth 40
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
}

Write-Host "`n=== Stage 1 prepared feature files in the container ==="
docker compose exec predictor sh -lc @'
find /modeling_prepared -type f \( -name '*gds_features.csv' -o -name '*gds_features.parquet' \) -printf '%p %s bytes\n' 2>/dev/null || true
'@

Write-Host "`n=== Fallback bundle files ==="
docker compose exec predictor sh -lc @'
ls -lh /models/production/production_stage3_fallback.joblib /models/production/stage3_fallback_manifest.json 2>/dev/null || true
'@

Write-Host "`n=== Recent predictor logs ==="
docker compose logs --tail=150 predictor
