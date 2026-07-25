$ErrorActionPreference = "Continue"
Write-Host "=== Docker Compose status ==="
docker compose ps
Write-Host "`n=== Predictor state ==="
docker inspect kg-admet-predictor --format "{{json .State}}"
Write-Host "`n=== Predictor health ==="
try { Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 12 } catch { Write-Warning $_ }
Write-Host "`n=== Predictor logs ==="
docker logs kg-admet-predictor --tail 300
Write-Host "`n=== Docker memory ==="
docker info --format "{{json .MemTotal}}"
