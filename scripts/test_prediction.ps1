param(
  [string]$Compound = "CID2244",
  [string]$Target = "CYP3A4"
)
$body = @{ compounds = @($Compound); targets = @($Target) } | ConvertTo-Json
Write-Host "Health:"
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 12
Write-Host "Prediction:"
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 20
