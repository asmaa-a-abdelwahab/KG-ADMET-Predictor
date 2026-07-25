$ErrorActionPreference = "Continue"

Write-Host "=== Container status ==="
docker compose ps predictor streamlit neo4j

Write-Host "`n=== Predictor liveness ==="
try {
    Invoke-RestMethod http://localhost:8000/live | ConvertTo-Json -Depth 10
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
}

Write-Host "`n=== Predictor hybrid status ==="
try {
    Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 40
} catch {
    Write-Host $_.Exception.Message
    if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
}

Write-Host "`n=== Production score cache mount ==="
docker compose exec predictor sh -lc 'ls -lh /results/production/finalized_training_frame.csv && test -w /results/production/finalized_training_frame.csv && echo WRITABLE || echo NOT_WRITABLE'

Write-Host "`n=== Live model artifacts ==="
docker compose exec predictor sh -lc @'
for path in \
  /models/improved_v2/stage1_gds_extra_trees/stage1_tabular_extra_trees.joblib \
  /models/improved_v2/stage1_gds_extra_trees/feature_columns.json \
  /models/improved_v2/stage3_rgcn_sampled/best_model.pt \
  /models/improved_v2/stage3_rgcn_sampled/rgcn_sampled_metadata.json \
  /models/improved_v2/stage3_hgt_sampled/best_model.pt \
  /models/improved_v2/stage3_hgt_sampled/hgt_sampled_metadata.json; do
  if [ -f "$path" ]; then echo "OK $path"; else echo "MISSING $path"; fi
done
'@

Write-Host "`n=== PyG runtime ==="
docker compose exec predictor python -m pring_modeling.pyg_runtime

Write-Host "`n=== Recent predictor logs ==="
docker compose logs --tail=150 predictor
