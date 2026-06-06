# Start Neo4j alone and print the logs needed for diagnosis.
# Usage from project root:
#   powershell -ExecutionPolicy Bypass -File scripts/start_neo4j_safe.ps1

docker compose down
Write-Host "Starting Neo4j only..."
docker compose up -d --build neo4j
Start-Sleep -Seconds 5
Write-Host "Current containers:"
docker compose ps
Write-Host "Neo4j logs:"
docker logs kg-admet-neo4j --tail 200
Write-Host "If Neo4j is running, wait until it becomes healthy, then run:"
Write-Host "docker compose --profile load up --build pring-loader"
