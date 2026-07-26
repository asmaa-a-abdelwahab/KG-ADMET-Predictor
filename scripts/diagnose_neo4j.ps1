Write-Host "Docker Compose services:" -ForegroundColor Cyan
docker compose ps

Write-Host "\nLast Neo4j logs:" -ForegroundColor Cyan
docker logs pring-app-neo4j --tail 200

Write-Host "\nNeo4j healthcheck state:" -ForegroundColor Cyan
docker inspect pring-app-neo4j --format '{{json .State.Health}}'
