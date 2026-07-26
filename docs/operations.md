# Deployment and security

## Production configuration

Use both Compose files:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.production.yml \
  config
```

The production overlay requires a Neo4j password, prediction API key, and graph
version. Store them in a secret manager or protected deployment environment.

## Filesystem policy

| Path | Production mode |
|---|---|
| Package runs | Read-only |
| Prepared modeling data | Read-only |
| Model bundles | Read-only |
| Reference result frame | Read-only |
| Prediction cache | Writable, separate, backed up |
| Neo4j data | Persistent volume |

## Network policy

- Bind directly published ports to localhost unless a reviewed reverse proxy is
  present.
- Terminate TLS before exposing services.
- Restrict Neo4j to trusted networks.
- Apply authentication, rate limits, request-size limits, and audit logging.

## Neo4j policy

APOC file import/export is disabled by default. Only GDS procedures are
unrestricted. Review plugin versions and database migrations before upgrades.

## Operational acceptance

- [ ] Images pass dependency, vulnerability, and secret scans.
- [ ] Containers run as non-root where supported.
- [ ] Health and readiness checks reflect dependency availability.
- [ ] Live/offline parity passes on the deployed graph snapshot.
- [ ] Load, concurrency, restart, Neo4j outage, cache corruption, and disk-full
      tests meet defined service objectives.
- [ ] Neo4j and prediction-cache backup, restore, and rollback are tested.

## Documentation deployment

The documentation workflow builds with `mkdocs build --strict` and deploys a
Pages artifact. Before the first deployment, a repository administrator must
open **Settings → Pages → Build and deployment → Source** and select **GitHub
Actions**. If the workflow reports `Get Pages site failed` or an HTTP 404 from
`configure-pages`, Pages has not yet been enabled for that repository. Re-run
the documentation workflow after changing the setting.

Do not add an administrative personal access token merely to enable Pages from
the workflow. The built-in `GITHUB_TOKEN` is sufficient for routine deployment
after the one-time repository setting is complete.
