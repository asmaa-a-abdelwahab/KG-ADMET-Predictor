# Security policy

## Supported version

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Use the repository's private **Security → Report a vulnerability** workflow on
GitHub. Do not publish credentials, exploitable Cypher, model paths, or service
details in a public issue.

Include the affected version, reproduction steps, impact, and any proposed
mitigation.

## Production expectations

- Require the prediction API key.
- Keep Neo4j on a trusted network.
- Terminate TLS before exposing application services.
- Use non-default secrets and a versioned graph snapshot.
- Reject stale caches, runtime mismatches, digest failures, and unverified
  legacy bundles.
- Keep debug errors disabled.

