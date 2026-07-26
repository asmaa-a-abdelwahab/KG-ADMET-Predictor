# Production-directory status

The model artifacts currently retained in this directory are legacy diagnostic
artifacts. Their historical manifest is intentionally unchanged for provenance.

Production startup now requires an audited manifest with `status=ready`,
`publishable=true`, dataset/split/feature/label identifiers, runtime versions,
matching artifact digests, and an uncontaminated reference-frame digest. These
legacy artifacts must fail that gate.

Replace them only with a newly trained, independently validated, signed release
bundle. Do not edit the historical manifest or model solely to bypass validation.
