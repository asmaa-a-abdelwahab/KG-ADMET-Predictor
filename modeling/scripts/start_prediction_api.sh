#!/usr/bin/env bash
set -euo pipefail

require_verified="${PREDICTION_REQUIRE_VERIFIED_ARTIFACTS:-false}"
require_publishable="${PREDICTION_REQUIRE_PUBLISHABLE_ARTIFACTS:-${require_verified}}"

if [[ "${require_verified,,}" =~ ^(1|true|yes|on)$ ]] \
  || [[ "${require_publishable,,}" =~ ^(1|true|yes|on)$ ]]; then
  echo "Validating prediction assets before loading serialized models."
  python /opt/kg/validate_prediction_assets.py
fi

exec python -m uvicorn pring_modeling.prediction_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --timeout-keep-alive 10
