#!/usr/bin/env bash
set -euo pipefail
python -m pip install --no-cache-dir neo4j >/dev/null
python /opt/kg/bin/check_gds.py
