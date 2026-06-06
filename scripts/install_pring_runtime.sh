#!/usr/bin/env bash
set -euo pipefail

# Runtime helper shared by the loader, modeling, and Streamlit containers.
# It keeps the images usable before PRING is published by installing from a
# mounted local package directory when available, otherwise from PRING_PACKAGE_SPEC. The default PRING_PACKAGE_SPEC installs from the public PRING GitHub repository because PRING is not published on PyPI yet.

PRING_PACKAGE_DIR="${PRING_PACKAGE_DIR:-/workspace/pring}"
PRING_PACKAGE_SPEC="${PRING_PACKAGE_SPEC:-https://github.com/asmaa-a-abdelwahab/PRING/archive/refs/heads/main.zip}"

if python - <<'PY' >/dev/null 2>&1
import pring
print(getattr(pring, '__version__', 'installed'))
PY
then
  echo "PRING is already importable."
  exit 0
fi

if [ -f "${PRING_PACKAGE_DIR}/pyproject.toml" ] || [ -f "${PRING_PACKAGE_DIR}/setup.py" ]; then
  echo "Installing PRING from local directory: ${PRING_PACKAGE_DIR}"
  python -m pip install --no-cache-dir -e "${PRING_PACKAGE_DIR}"
else
  echo "Installing PRING from package spec: ${PRING_PACKAGE_SPEC}"
  python -m pip install --no-cache-dir "${PRING_PACKAGE_SPEC}"
fi
