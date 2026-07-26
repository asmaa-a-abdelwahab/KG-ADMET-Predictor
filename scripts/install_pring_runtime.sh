#!/usr/bin/env bash
set -euo pipefail

# Runtime helper shared by the loader, modeling, and Streamlit containers.
# It keeps the images usable before PRING is published by installing from a
# mounted local package directory when available, otherwise from an explicitly
# configured immutable PRING_PACKAGE_SPEC. Mutable branch archives are rejected.

PRING_PACKAGE_DIR="${PRING_PACKAGE_DIR:-/workspace/pring}"
PRING_PACKAGE_SPEC="${PRING_PACKAGE_SPEC:-}"
PRING_ALLOW_MUTABLE_PACKAGE_SPEC="${PRING_ALLOW_MUTABLE_PACKAGE_SPEC:-false}"

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
  if [ -z "${PRING_PACKAGE_SPEC}" ]; then
    echo "ERROR: PRING-PACKAGE is not mounted at ${PRING_PACKAGE_DIR}." >&2
    echo "Mount the sibling PRING-PACKAGE repository or set PRING_PACKAGE_SPEC to an immutable commit/tag artifact." >&2
    exit 2
  fi
  if [[ ! "${PRING_ALLOW_MUTABLE_PACKAGE_SPEC,,}" =~ ^(1|true|yes|on)$ ]] \
    && [[ "${PRING_PACKAGE_SPEC}" =~ /refs/heads/|@main([#/:]|$)|@master([#/:]|$) ]]; then
    echo "ERROR: Refusing mutable PRING_PACKAGE_SPEC: ${PRING_PACKAGE_SPEC}" >&2
    echo "Pin a commit/tag artifact, or explicitly set PRING_ALLOW_MUTABLE_PACKAGE_SPEC=true for local diagnostics." >&2
    exit 2
  fi
  echo "Installing PRING from package spec: ${PRING_PACKAGE_SPEC}"
  python -m pip install --no-cache-dir "${PRING_PACKAGE_SPEC}"
fi
