#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -e "${MODELING_PACKAGE_DIR:-./modeling}[notebooks]"
