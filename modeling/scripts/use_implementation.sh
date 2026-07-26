#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMPL="${1:-${MODEL_IMPL:-${MODEL_IMPLEMENTATION:-improved_v2}}}"

case "$IMPL" in
  legacy|old) IMPL="legacy" ;;
  improved|new) IMPL="improved" ;;
  improved_v2|final|finalized|v2) IMPL="improved_v2" ;;
  *)
    echo "ERROR: Unknown implementation '$IMPL'. Use MODEL_IMPL=legacy, MODEL_IMPL=improved, or MODEL_IMPL=improved_v2." >&2
    exit 2
    ;;
esac

IMPL_DIR="$MODEL_ROOT/implementations/$IMPL"
if [ ! -d "$IMPL_DIR" ]; then
  echo "ERROR: Implementation directory not found: $IMPL_DIR" >&2
  exit 2
fi

echo "$IMPL" > "$MODEL_ROOT/.active_implementation"
echo "Selected PRING modeling implementation: $IMPL"
echo "Implementation directory: $IMPL_DIR"
echo "No source files were replaced. Wrappers run the selected implementation through PYTHONPATH."
echo "For an interactive shell, run:"
if [ "$IMPL" = "improved_v2" ]; then
  printf "  export PYTHONPATH='%s'\${PYTHONPATH:+:\$PYTHONPATH}\n" "$MODEL_ROOT"
else
  printf "  export PYTHONPATH='%s'\${PYTHONPATH:+:\$PYTHONPATH}\n" "$IMPL_DIR"
fi
