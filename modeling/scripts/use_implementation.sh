#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMPL="${1:-${MODEL_IMPL:-${MODEL_IMPLEMENTATION:-improved}}}"

case "$IMPL" in
  legacy|old) IMPL="legacy" ;;
  improved|new) IMPL="improved" ;;
  *)
    echo "ERROR: Unknown implementation '$IMPL'. Use MODEL_IMPL=legacy or MODEL_IMPL=improved." >&2
    exit 2
    ;;
esac

IMPL_DIR="$MODEL_ROOT/implementations/$IMPL"
if [ ! -d "$IMPL_DIR" ]; then
  echo "ERROR: Implementation directory not found: $IMPL_DIR" >&2
  exit 2
fi

# These entries make ./modeling behave like the selected implementation for
# editable installs and interactive python -m pring_modeling... commands.
items=(
  pyproject.toml
  requirements.txt
  requirements-pyg-cu124.txt
  Dockerfile
  node_embeddings.py
  pring_modeling
  stages
  notebooks
)

for item in "${items[@]}"; do
  target="$MODEL_ROOT/$item"
  source="$IMPL_DIR/$item"
  if [ -e "$source" ]; then
    rm -rf "$target"
    ln -s "implementations/$IMPL/$item" "$target"
  fi
done

echo "$IMPL" > "$MODEL_ROOT/.active_implementation"
echo "Active PRING modeling implementation: $IMPL"
echo "Implementation directory: $IMPL_DIR"
