#!/usr/bin/env bash
#SBATCH --job-name=stage1_gds_baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage1_gds_baseline_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage1_gds_baseline_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_selected_implementation.sh" "${BASH_SOURCE[0]}" "$@"
