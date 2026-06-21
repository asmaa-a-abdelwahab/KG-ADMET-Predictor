#!/usr/bin/env bash
#SBATCH --job-name=all_models_compare
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/all_models_compare_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/all_models_compare_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_selected_implementation.sh" "${BASH_SOURCE[0]}" "$@"
