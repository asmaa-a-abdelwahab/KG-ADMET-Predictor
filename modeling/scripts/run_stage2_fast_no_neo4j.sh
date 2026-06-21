#!/usr/bin/env bash
#SBATCH --job-name=stage2_fast
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage2_fast_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage2_fast_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_selected_implementation.sh" "${BASH_SOURCE[0]}" "$@"
