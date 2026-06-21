#!/usr/bin/env bash
#SBATCH --job-name=stage3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%12
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_gpu_%A_%a.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_gpu_%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_selected_implementation.sh" "${BASH_SOURCE[0]}" "$@"
