#!/bin/bash
#SBATCH --job-name=eng-hin-gpt2
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH --gpus=4
#SBATCH --output=logs/gpt2_multigpu_%j.out
#SBATCH --error=logs/gpt2_multigpu_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p logs

source activate telugu_llm

N_GPUS=4 bash scripts/run_gpt2_multigpu.sh
