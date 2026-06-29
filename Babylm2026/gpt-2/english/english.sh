#!/bin/bash
#SBATCH --job-name=babylm-eng-gpt2
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/english_%j.out
#SBATCH --error=logs/english_%j.err

set -euo pipefail

cd /path/to/babylm-baselines/gpt-2/english
mkdir -p logs

source ~/.bashrc
conda activate telugu_babylm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

bash scripts/train_model.sh
