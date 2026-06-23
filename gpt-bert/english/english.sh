#!/bin/bash
#SBATCH --job-name=babylm-eng-gptbert
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/english_%j.out
#SBATCH --error=logs/english_%j.err

set -eo pipefail

cd /path/to/babylm-baselines/gpt-bert/english
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

bash scripts/train_model.sh
