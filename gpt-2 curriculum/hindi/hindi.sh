#!/bin/bash
#SBATCH --job-name=babylm-hin-gpt2
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/hindi_%j.out
#SBATCH --error=logs/hindi_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/multilingual-babylm/gpt-2/hindi
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

bash scripts/train_model.sh