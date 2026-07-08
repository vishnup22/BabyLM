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

set -eo pipefail

cd "/nfs/storage1/home/pulipakv/BabyLM/gpt-2 curriculum/english"
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "=== Training tokenizer ==="
python train_tokenizer.py BabyLM-2026-Strict --curriculum_file curriculum_data/english.txt

echo "=== Training model ==="
bash scripts/train_model.sh
