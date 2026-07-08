#!/bin/bash
#SBATCH --job-name=babylm-tel-2seeds
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/telugu_%j.out
#SBATCH --error=logs/telugu_%j.err

set -eo pipefail

cd "/nfs/storage1/home/pulipakv/BabyLM/gpt-2 curriculum/telugu"
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "=== Training tokenizer ==="
python train_tokenizer.py translated-babylm-telugu --curriculum_file curriculum_data/telugu.txt

echo "=== Training model ==="
bash scripts/train_model.sh


