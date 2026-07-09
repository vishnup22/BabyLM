#!/bin/bash
#SBATCH --job-name=eval-gptbert-en-hi-seed2
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_gptbert_en_hi_seed2_%j.out
#SBATCH --error=logs/eval_gptbert_en_hi_seed2_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

python eval_gptbert_all.py --models en_hi_seed2 --output results_gptbert_en_hi_seed2.json
