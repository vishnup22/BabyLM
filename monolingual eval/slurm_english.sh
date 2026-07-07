#!/bin/bash
#SBATCH --job-name=eval_english
#SBATCH --output=logs/english_%j.out
#SBATCH --error=logs/english_%j.err
#SBATCH --partition=gpu-week-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate telugu_llm

cd "$(dirname "$0")"
mkdir -p logs

python english.py --cleaned_dir cleaned
