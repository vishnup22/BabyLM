#!/bin/bash
#SBATCH --job-name=eval_telugu
#SBATCH --output=logs/telugu_%j.out
#SBATCH --error=logs/telugu_%j.err
#SBATCH --partition=gpu-week-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate telugu_llm

cd "$(dirname "$0")"
mkdir -p logs

python telugu.py --cleaned_dir cleaned
