#!/bin/bash
#SBATCH --job-name=eval-hindi
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/eval_hindi_%j.out
#SBATCH --error=logs/eval_hindi_%j.err

source ~/.bashrc
conda activate telugu_llm

cd "$(dirname "$0")"
mkdir -p logs

python hindi.py --cleaned_dir cleaned
