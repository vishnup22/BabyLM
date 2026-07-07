#!/bin/bash
#SBATCH --job-name=eval_hindi
#SBATCH --output=logs/hindi_%j.out
#SBATCH --error=logs/hindi_%j.err
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

source ~/.bashrc
conda activate telugu_llm

cd "$(dirname "$0")"
mkdir -p logs

python hindi.py --cleaned_dir cleaned
