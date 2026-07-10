#!/bin/bash
#SBATCH --job-name=eval-llama-missing
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_llama_missing_%j.out
#SBATCH --error=logs/eval_llama_missing_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no \
  eval_llama_missing.py --output results_llama_missing.json
