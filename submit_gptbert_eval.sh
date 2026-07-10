#!/bin/bash
#SBATCH --job-name=eval-gptbert-seq
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/eval_gptbert_seq_%j.out
#SBATCH --error=logs/eval_gptbert_seq_%j.err


set -eo pipefail


cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs


eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm


python eval_gptbert_all.py \
  --models mono_en mono_hi mono_te en_hi_seed1 en_tel_seed2 \
  --output results_gptbert_seq.json




