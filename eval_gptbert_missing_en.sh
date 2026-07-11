#!/bin/bash
#SBATCH --job-name=eval-gptbert-missing-en
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_gptbert_missing_en_%j.out
#SBATCH --error=logs/eval_gptbert_missing_en_%j.err

# English-side gaps only:
#   mono_en            -- MuBench
#   en_hi_seed1  (en)   -- MuBench
#   en_tel_seed2 (en)   -- MuBench

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

OUT=results_gptbert_missing_en.json
RUN="accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no eval_gptbert_all.py --output $OUT"

$RUN --models mono_en                     --evals mubench
$RUN --models en_hi_seed1  --langs en     --evals mubench
$RUN --models en_tel_seed2 --langs en     --evals mubench
