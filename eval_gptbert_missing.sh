#!/bin/bash
#SBATCH --job-name=eval-gptbert-missing
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_gptbert_missing_%j.out
#SBATCH --error=logs/eval_gptbert_missing_%j.err

# Runs ONLY the GPT-BERT evals still missing from evaluation.md, each distributed across
# 4 GPUs. Skips en_hi_seed2 and en_tel_seed1 per current scope.
#
# Missing set:
#   mono_en            -- MuBench only
#   mono_hi            -- MuBench only
#   mono_te            -- everything (perplexity, SIB-200, MuBench -- was broken/never run)
#   en_hi_seed1  (en)  -- MuBench only
#   en_hi_seed1  (hi)  -- SIB-200, MuBench
#   en_tel_seed2 (en)  -- MuBench only
#   en_tel_seed2 (te)  -- everything (perplexity, SIB-200, MuBench)

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

OUT=results_gptbert_missing.json
RUN="accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no eval_gptbert_all.py --output $OUT"

$RUN --models mono_en                     --evals mubench
$RUN --models mono_hi                     --evals mubench
$RUN --models mono_te                     --evals perplexity sib200 mubench
$RUN --models en_hi_seed1  --langs en     --evals mubench
$RUN --models en_hi_seed1  --langs hi     --evals sib200 mubench
$RUN --models en_tel_seed2 --langs en     --evals mubench
$RUN --models en_tel_seed2 --langs te     --evals perplexity sib200 mubench
