#!/bin/bash
#SBATCH --job-name=eval-gptbert-missing-te
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_gptbert_missing_te_%j.out
#SBATCH --error=logs/eval_gptbert_missing_te_%j.err

# Telugu-side gaps only:
#   mono_te             -- everything (perplexity, SIB-200, MuBench -- was broken/never run)
#   en_tel_seed2 (te)    -- everything (perplexity, SIB-200, MuBench)

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

OUT=results_gptbert_missing_te.json
RUN="accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no eval_gptbert_all.py --output $OUT"

$RUN --models mono_te                     --evals perplexity sib200 mubench
$RUN --models en_tel_seed2 --langs te     --evals perplexity sib200 mubench
