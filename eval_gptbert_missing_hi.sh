#!/bin/bash
#SBATCH --job-name=eval-gptbert-missing-hi
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/eval_gptbert_missing_hi_%j.out
#SBATCH --error=logs/eval_gptbert_missing_hi_%j.err

# Hindi-side gaps only (mono_hi MuBench already done, skipped here):
#   en_hi_seed1  (hi)   -- SIB-200, MuBench

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

OUT=results_gptbert_missing_hi.json
RUN="accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no eval_gptbert_all.py --output $OUT"

$RUN --models en_hi_seed1  --langs hi     --evals sib200 mubench
