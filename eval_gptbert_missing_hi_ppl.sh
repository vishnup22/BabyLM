#!/bin/bash
#SBATCH --job-name=eval-gptbert-missing-hi-ppl
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/eval_gptbert_missing_hi_ppl_%j.out
#SBATCH --error=logs/eval_gptbert_missing_hi_ppl_%j.err

# Hindi-side perplexity for en_hi_seed1 (eng-hin GPT-BERT bilingual) -- never queued before
# (eval_gptbert_missing_hi.sh only ever ran SIB-200/MuBench for this model). Separate output
# file so this doesn't race the MuBench job's read-modify-write of results_gptbert_missing_hi.json
# if both are queued around the same time.

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

OUT=results_gptbert_missing_hi_ppl.json
RUN="accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no eval_gptbert_all.py --output $OUT"

$RUN --models en_hi_seed1  --langs hi     --evals perplexity
