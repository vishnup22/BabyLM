#!/bin/bash
#SBATCH --job-name=eval-gptbert-mubench-balanced
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/eval_gptbert_mubench_balanced_%j.out
#SBATCH --error=logs/eval_gptbert_mubench_balanced_%j.err

# All 5 remaining MuBench cells, using item-level load-balanced sharding across all 4 GPUs
# (see eval_gptbert_mubench_balanced.py) instead of the whole-task sharding that let one GPU
# get stuck alone for hours on a heavy task while the other 3 idled:
#   en_hi_seed1  (hi)  -- MuBench
#   mono_te            -- MuBench
#   en_tel_seed2 (te)  -- MuBench
#   en_hi_seed1  (en)  -- MuBench
#   en_tel_seed2 (en)  -- MuBench

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

OUT=results_gptbert_mubench_balanced.json
RUN="accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no eval_gptbert_mubench_balanced.py --output $OUT"

$RUN --models en_hi_seed1  --langs hi
$RUN --models mono_te
$RUN --models en_tel_seed2 --langs te
$RUN --models en_hi_seed1  --langs en
$RUN --models en_tel_seed2 --langs en
