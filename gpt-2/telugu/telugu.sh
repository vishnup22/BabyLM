#!/bin/bash
#SBATCH --job-name=babylm-tel-2seeds
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/telugu_%j.out
#SBATCH --error=logs/telugu_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/multilingual-babylm/gpt-2/telugu
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1

for SEED in 1 2; do
  export MASTER_PORT=$((29500 + SEED))

  accelerate launch \
    --config_file accelerate_4xa100_bf16.yaml \
    --num_processes 4 \
    training.py \
    --dataset "translated-babylm-telugu" \
    --words_per_epoch 100000000 \
    --batch_size 4 \
    --seed "$SEED" \
    --experiment_name "telugu-strict-100m-seed${SEED}"
done


