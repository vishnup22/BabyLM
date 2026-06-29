#!/bin/bash
#SBATCH --job-name=babylm-gptbert-all
#SBATCH --partition=l40s-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/gptbert_all_%j.out
#SBATCH --error=logs/gptbert_all_%j.err

set -euo pipefail

REPO_ROOT="/nfs/storage1/home/pulipakv/multilingual-babylm"
LANGS=("english" "hindi" "telugu")
SEEDS=(1 2)

mkdir -p "$REPO_ROOT/gpt-bert/logs"

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export N_GPUS=3

for LANG in "${LANGS[@]}"; do
  echo "===== PREPARING $LANG ====="
  cd "$REPO_ROOT/gpt-bert/$LANG"

  bash scripts/train_tokenizer.sh
  bash scripts/prepare_shards.sh

  for SEED in "${SEEDS[@]}"; do
    echo "===== TRAINING $LANG seed${SEED} ====="
    export SEED
    export NAME="${LANG}-gptbert-base-seed${SEED}"
    export MASTER_PORT=$((29500 + SEED + (${#LANG} * 10)))

    bash scripts/train_model.sh
  done
done


