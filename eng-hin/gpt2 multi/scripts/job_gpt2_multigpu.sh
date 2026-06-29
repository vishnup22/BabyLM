#!/bin/bash
#SBATCH --job-name=eng-hin-gpt2
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/eng_hin_gpt2_%j.out
#SBATCH --error=logs/eng_hin_gpt2_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM/eng-hin/gpt2\ multi
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

# Train tokenizer if missing
if [[ ! -f "tokenizers/en_hi_equal/tokenizer.json" ]]; then
  echo "Training tokenizer..."
  python train_tokenizer.py en_hi_equal --vocab_size 32768
fi

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1

for SEED in 1 2; do
  export MASTER_PORT=$((29500 + SEED))

  accelerate launch \
    --config_file accelerate_4xa100_bf16.yaml \
    --num_processes 4 \
    training.py \
    --dataset "en_hi_equal" \
    --words_per_epoch 100000000 \
    --batch_size 64 \
    --seed "$SEED" \
    --experiment_name "eng-hin-gpt2-seed${SEED}"
done
