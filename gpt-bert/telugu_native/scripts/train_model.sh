#!/usr/bin/env bash
set -euo pipefail

N_GPUS="${N_GPUS:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-1}"
# matches convert_gptbert_to_hf.py's expected ckpt_name pattern exactly (note the space
# before the seed number -- that script does f"{lang}-gptbert-base-seed {seed}_ema.bin")
NAME="${NAME:-telugu_native-gptbert-base-seed $SEED}"

torchrun --nproc_per_node="$N_GPUS" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  pretraining/train_multi_gpu.py \
  --train_path data/processed/train \
  --config_file configs/base.json \
  --tokenizer_path tokenizers/tokenizer_base_16384.json \
  --name "$NAME" \
  --output_dir model_checkpoints \
  --hybrid_denominator 3 \
  --hybrid_numerator 2 \
  --global_batch_size 32768 \
  --local_batch_size 64 \
  --seq_length 128 \
  --max_steps 15625 \
  --save_every 1000 \
  --validate_every 0 \
  --seed "$SEED" \
  --no_validation \
  --wandb_disabled
