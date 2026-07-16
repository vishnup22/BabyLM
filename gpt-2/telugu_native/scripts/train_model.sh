#!/usr/bin/env bash
set -euo pipefail

accelerate launch \
  --config_file accelerate_4xa100_bf16.yaml \
  --num_processes 4 \
  training.py \
  --dataset "native-babylm-telugu" \
  --words_per_epoch 100000000 \
  --batch_size 4 \
  --experiment_name "telugu-native-strict-100m" \
