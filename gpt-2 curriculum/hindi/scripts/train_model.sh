#!/usr/bin/env bash
set -eo pipefail

accelerate launch \
  --config_file accelerate_4xa100_bf16.yaml \
  --num_processes 4 \
  training.py \
  --dataset "translated-babylm-hindi" \
  --curriculum \
  --curriculum_file "curriculum_data/hindi.txt" \
  --words_per_epoch 100000000 \
  --batch_size 4 \
  --experiment_name "hindi-strict-100m-curriculum" \
