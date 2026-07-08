#!/usr/bin/env bash
set -eo pipefail

accelerate launch \
  --config_file accelerate_4xa100_bf16.yaml \
  --num_processes 4 \
  training.py \
  --dataset "translated-babylm-telugu" \
  --curriculum \
  --curriculum_file "curriculum_data/telugu.txt" \
  --words_per_epoch 100000000 \
  --batch_size 4 \
  --experiment_name "telugu-strict-100m-curriculum" \
  --use_wandb \
  --wandb_project_name "babylm_2026_gpt2" \
  --wandb_experiment_name "telugu-strict-100m-curriculum"
