#!/usr/bin/env bash
set -euo pipefail

accelerate launch \
  --config_file accelerate_4xa100_bf16.yaml \
  --num_processes 4 \
  training.py \
  --dataset "BabyLM-2026-Strict" \
  --words_per_epoch 100000000 \
  --batch_size 4 \
  --experiment_name "english-strict-100m" \
  --use_wandb \
  --wandb_project_name "babylm_2026_gpt2" \
  --wandb_experiment_name "english-strict-100m"
