#!/usr/bin/env bash
set -eo pipefail

accelerate launch \
  --config_file accelerate_4xa100_bf16.yaml \
  --num_processes 4 \
  training.py \
  --dataset "BabyLM-2026-Strict" \
  --curriculum \
  --curriculum_file "curriculum_data/english.txt" \
  --words_per_epoch 100000000 \
  --batch_size 4 \
  --experiment_name "english-strict-100m-curriculum" \
  --experiment_name "english-strict-100m-curriculum"
