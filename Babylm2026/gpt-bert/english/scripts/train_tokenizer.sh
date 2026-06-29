#!/usr/bin/env bash
set -euo pipefail

python tools/train_tokenizer_local.py \
  --dataset BabyLM-2026-Strict \
  --data_root ../../gpt-2/english/data \
  --output tokenizers/tokenizer_base_16384.json \
  --vocab_size 16384
