#!/usr/bin/env bash
set -euo pipefail

python tools/train_tokenizer_local.py \
  --dataset translated-babylm-hindi \
  --data_root ../../gpt-2/hindi/data \
  --output tokenizers/tokenizer_base_16384.json \
  --vocab_size 16384
