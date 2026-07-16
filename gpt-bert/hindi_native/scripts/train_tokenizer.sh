#!/usr/bin/env bash
set -euo pipefail

python tools/train_tokenizer_local.py \
  --dataset native-babylm-hindi \
  --data_root data/raw \
  --output tokenizers/tokenizer_base_16384.json \
  --vocab_size 16384
