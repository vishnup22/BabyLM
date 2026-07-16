#!/usr/bin/env bash
set -euo pipefail

python tools/prepare_local_shards.py \
  --dataset native-babylm-hindi \
  --data_root data/raw \
  --tokenizer tokenizers/tokenizer_base_16384.json \
  --output_base data/processed \
  --valid_fraction 0
