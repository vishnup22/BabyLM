#!/usr/bin/env bash
set -euo pipefail

python tools/prepare_local_shards.py \
  --dataset translated-babylm-telugu \
  --data_root ../../gpt-2/telugu/data \
  --tokenizer tokenizers/tokenizer_base_16384.json \
  --output_base data/processed
