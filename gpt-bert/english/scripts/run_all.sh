#!/usr/bin/env bash
set -euo pipefail

bash scripts/download_dataset.sh
bash scripts/clean_dataset.sh
bash scripts/train_tokenizer.sh
bash scripts/prepare_shards.sh
bash scripts/train_model.sh
