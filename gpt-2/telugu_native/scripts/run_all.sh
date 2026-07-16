#!/usr/bin/env bash
set -euo pipefail

bash scripts/download_dataset.sh
bash scripts/clean_dataset.sh
bash scripts/train_tokenizer.sh
bash scripts/train_model.sh

# push the trained checkpoint to HF immediately after training finishes -- so it's
# accessible even if the cluster session/job gets cleaned up afterward
python push_to_hf.py \
  --dataset native-babylm-telugu \
  --experiment telugu-native-strict-100m \
  --repo pulipakav-1/telugu-native-gpt2-base
