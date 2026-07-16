#!/usr/bin/env bash
set -euo pipefail

bash scripts/download_dataset.sh
bash scripts/clean_dataset.sh
bash scripts/train_tokenizer.sh
bash scripts/prepare_shards.sh
bash scripts/train_model.sh

# push the trained checkpoint to HF immediately after training finishes -- so it's
# accessible even if the cluster session/job gets cleaned up afterward
SEED="${SEED:-1}"
python ../convert_gptbert_to_hf.py \
  --lang telugu_native \
  --seed "$SEED" \
  --repo "pulipakav-1/telugu-native-gptbert-base-seed${SEED}"
