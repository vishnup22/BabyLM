#!/bin/bash
#SBATCH --job-name=eng-tel-gpt2
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/eng_tel_gpt2_%j.out
#SBATCH --error=logs/eng_tel_gpt2_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM/eng-tel/gpt2\ multi
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

# Train tokenizer if missing
if [[ ! -f "tokenizers/en_tel_equal/tokenizer.json" ]]; then
  echo "Training tokenizer..."
  python train_tokenizer.py en_tel_equal --vocab_size 32768
fi

# Pre-build dataset cache with a single process to avoid distributed barrier timeout
CACHE_FILE="data/cached_train/train_gpt2_en_tel_equal.pkl"
if [[ ! -f "$CACHE_FILE" ]]; then
  echo "Pre-building dataset cache (single process)..."
  python - <<'EOF'
import pickle
from pathlib import Path
from data_utils import FullBabyLMDataset

cfg = {"dataset": "en_tel_equal", "datapoint_length": 512}
cache_dir = Path("data/cached_train")
cache_dir.mkdir(parents=True, exist_ok=True)
filename = cache_dir / "train_gpt2_en_tel_equal.pkl"
dset = FullBabyLMDataset(cfg)
with open(filename, "wb") as f:
    pickle.dump(dset, f)
print(f"Cache saved to {filename}")
EOF
fi

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1

for SEED in 1 2; do
  export MASTER_PORT=$((29500 + SEED))

  accelerate launch \
    --config_file accelerate_4xa100_bf16.yaml \
    --num_processes 4 \
    training.py \
    --dataset "en_tel_equal" \
    --words_per_epoch 88500000 \
    --batch_size 64 \
    --seed "$SEED" \
    --experiment_name "eng-tel-gpt2-seed${SEED}"
done
