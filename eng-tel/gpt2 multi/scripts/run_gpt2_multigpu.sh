#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET=${DATASET:-en_tel_equal}
VOCAB_SIZE=${VOCAB_SIZE:-32768}
N_GPUS=${N_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
N_EPOCHS=${N_EPOCHS:-10}
WORDS_PER_EPOCH=${WORDS_PER_EPOCH:-100000000}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DATASET}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(( ( RANDOM % 16384 ) + 49152 ))}

TOK_DIR="$ROOT_DIR/tokenizers/$DATASET"
DATA_DIR="$ROOT_DIR/data/$DATASET"

export WANDB_DISABLED=1

[[ -d "$DATA_DIR" ]] || { echo "Missing dataset dir: $DATA_DIR (run build_multilingual_dataset.py first)"; exit 1; }

# Train tokenizer if missing
if [[ ! -f "$TOK_DIR/tokenizer.json" ]]; then
  echo "Tokenizer not found at $TOK_DIR. Training tokenizer..."
  (cd "$ROOT_DIR" && python train_tokenizer.py "$DATASET" --vocab_size "$VOCAB_SIZE")
else
  echo "Using existing tokenizer: $TOK_DIR"
fi

echo "Launching GPT-2 training on $N_GPUS GPUs"
echo "  Dataset:    $DATASET ($DATA_DIR)"
echo "  Tokenizer:  $TOK_DIR"
echo "  Batch/GPU:  $BATCH_SIZE  |  Epochs: $N_EPOCHS  |  Words/epoch: $WORDS_PER_EPOCH"
echo ""

export MASTER_ADDR MASTER_PORT
TORCHRUN_EXTRA_ARGS=${TORCHRUN_EXTRA_ARGS:-}
exec torchrun \
  --nproc_per_node="$N_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  $TORCHRUN_EXTRA_ARGS \
  "$ROOT_DIR/training.py" \
  --dataset "$DATASET" \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$N_EPOCHS" \
  --words_per_epoch "$WORDS_PER_EPOCH" \
  --experiment_name "$EXPERIMENT_NAME" \
  "$@"
