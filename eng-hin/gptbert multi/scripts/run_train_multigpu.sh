#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAIN_DIR="$ROOT_DIR/pretraining"
TOK_DIR="$ROOT_DIR/tokenizers"
CONFIG="${CONFIG:-$ROOT_DIR/configs/multilingual.json}"
DATASET_SOURCE_DIR="${DATASET_SOURCE_DIR:-$ROOT_DIR/../gpt2 multi/data/en_hi_equal}"
DATA_BASE_DIR="${DATA_BASE_DIR:-$ROOT_DIR/data/EN_HI_EQUAL}"

[[ -f "$CONFIG" ]] || { echo "Missing $CONFIG"; exit 1; }
[[ -d "$DATASET_SOURCE_DIR" ]] || { echo "Missing dataset dir $DATASET_SOURCE_DIR"; exit 1; }

if [[ "${N_GPUS:-2}" -le 1 ]]; then
  echo "This launcher is for multi-GPU only. Use the single-GPU training script for 1 GPU." >&2
  exit 2
fi

TRAIN_PATH="${TRAIN_PATH:-$DATA_BASE_DIR/train}"
DEV_PATH="${DEV_PATH:-$DATA_BASE_DIR/valid}"

N_GPUS=${N_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-15625}
SAVE_EVERY=${SAVE_EVERY:-1000}
VALIDATE_EVERY=${VALIDATE_EVERY:-1000}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(( ( RANDOM % 16384 ) + 49152 ))}
NAME=${NAME:-en-hi-gptbert}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/model_checkpoints"}

LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-128}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32768}
# 3/4 GPUs do masked LM, 1/4 does causal LM (approximates the original 15/16 ratio)
HYBRID_DENOMINATOR=${HYBRID_DENOMINATOR:-4}
HYBRID_NUMERATOR=${HYBRID_NUMERATOR:-3}
CONFIG_VOCAB=$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("vocab_size", 32768))' "$CONFIG" 2>/dev/null || echo 32768)
VOCAB_SIZE="${VOCAB_SIZE:-$CONFIG_VOCAB}"

export WANDB_DISABLED=${WANDB_DISABLED:-1}
WANDB_PROJECT=${WANDB_PROJECT:-BabyLM-GPT-BERT}
WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_PROJECT
if [[ -n "$WANDB_ENTITY" ]]; then export WANDB_ENTITY; else unset WANDB_ENTITY; fi
export WANDB_NAME="${WANDB_NAME:-$NAME}"

COMMON_ARGS=(
  --train_path "$TRAIN_PATH"
  --valid_path "$DEV_PATH"
  --config_file "$CONFIG"
  --name "$NAME"
  --output_dir "$OUTPUT_DIR"
)

CHECKPOINT_FILENAME=${CHECKPOINT_FILENAME:-}
RESUME=${RESUME:-0}
if [[ -z "$CHECKPOINT_FILENAME" && ("$RESUME" == "1" || "${AUTO_RESUME:-0}" == "1") ]]; then
  default_ckpt="$OUTPUT_DIR/${NAME}_state_dict.bin"
  if [[ -f "$default_ckpt" ]]; then
    CHECKPOINT_FILENAME="$default_ckpt"
  fi
fi

RESUME_ARGS=()
if [[ -n "$CHECKPOINT_FILENAME" ]]; then
  echo "[resume] Using checkpoint: $CHECKPOINT_FILENAME" >&2
  RESUME_ARGS+=(--checkpoint_filename "$CHECKPOINT_FILENAME")
fi

TOK_FILE="${TOKENIZER_PATH:-$TOK_DIR/tokenizer_en_hi_vs${VOCAB_SIZE}.json}"

has_shards() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  shopt -s nullglob
  local files=("$d"/*.bin)
  shopt -u nullglob
  [[ ${#files[@]} -gt 0 ]]
}

need_preprocess=0
if [[ ! -f "$TOK_FILE" ]] || ! has_shards "$TRAIN_PATH" || ! has_shards "$DEV_PATH"; then
  need_preprocess=1
fi

if [[ $need_preprocess -eq 1 ]]; then
  mkdir -p "$TOK_DIR"

  if [[ ! -f "$TOK_FILE" ]]; then
    echo "Tokenizer not found at $TOK_FILE. Training tokenizer..."
    python "$ROOT_DIR/tokenizers/tokenizer.py" \
      --dataset_dir "$DATASET_SOURCE_DIR" \
      --vocab_size "$VOCAB_SIZE" \
      --output "tokenizer_en_hi_vs${VOCAB_SIZE}.json"
  else
    echo "Using existing tokenizer: $TOK_FILE"
  fi

  echo "Shards missing. Running preprocessing..."
  python "$ROOT_DIR/preprocess/updated_preprocess.py" \
    --dataset_dir "$DATASET_SOURCE_DIR" \
    --tokenizer "$TOK_FILE" \
    --seq_length ${SEQ_LENGTH:-128} \
    --shard_size_bytes 100000000 \
    --batch_size 1000 \
    --max_workers 8 \
    --base_dir "$DATA_BASE_DIR"
fi

if [[ "${PREPROCESS_ONLY:-0}" == "1" ]]; then
  echo "PREPROCESS_ONLY=1 set; exiting before training."
  exit 0
fi

echo "Launching $N_GPUS GPUs"
export MASTER_ADDR MASTER_PORT
TORCHRUN_EXTRA_ARGS=${TORCHRUN_EXTRA_ARGS:-}
exec torchrun --nproc_per_node="$N_GPUS" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" $TORCHRUN_EXTRA_ARGS \
  "$PRETRAIN_DIR/train_multi_gpu.py" \
  "${COMMON_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --tokenizer_path "$TOK_FILE" \
  --hybrid_denominator "$HYBRID_DENOMINATOR" \
  --hybrid_numerator "$HYBRID_NUMERATOR" \
  --optimizer lamb \
  --optimizer_beta1 0.9 \
  --optimizer_beta2 0.98 \
  --optimizer_eps 1e-8 \
  --weight_decay 0.1 \
  --warmup_proportion 0.016 \
  --learning_rate 1e-2 \
  --seq_length 128 \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --batch_reduction 4 \
  --local_batch_size "$LOCAL_BATCH_SIZE" \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" \
  --validate_every "$VALIDATE_EVERY" \
  --wandb_disabled \
  "$@"
