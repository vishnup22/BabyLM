#!/usr/bin/env bash
set -euo pipefail

N_GPUS="${N_GPUS:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-1}"
# matches convert_gptbert_to_hf.py's expected ckpt_name pattern exactly (note the space
# before the seed number -- that script does f"{lang}-gptbert-base-seed {seed}_ema.bin")
NAME="${NAME:-hindi_native-gptbert-base-seed $SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-model_checkpoints}"

# Resume support (mirrors eng-hin/gptbert multi/scripts/run_train_multigpu.sh): set RESUME=1
# or AUTO_RESUME=1 to pick up the last checkpoint automatically, or set CHECKPOINT_FILENAME
# explicitly to resume from a specific one. The full training-state file (model + ema_model +
# optimizer + scheduler + global_step + epoch) is {NAME}_state_dict.bin -- NOT the *_ema.bin
# used for inference/HF conversion, which is weights-only and can't resume the optimizer state.
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-}"
RESUME="${RESUME:-0}"
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

torchrun --nproc_per_node="$N_GPUS" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  pretraining/train_multi_gpu.py \
  --train_path data/processed/train \
  --config_file configs/base.json \
  --tokenizer_path tokenizers/tokenizer_base_16384.json \
  --name "$NAME" \
  --output_dir "$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}" \
  --hybrid_denominator "$N_GPUS" \
  --hybrid_numerator "$((N_GPUS - 1))" \
  --global_batch_size 32768 \
  --local_batch_size "${LOCAL_BATCH_SIZE:-64}" \
  --seq_length 128 \
  --max_steps 15625 \
  --save_every 1000 \
  --validate_every 0 \
  --seed "$SEED" \
  --no_validation \
  --wandb_disabled
