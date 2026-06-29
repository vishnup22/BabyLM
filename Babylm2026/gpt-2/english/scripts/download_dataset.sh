#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
DATASET_REPO="BabyLM-community/BabyLM-2026-Strict"
DATASET_NAME="BabyLM-2026-Strict"

mkdir -p "$DATA_DIR"

echo "Downloading $DATASET_REPO -> $DATA_DIR/$DATASET_NAME"
huggingface-cli download "$DATASET_REPO" --repo-type dataset --local-dir "$DATA_DIR/$DATASET_NAME"

echo "Done."
