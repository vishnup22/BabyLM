#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
DATASET_REPO="pulipakav-1/translated-babylm-hindi"
DATASET_NAME="translated-babylm-hindi"

mkdir -p "$DATA_DIR"

echo "Downloading $DATASET_REPO -> $DATA_DIR/$DATASET_NAME"
huggingface-cli download "$DATASET_REPO" --repo-type dataset --local-dir "$DATA_DIR/$DATASET_NAME"

echo "Done."
