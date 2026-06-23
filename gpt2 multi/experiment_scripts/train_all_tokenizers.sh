#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
SCRIPT="train_tokenizer.py"

DATASETS=(
    # Monolingual
    BabyLM-2026-Strict
    BabyLM-2026-Strict-Small
    translated-babylm-hindi
    babylm-nld
    babylm-zho
    # Multilingual
    en_hi_equal
    en_nld_equal
    en_zho_equal
    nld_zho_equal
    en_nld_zho_equal
)

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Training tokenizer for: $ds"
    echo "=========================================="
    if [[ "$ds" == "en_hi_equal" ]]; then
        $PYTHON $SCRIPT "$ds" --vocab_size 32768
    else
        $PYTHON $SCRIPT "$ds"
    fi
done

echo ""
echo "All tokenizers trained."
