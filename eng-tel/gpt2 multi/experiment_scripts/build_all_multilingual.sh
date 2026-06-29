#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="build_multilingual_dataset.py"

# English-Telugu: 46M English (all available) + 54M Telugu = 100M total
$PYTHON $SCRIPT --languages en te --lang-words en=46000000 te=54000000 --output data/en_tel_equal

echo ""
echo "Done."
