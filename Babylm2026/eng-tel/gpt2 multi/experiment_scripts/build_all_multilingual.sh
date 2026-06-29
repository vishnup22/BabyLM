#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="build_multilingual_dataset.py"

# English: 46M (all available from 4 sources)
# Telugu:  42.5M (all available from childes + gutenberg)
# Total:   ~88.5M
$PYTHON $SCRIPT --languages en te --lang-words en=50000000 te=50000000 --output data/en_tel_equal

echo ""
echo "Done."
