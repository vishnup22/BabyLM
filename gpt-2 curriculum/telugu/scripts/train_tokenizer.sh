#!/usr/bin/env bash
set -eo pipefail

python train_tokenizer.py translated-babylm-telugu --curriculum_file curriculum_data/telugu.txt
