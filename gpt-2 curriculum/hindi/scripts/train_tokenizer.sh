#!/usr/bin/env bash
set -eo pipefail

python train_tokenizer.py translated-babylm-hindi --curriculum_file curriculum_data/hindi.txt
