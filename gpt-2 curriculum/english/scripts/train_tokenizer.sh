#!/usr/bin/env bash
set -eo pipefail

python train_tokenizer.py BabyLM-2026-Strict --curriculum_file curriculum_data/english.txt
