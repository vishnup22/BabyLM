#!/bin/bash
# Submits all GPT-BERT eval jobs as separate SLURM jobs (parallel, one GPU each).
# Skips: eng-hi seed2, eng-tel seed1.

set -eo pipefail

sbatch eval_gptbert_mono_en.sh
sbatch eval_gptbert_mono_hi.sh
sbatch eval_gptbert_mono_te.sh
sbatch eval_gptbert_en_hi_seed1.sh
sbatch eval_gptbert_en_tel_seed2.sh
