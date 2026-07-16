#!/bin/bash
#SBATCH --job-name=babylm-tel-native-combined
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/telugu_native_combined_%j.out
#SBATCH --error=logs/telugu_native_combined_%j.err

# Trains native (non-translated, CC-100) Telugu models sequentially in one job:
#   [1/2] GPT-BERT (gpt-bert/telugu_native)
#   [2/2] GPT-2 / GPT-Wee (gpt-2/telugu_native)
# Each stage pushes its own checkpoint to HF as soon as it finishes (see each
# bundle's scripts/run_all.sh), so partial progress is never stranded even if
# the second stage fails or the job runs out of time.
#
# Submit from the repo root: HF_TOKEN=<token> sbatch train_telugu_native.sh

set -eo pipefail

REPO_ROOT="/nfs/storage1/home/pulipakv/BabyLM"
mkdir -p "$REPO_ROOT/logs"

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before submitting}"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== [1/2] Training GPT-BERT (Telugu, native) ====="
cd "$REPO_ROOT/gpt-bert/telugu_native"
mkdir -p logs
export MASTER_PORT=29502
export N_GPUS=4
bash scripts/run_all.sh

echo "===== [2/2] Training GPT-2 (Telugu, native) ====="
cd "$REPO_ROOT/gpt-2/telugu_native"
mkdir -p logs
export MASTER_PORT=29602
bash scripts/run_all.sh

echo "===== Done: GPT-BERT and GPT-2 (Telugu, native) both trained and pushed to HF ====="
