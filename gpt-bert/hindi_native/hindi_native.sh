#!/bin/bash
#SBATCH --job-name=babylm-hin-native-gptbert
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/hindi_native_%j.out
#SBATCH --error=logs/hindi_native_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM/gpt-bert/hindi_native
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# needed by scripts/download_dataset.sh (pulipakav-1/hi-te) and the final HF push in run_all.sh
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before submitting}"

bash scripts/run_all.sh
