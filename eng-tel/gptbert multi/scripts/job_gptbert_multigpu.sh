#!/bin/bash
#SBATCH --job-name=eng-tel-gptbert
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/eng_tel_gptbert_%j.out
#SBATCH --error=logs/eng_tel_gptbert_%j.err

set -eo pipefail

cd /nfs/storage1/home/pulipakv/BabyLM/eng-tel/gptbert\ multi
mkdir -p logs

eval "$($(which conda) shell.bash hook)"
conda activate telugu_llm

export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1

for SEED in 1 2; do
  export MASTER_PORT=$((29500 + SEED))

  NAME="en-tel-gptbert-seed${SEED}" \
  N_GPUS=4 \
  bash scripts/run_train_multigpu.sh --seed "$SEED"
done
