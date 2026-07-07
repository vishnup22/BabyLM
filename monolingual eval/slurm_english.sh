#!/bin/bash
#SBATCH --job-name=eval-english
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nfs/storage1/home/pulipakv/BabyLM/monolingual eval/logs/eval_english_%j.out
#SBATCH --error=/nfs/storage1/home/pulipakv/BabyLM/monolingual eval/logs/eval_english_%j.err

source ~/.bashrc
conda activate telugu_llm

WORKDIR="/nfs/storage1/home/pulipakv/BabyLM/monolingual eval"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

python english.py --cleaned_dir cleaned
