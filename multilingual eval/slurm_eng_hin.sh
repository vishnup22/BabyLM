#!/bin/bash
#SBATCH --job-name=eval-enghin
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/storage1/home/pulipakv/BabyLM/logs/eval_enghin_%j.out
#SBATCH --error=/nfs/storage1/home/pulipakv/BabyLM/logs/eval_enghin_%j.err

source /home/pulipakv/miniconda3/etc/profile.d/conda.sh
conda activate telugu_llm

cd "/nfs/storage1/home/pulipakv/BabyLM/multilingual eval"

python eng_hin.py \
    --cleaned_dir "/nfs/storage1/home/pulipakv/BabyLM/monolingual eval/cleaned"
