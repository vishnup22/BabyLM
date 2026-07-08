#!/bin/bash
#SBATCH --job-name=eval-small-all
#SBATCH --partition=gpu-week-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/storage1/home/pulipakv/BabyLM/logs/eval_small_all_%j.out
#SBATCH --error=/nfs/storage1/home/pulipakv/BabyLM/logs/eval_small_all_%j.err

source /home/pulipakv/miniconda3/etc/profile.d/conda.sh
conda activate telugu_llm

cd "/nfs/storage1/home/pulipakv/BabyLM/small models"

echo "=== English ==="
python english.py

echo "=== Hindi ==="
python hindi.py

echo "=== Telugu ==="
python telugu.py
