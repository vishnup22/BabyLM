"""
Convert raw GPT-2 state-dict checkpoints to Hugging Face format.

Usage:
  # Convert a single checkpoint
  python convert_to_hf.py \
    --checkpoint /path/to/experiments/english-strict-100m-seed1/checkpoints/epoch_0/latest_student.pt \
    --config    /path/to/gpt-2/english/configs/BabyLM-2026-Strict \
    --output    /path/to/hf_checkpoints/epoch_0

  # Convert ALL checkpoints in an experiment folder
  python convert_to_hf.py \
    --experiment /path/to/experiments/english-strict-100m-seed1/checkpoints \
    --config     /path/to/gpt-2/english/configs/BabyLM-2026-Strict \
    --output_dir /path/to/hf_checkpoints
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config


def convert_checkpoint(checkpoint_path: str, config_dir: str, output_dir: str):
    print(f"Converting {checkpoint_path} -> {output_dir}")

    config = GPT2Config.from_pretrained(config_dir)
    model = GPT2LMHeadModel(config)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}")


def convert_experiment(experiment_dir: str, config_dir: str, output_dir: str):
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)

    checkpoint_folders = sorted(experiment_dir.iterdir())
    print(f"Found {len(checkpoint_folders)} checkpoint folder(s).")

    for folder in checkpoint_folders:
        pt_file = folder / "latest_student.pt"
        if not pt_file.exists():
            print(f"  Skipping {folder.name} (no latest_student.pt)")
            continue
        out = output_dir / folder.name
        convert_checkpoint(str(pt_file), config_dir, str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to a single latest_student.pt file")
    group.add_argument("--experiment", help="Path to the checkpoints/ folder (converts all epochs)")

    parser.add_argument("--config", required=True, help="Path to the folder containing config.json")
    parser.add_argument("--output", help="Output directory for a single checkpoint")
    parser.add_argument("--output_dir", help="Output root directory when converting all checkpoints")

    args = parser.parse_args()

    if args.checkpoint:
        if not args.output:
            parser.error("--output is required when using --checkpoint")
        convert_checkpoint(args.checkpoint, args.config, args.output)
    else:
        if not args.output_dir:
            parser.error("--output_dir is required when using --experiment")
        convert_experiment(args.experiment, args.config, args.output_dir)
