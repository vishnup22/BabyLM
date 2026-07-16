"""
Convert the final GPT-2 checkpoint to HuggingFace format and upload.

Run from gpt-2/hindi_native/ (or gpt-2/telugu_native/) on the cluster:

  python push_to_hf.py --dataset native-babylm-hindi --experiment hindi-native-strict-100m --repo pulipakav-1/hindi-native-gpt2-base
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


def find_latest_checkpoint(experiment_name: str) -> Path:
    checkpoint_dir = Path("experiments") / experiment_name / "checkpoints"
    epoch_dirs = sorted(
        checkpoint_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* checkpoints found under {checkpoint_dir}")
    latest = epoch_dirs[-1]
    ckpt_path = latest / "latest_student.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected checkpoint file not found: {ckpt_path}")
    print(f"Using latest checkpoint: {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. native-babylm-hindi")
    parser.add_argument("--experiment", required=True, help="e.g. hindi-native-strict-100m")
    parser.add_argument("--repo", required=True, help="e.g. pulipakav-1/hindi-native-gpt2-base")
    args = parser.parse_args()

    config = GPT2Config.from_pretrained(f"configs/{args.dataset}")
    model = GPT2LMHeadModel(config)

    ckpt_path = find_latest_checkpoint(args.experiment)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  WARNING -- missing keys: {missing}")
    if unexpected:
        print(f"  WARNING -- unexpected keys: {unexpected}")

    out_dir = Path("hf_converted") / args.repo.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(f"tokenizers/{args.dataset}")
    tokenizer.save_pretrained(out_dir)

    print(f"Saved HF model + tokenizer to {out_dir}")

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Uploading to {args.repo} ...")
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=args.repo, repo_type="model")
    print("Done.")


if __name__ == "__main__":
    main()
