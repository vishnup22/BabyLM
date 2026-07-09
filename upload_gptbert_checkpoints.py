"""Upload raw GPT-BERT bilingual checkpoint files (.bin) to a Hugging Face repo as-is.

No conversion, no HF model format, no config.json -- just pushes the checkpoint
directory's files straight to the Hub.

Run from repo root on the cluster:

    python upload_gptbert_checkpoints.py --pair hi  --repo pulipakav-1/en-hi-gptbert-checkpoints
    python upload_gptbert_checkpoints.py --pair tel --repo pulipakav-1/en-tel-gptbert-checkpoints
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).parent

PAIR_DIRS = {
    "hi":  REPO_ROOT / "Babylm2026/eng-hin/gptbert multi/model_checkpoints",
    "tel": REPO_ROOT / "Babylm2026/eng-tel/gptbert multi/model_checkpoints",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=["hi", "tel"])
    parser.add_argument("--repo", required=True, help="e.g. pulipakav-1/en-hi-gptbert-checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                         help="Override the default checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir or PAIR_DIRS[args.pair]
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    files = sorted(ckpt_dir.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"No .bin files found in {ckpt_dir}")
    print(f"Found {len(files)} checkpoint file(s) in {ckpt_dir}:")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    print(f"\nUploading to {args.repo} ...")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=["*.bin"],
    )
    print(f"Done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
