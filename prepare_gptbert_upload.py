"""
Prepare a GPT-BERT checkpoint folder for HuggingFace upload, then upload it.

Run this on the HPC cluster from the repo root:

  python prepare_gptbert_upload.py \
      --lang english \
      --seed 1 \
      --repo pulipakav-1/english-gptbert-base-seed1

For bilingual:
  python prepare_gptbert_upload.py \
      --lang en_hi \
      --seed 1 \
      --repo pulipakav-1/en-hi-gptbert-bilingual-seed1 \
      --base-dir "gptbert multi"
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi

GPTBERT_LANGS = ["english", "hindi", "telugu"]


def prepare_upload_folder(lang: str, seed: int, base_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. EMA weights → pytorch_model.bin
    ckpt_name = f"{lang}-gptbert-base-seed {seed}_ema.bin"
    ckpt_path = base_dir / "model_checkpoints" / ckpt_name
    if not ckpt_path.exists():
        # fallback to state_dict
        ckpt_name = f"{lang}-gptbert-base-seed {seed}_state_dict.bin"
        ckpt_path = base_dir / "model_checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {base_dir / 'model_checkpoints'}")

    print(f"  Copying weights: {ckpt_path.name} -> pytorch_model.bin")
    shutil.copy(ckpt_path, output_dir / "pytorch_model.bin")

    # 2. config.json (base.json + model_type field)
    config_path = base_dir / "configs" / "base.json"
    with open(config_path) as f:
        config = json.load(f)
    config["model_type"] = "gptbert"
    config["architectures"] = ["Bert"]
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Written config.json")

    # 3. model code (needed for trust_remote_code loading)
    model_src = base_dir / "pretraining" / "model_extra.py"
    shutil.copy(model_src, output_dir / "modeling_gptbert.py")
    print("  Copied modeling_gptbert.py")

    # 4. tokenizer
    tok_path = base_dir / "tokenizers" / "tokenizer_base_16384.json"
    if tok_path.exists():
        shutil.copy(tok_path, output_dir / "tokenizer.json")
        print("  Copied tokenizer.json")
    else:
        print(f"  WARNING: tokenizer not found at {tok_path}")

    print(f"  Upload folder ready: {output_dir}")


def upload(output_dir: Path, repo_id: str, batch_size: int = 5):
    api = HfApi()
    print(f"\nCreating repo '{repo_id}' (skipped if exists)...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    files = [(p, str(p.relative_to(output_dir))) for p in output_dir.rglob("*") if p.is_file()]
    print(f"Uploading {len(files)} files in batches of {batch_size}...")

    failed = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        for full_path, rel_path in batch:
            try:
                api.upload_file(
                    path_or_fileobj=str(full_path),
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  ✓ {rel_path}")
            except Exception as e:
                print(f"  ✗ {rel_path} — {e}")
                failed.append((full_path, rel_path))

    for full_path, rel_path in failed:
        try:
            api.upload_file(str(full_path), rel_path, repo_id=repo_id, repo_type="model")
            print(f"  ✓ {rel_path} (retry)")
        except Exception as e:
            print(f"  ✗ {rel_path} — gave up: {e}")

    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",     required=True, help="e.g. english, hindi, telugu, en_hi")
    parser.add_argument("--seed",     type=int, required=True)
    parser.add_argument("--repo",     required=True, help="HF repo id, e.g. pulipakav-1/english-gptbert-base-seed1")
    parser.add_argument("--base-dir", default=None, help="Override base dir (default: gpt-bert/<lang>)")
    parser.add_argument("--output",   default="hf_upload_tmp", help="Temp folder for prepared files")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    if args.base_dir:
        base_dir = Path(args.base_dir)
    elif args.lang in GPTBERT_LANGS:
        base_dir = repo_root / "gpt-bert" / args.lang
    else:
        base_dir = repo_root / "gptbert multi"

    output_dir = repo_root / args.output / args.repo.split("/")[-1]

    print(f"=== Preparing {args.lang} seed{args.seed} ===")
    print(f"  Source : {base_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Repo   : {args.repo}")

    prepare_upload_folder(args.lang, args.seed, base_dir, output_dir)
    upload(output_dir, args.repo, args.batch_size)


if __name__ == "__main__":
    main()
