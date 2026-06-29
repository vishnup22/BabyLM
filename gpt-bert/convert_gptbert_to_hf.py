"""
Convert a GPT-BERT EMA checkpoint to HuggingFace format and upload.

Run from gpt-bert/ on the cluster:

  # English seed 1
  python convert_gptbert_to_hf.py \
      --lang english \
      --seed 1 \
      --repo pulipakav-1/english-gptbert-base-seed1

  # Hindi seed 1
  python convert_gptbert_to_hf.py \
      --lang hindi \
      --seed 1 \
      --repo pulipakav-1/hindi-gptbert-base-seed1

  # Telugu seed 1
  python convert_gptbert_to_hf.py \
      --lang telugu \
      --seed 1 \
      --repo pulipakav-1/telugu-gptbert-base-seed1
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

# Make local modules importable
sys.path.insert(0, str(Path(__file__).parent))

from configuration_gptbert import GptBertConfig
from modeling_gptbert import GptBertForMaskedLM


def convert(lang: str, seed: int, repo: str):
    lang_dir = Path(__file__).parent / lang

    # ── 1. Load config ────────────────────────────────────────────────────────
    config_path = lang_dir / "configs" / "base.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = GptBertConfig(**cfg_dict)
    print(f"Config loaded from {config_path}")

    # ── 2. Load EMA weights ───────────────────────────────────────────────────
    ckpt_name = f"{lang}-gptbert-base-seed {seed}_ema.bin"
    ckpt_path = lang_dir / "model_checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded EMA weights: {ckpt_path.name}  ({len(state_dict)} tensors)")

    # ── 3. Build model and load weights ──────────────────────────────────────
    model = GptBertForMaskedLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  WARNING — missing keys : {missing}")
    if unexpected:
        print(f"  WARNING — unexpected   : {unexpected}")
    print("State dict loaded successfully.")

    # ── 4. Save in HF format ─────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "hf_converted" / repo.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    config.save_pretrained(out_dir)

    # Add auto_map so users can load with AutoModel.from_pretrained(..., trust_remote_code=True)
    config_json_path = out_dir / "config.json"
    with open(config_json_path) as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig":         "configuration_gptbert.GptBertConfig",
        "AutoModel":          "modeling_gptbert.GptBertModel",
        "AutoModelForMaskedLM": "modeling_gptbert.GptBertForMaskedLM",
    }
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print("  auto_map added to config.json")

    # Copy model code so it can be loaded with trust_remote_code=True
    for fname in ["modeling_gptbert.py", "configuration_gptbert.py"]:
        shutil.copy(Path(__file__).parent / fname, out_dir / fname)
    print("  Copied modeling_gptbert.py and configuration_gptbert.py")

    # Copy tokenizer
    tok_path = lang_dir / "tokenizers" / "tokenizer_base_16384.json"
    if tok_path.exists():
        shutil.copy(tok_path, out_dir / "tokenizer.json")
        # Write tokenizer_config.json so HF recognizes it
        tok_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "model_max_length": 512,
        }
        with open(out_dir / "tokenizer_config.json", "w") as f:
            json.dump(tok_config, f, indent=2)
        print("  Tokenizer copied + tokenizer_config.json written")
    else:
        print(f"  WARNING: tokenizer not found at {tok_path}")

    print(f"Saved HF model to {out_dir}")

    # ── 5. Upload ─────────────────────────────────────────────────────────────
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nUploading to {repo} ...")
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="model")
    print(f"Done → https://huggingface.co/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["english", "hindi", "telugu"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repo", required=True, help="e.g. pulipakav-1/english-gptbert-base-seed1")
    args = parser.parse_args()
    convert(args.lang, args.seed, args.repo)
