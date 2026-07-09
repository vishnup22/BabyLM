"""Convert a bilingual GPT-BERT checkpoint (en-hi or en-tel) to HuggingFace format and upload.

Same procedure as gpt-bert/convert_gptbert_to_hf.py (monolingual), pointed at the
bilingual training layout under Babylm2026/eng-{hin,tel}/gptbert multi/.

Run from repo root on the cluster:

    # English-Hindi, seed 1
    python convert_gptbert_bilingual_to_hf.py --pair hi --seed 1 \
        --repo pulipakav-1/en-hi-gptbert-seed1

    # English-Telugu, seed 2
    python convert_gptbert_bilingual_to_hf.py --pair tel --seed 2 \
        --repo pulipakav-1/en-tel-gptbert-seed2
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "gpt-bert"))
from configuration_gptbert import GptBertConfig
from modeling_gptbert import GptBertForMaskedLM

PAIR_DIRS = {
    "hi":  REPO_ROOT / "Babylm2026/eng-hin/gptbert multi",
    "tel": REPO_ROOT / "Babylm2026/eng-tel/gptbert multi",
}
NAME_PREFIX = {
    "hi":  "en-hi-gptbert-seed",
    "tel": "en-tel-gptbert-seed",
}
TOKENIZER_NAME = {
    "hi":  "tokenizer_en_hi_vs32768.json",
    "tel": "tokenizer_en_tel_vs32768.json",
}

VARIANT_SUFFIX = {
    "ema":        "_ema.bin",
    "state_dict": "_state_dict.bin",
    "plain":      ".bin",
}


def convert(pair: str, seed: int, variant: str, repo: str):
    pair_dir = PAIR_DIRS[pair]

    # ── 1. Load config ────────────────────────────────────────────────────────
    config_path = pair_dir / "configs" / "multilingual.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = GptBertConfig(**cfg_dict)
    print(f"Config loaded from {config_path}")

    # ── 2. Load weights ───────────────────────────────────────────────────────
    ckpt_name = f"{NAME_PREFIX[pair]}{seed}{VARIANT_SUFFIX[variant]}"
    ckpt_path = pair_dir / "model_checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded weights: {ckpt_path.name}  ({len(state_dict)} tensors)")

    # ── 3. Build model and load weights ──────────────────────────────────────
    model = GptBertForMaskedLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  WARNING — missing keys : {missing}")
    if unexpected:
        print(f"  WARNING — unexpected   : {unexpected}")
    print("State dict loaded successfully.")

    # ── 4. Save in HF format ─────────────────────────────────────────────────
    out_dir = REPO_ROOT / "gpt-bert" / "hf_converted" / repo.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    config.save_pretrained(out_dir)

    config_json_path = out_dir / "config.json"
    with open(config_json_path) as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig":           "configuration_gptbert.GptBertConfig",
        "AutoModel":            "modeling_gptbert.GptBertModel",
        "AutoModelForMaskedLM": "modeling_gptbert.GptBertForMaskedLM",
    }
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print("  auto_map added to config.json")

    for fname in ["modeling_gptbert.py", "configuration_gptbert.py"]:
        shutil.copy(REPO_ROOT / "gpt-bert" / fname, out_dir / fname)
    print("  Copied modeling_gptbert.py and configuration_gptbert.py")

    tok_path = pair_dir / "tokenizers" / TOKENIZER_NAME[pair]
    if tok_path.exists():
        shutil.copy(tok_path, out_dir / "tokenizer.json")
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
    print(f"Done -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=["hi", "tel"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=list(VARIANT_SUFFIX), default="ema")
    parser.add_argument("--repo", required=True, help="e.g. pulipakav-1/en-hi-gptbert-seed1")
    args = parser.parse_args()
    convert(args.pair, args.seed, args.variant, args.repo)
