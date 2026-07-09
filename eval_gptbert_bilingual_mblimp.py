"""M-BLiMP (Hindi) evaluation for locally-trained bilingual (en-hi) GPT-BERT checkpoints.

Loads raw .bin state_dict checkpoints (not an HF hub repo) the same way
gpt-bert/convert_gptbert_to_hf.py does: build GptBertForMaskedLM from the
training config, then model.load_state_dict(...).

Usage (run from repo root on the cluster):
    python eval_gptbert_bilingual_mblimp.py
    python eval_gptbert_bilingual_mblimp.py --seeds 1 2 --variant ema
    python eval_gptbert_bilingual_mblimp.py --checkpoint-dir "Babylm2026/eng-hin/gptbert multi/model_checkpoints" \
        --name-prefix en-hi-gptbert-seed
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "gpt-bert"))
from configuration_gptbert import GptBertConfig
from modeling_gptbert import GptBertForMaskedLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG      = REPO_ROOT / "Babylm2026/eng-hin/gptbert multi/configs/multilingual.json"
DEFAULT_TOKENIZER   = REPO_ROOT / "Babylm2026/eng-hin/gptbert multi/tokenizers/tokenizer_en_hi_vs32768.json"
DEFAULT_CKPT_DIR    = REPO_ROOT / "Babylm2026/eng-hin/gptbert multi/model_checkpoints"
DEFAULT_NAME_PREFIX = "en-hi-gptbert-seed"

VARIANT_SUFFIX = {
    "ema":         "_ema.bin",
    "state_dict":  "_state_dict.bin",
    "plain":       ".bin",
}


@torch.no_grad()
def score_gptbert_pll(model, token_ids, cls_id, mask_id):
    """Pseudo-log-likelihood, same convention as monolingual eval/hindi.py::score_gptbert_pll."""
    full_ids = [cls_id] + token_ids
    seq_len  = len(full_ids)
    n        = seq_len - 1
    if n == 0:
        return float("-inf")
    base     = torch.tensor(full_ids, dtype=torch.long, device=DEVICE)
    input_t  = base.unsqueeze(1).expand(-1, n).clone()
    labels_t = torch.full((seq_len, n), -100, dtype=torch.long, device=DEVICE)
    for j in range(n):
        pos = j + 1
        input_t[pos, j]  = mask_id
        labels_t[pos, j] = base[pos]
    attn = torch.zeros(n, 1, seq_len, seq_len, dtype=torch.bool, device=DEVICE)
    static_emb, rel_emb = model.embedding(input_t)
    hidden               = model.transformer(static_emb, attn, rel_emb)
    logits               = model.classifier(hidden, labels_t)
    log_probs = F.log_softmax(logits, dim=-1)
    gold      = base[1:].to(DEVICE)
    return log_probs[torch.arange(n, device=DEVICE), gold].sum().item()


def load_checkpoint(config_path, tokenizer_path, ckpt_path):
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = GptBertConfig(**cfg_dict)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model = GptBertForMaskedLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  WARNING — missing keys: {missing}")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected}")
    model = model.eval().to(DEVICE)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    cls_id  = tokenizer.convert_tokens_to_ids("<s>")
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    return model, tokenizer, cls_id, mask_id


def eval_mblimp(model, tokenizer, cls_id, mask_id):
    ds = load_dataset("jumelet/multiblimp", "hin", split="train")
    correct = 0
    for row in tqdm(ds, desc="  M-BLiMP", leave=False):
        good_ids = tokenizer.encode(row["sen"], add_special_tokens=False)
        bad_ids  = tokenizer.encode(row["wrong_sen"], add_special_tokens=False)
        good = score_gptbert_pll(model, good_ids, cls_id, mask_id)
        bad  = score_gptbert_pll(model, bad_ids, cls_id, mask_id)
        if good > bad:
            correct += 1
    return correct / len(ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--variant", choices=list(VARIANT_SUFFIX), default="ema")
    args = parser.parse_args()

    results = {}
    for seed in args.seeds:
        ckpt_name = f"{args.name_prefix}{seed}{VARIANT_SUFFIX[args.variant]}"
        ckpt_path = args.checkpoint_dir / ckpt_name
        print(f"\n=== seed {seed} ({ckpt_path.name}) ===")
        if not ckpt_path.exists():
            print(f"  SKIP — checkpoint not found: {ckpt_path}")
            continue

        model, tokenizer, cls_id, mask_id = load_checkpoint(args.config, args.tokenizer, ckpt_path)
        acc = eval_mblimp(model, tokenizer, cls_id, mask_id)
        results[seed] = round(acc, 4)
        print(f"  M-BLiMP (hin): {acc:.4f}")

        del model
        torch.cuda.empty_cache()

    print("\n=== M-BLiMP Results (en-hi-gptbert) ===")
    for seed, acc in results.items():
        print(f"  seed {seed}: {acc}")
    if results:
        mean_acc = sum(results.values()) / len(results)
        print(f"  mean:   {mean_acc:.4f}")


if __name__ == "__main__":
    main()
