import argparse
import json
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def iter_training_texts(dataset_dir: Path):
    for txt_file in sorted(dataset_dir.glob("*.train.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield line

    for parquet_file in sorted(dataset_dir.glob("*.train.parquet")):
        df = pd.read_parquet(parquet_file)
        if "text" not in df.columns:
            continue
        for text in df["text"].astype(str):
            text = text.strip()
            if text:
                yield text


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_dataset_dir = repo_root / "gpt2 multi" / "data" / "en_hi_equal"

    parser = argparse.ArgumentParser(
        description="Train a GPT-BERT tokenizer on the local English+Hindi dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_dataset_dir,
        help="Directory containing *.train.txt and/or *.train.parquet files.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32768,
        help="Tokenizer vocab size. Default=32768.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer_en_hi_vs32768.json",
        help="Output filename relative to this folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [
        f"<special_{i}>" for i in range(11)
    ]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)

    texts = list(iter_training_texts(dataset_dir))
    if not texts:
        raise FileNotFoundError(
            f"No training texts found in {dataset_dir} (.train.txt or .train.parquet)."
        )

    print(f"Training tokenizer on {len(texts):,} text segments from {dataset_dir}")
    tokenizer.train_from_iterator(texts, trainer)

    script_dir = Path(__file__).resolve().parent
    tok_path = script_dir / args.output
    tokenizer.save(str(tok_path))

    meta = {
        "dataset_dir": str(dataset_dir),
        "vocab_size": args.vocab_size,
        "output": args.output,
        "special_tokens": special_tokens,
    }
    with open(script_dir / "tokenizer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved tokenizer to {tok_path}")


if __name__ == "__main__":
    main()
