import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def resolve_input_dir(dataset_name: str, data_root: Path) -> Path:
    base_dir = data_root / dataset_name
    train_dir = base_dir / "train"

    base_has_files = any(base_dir.glob("*.train*.txt"))
    train_has_files = any(train_dir.glob("*.train*.txt"))

    if base_has_files:
        return base_dir
    if train_has_files:
        return train_dir
    return base_dir


def train_tokenizer(dataset_name: str, data_root: Path, output_path: Path, vocab_size: int) -> None:
    input_dir = resolve_input_dir(dataset_name, data_root)
    files = sorted(str(f) for f in input_dir.glob("*.train*.txt"))
    if not files:
        raise FileNotFoundError(f"No .train*.txt files found in {input_dir}")

    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    print(f"Training tokenizer on {len(files)} files from {input_dir}")
    tokenizer.train(files, trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    meta = {
        "dataset": dataset_name,
        "input_dir": str(input_dir),
        "vocab_size": vocab_size,
        "files": files,
        "special_tokens": special_tokens,
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved tokenizer to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=16384)
    args = parser.parse_args()
    train_tokenizer(args.dataset, args.data_root, args.output, args.vocab_size)
